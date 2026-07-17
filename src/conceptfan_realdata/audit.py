from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import sha256_file
from .models import MODEL_ARMS, STABILITY_ANALYSIS_ARMS


def _check(checks: list[dict], name: str, passed: bool, detail: object) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})


def validate_release(artifacts_root: Path, report_dir: Path, output_path: Path) -> dict:
    checks: list[dict] = []
    raw = json.loads((artifacts_root / "raw_audit" / "audit.json").read_text(encoding="utf-8"))
    _check(checks, "raw audit passes", raw.get("status") == "PHYSIONET2012_RAW_AUDIT_PASS", raw.get("status"))
    _check(checks, "4000 patients audited", raw.get("patients") == 4000, raw.get("patients"))
    prepared = json.loads((artifacts_root / "prepared" / "prepare_manifest.json").read_text(encoding="utf-8"))
    _check(checks, "train-only normalization", prepared.get("normalization_scope") == "train_only", prepared.get("normalization_scope"))
    _check(checks, "no future fill", prepared.get("future_fill_used") is False, prepared.get("future_fill_used"))
    split = json.loads(next((artifacts_root / "splits").glob("split_seed_*.json")).read_text(encoding="utf-8"))
    _check(checks, "split sizes exact", split.get("counts") == {"train": 2400, "validation": 600, "calibration": 400, "test": 600}, split.get("counts"))
    fit = pd.read_parquet(artifacts_root / "TABLES" / "checkpoint_fit_metrics.parquet")
    counts = fit.groupby("model_arm")["run_id"].nunique().to_dict()
    for arm in MODEL_ARMS:
        _check(checks, f"30 completed main runs for {arm}", counts.get(arm) == 30, counts.get(arm, 0))
    _check(checks, "150 main run manifests", fit["run_id"].nunique() == 30 and len(fit) == 150, len(fit))
    additional = list((artifacts_root / "runs" / "channel_ablation").glob("*/*/run_manifest.json"))
    _check(checks, "30 channel-ablation runs", len(additional) == 30, len(additional))
    _check(checks, "180 completed run artifacts", len(fit) + len(additional) == 180, len(fit) + len(additional))
    _check(checks, "all parameter hashes unique within main grid", fit["parameter_sha256"].nunique() == 150, fit["parameter_sha256"].nunique())
    _check(checks, "exact decomposition below 1e-5", float(fit["max_decomposition_error"].max()) < 1e-5, fit["max_decomposition_error"].max())
    stability = pd.read_parquet(artifacts_root / "TABLES" / "stability_pair_summary.parquet")
    for arm in STABILITY_ANALYSIS_ARMS:
        pairs = stability.loc[stability["model_arm"].eq(arm), ["run_a", "run_b"]].drop_duplicates()
        _check(checks, f"435 model pairs for {arm}", len(pairs) == 435, len(pairs))
    sufficiency = pd.read_parquet(artifacts_root / "TABLES" / "exhaustive_32_mask_sufficiency.parquet", columns=["model_arm", "mask_id"])
    for arm in STABILITY_ANALYSIS_ARMS:
        masks = sorted(sufficiency.loc[sufficiency["model_arm"].eq(arm), "mask_id"].unique())
        _check(checks, f"32 masks for {arm}", masks == list(range(32)), len(masks))
    controls = pd.read_parquet(
        artifacts_root / "TABLES" / "sufficiency_controls.parquet", columns=["M", "random_draws"]
    )
    controls_valid = sorted(controls["M"].unique()) == [1, 2, 3, 4, 5] and set(controls["random_draws"]) == {100}
    _check(
        checks,
        "100-draw sufficiency controls M=1..5",
        controls_valid,
        {"M": sorted(controls["M"].unique()), "random_draws": sorted(controls["random_draws"].unique())},
    )
    leakage = pd.read_parquet(artifacts_root / "TABLES" / "residual_signal_diagnostics.parquet")
    _check(checks, "four leakage channel variants", sorted(leakage["channels"].unique()) == ["V", "V+D", "V+M", "V+M+D"], sorted(leakage["channels"].unique()))
    _check(checks, "test never used for selection", all(fit.get("test_used_for_selection", pd.Series(False)) == False), "all false")
    required_tables = [
        "table_cohort", "table_performance", "table_calibration", "table_stability", "table_robustness", "table_leakage", "table_stability_reg"
    ]
    for name in required_tables:
        _check(checks, f"report table {name} CSV/Parquet", (report_dir / "tables" / f"{name}.csv").exists() and (report_dir / "tables" / f"{name}.parquet").exists(), name)
    required_figures = [
        "fig_real_pipeline", "fig_calibration", "fig_stability_distribution", "fig_fuzzy_robustness", "fig_leakage_localization", "fig_stability_pareto"
    ]
    for name in required_figures:
        present = all((report_dir / "figures" / f"{name}.{suffix}").exists() for suffix in ["svg", "pdf", "png"])
        _check(checks, f"figure {name} SVG/PDF/PNG", present, name)
    article_files = [
        "ABSTRACT_PATCH_RU.md", "ABSTRACT_PATCH_EN.md", "METHODS_REAL_DATA_RU.md", "RESULTS_REAL_DATA_RU.md",
        "DISCUSSION_PATCH_RU.md", "LIMITATIONS_PATCH_RU.md", "CONCLUSION_PATCH_RU.md", "TABLE_AND_FIGURE_MAP.md",
        "CLAIMS_ALLOWED.md", "CLAIMS_FORBIDDEN.md",
    ]
    _check(checks, "article update package complete", all((report_dir / "article_update" / name).exists() for name in article_files), len(article_files))
    failed = [item for item in checks if not item["passed"]]
    report = {
        "status": "PHYSIONET2012_READONLY_VALIDATION_PASS" if not failed else "PHYSIONET2012_READONLY_VALIDATION_FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_archive(repo_root: Path, artifacts_root: Path, report_dir: Path, archive_path: Path) -> dict:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    sources: list[tuple[Path, Path]] = []
    for path in sorted(report_dir.rglob("*")):
        if path.is_file() and path.name not in {"archive_validation.json", "final_audit_lite.json"}:
            sources.append((path, Path("reports/physionet2012") / path.relative_to(report_dir)))
    for path in sorted((artifacts_root / "TABLES").glob("*")):
        if path.is_file():
            sources.append((path, Path("artifacts/TABLES") / path.name))
    for directory in ["raw_audit", "splits", "prepared", "concepts", "stability_selection", "audit"]:
        root = artifacts_root / directory
        if root.exists():
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix not in {".npz", ".pt"}:
                    sources.append((path, Path("artifacts") / path.relative_to(artifacts_root)))
    for manifest in sorted((artifacts_root / "runs").rglob("run_manifest.json")):
        sources.append((manifest, Path("artifacts/runs") / manifest.relative_to(artifacts_root / "runs")))
        for name in ["config.yaml", "environment.json", "git_commit.txt", "split_sha256.txt", "preprocessing_sha256.txt", "metrics_val.json", "metrics_calibration.json", "metrics_test.json"]:
            path = manifest.parent / name
            if path.exists():
                sources.append((path, Path("artifacts/runs") / path.relative_to(artifacts_root / "runs")))
    for root_name in ["src/conceptfan_realdata", "configs/physionet2012", "tests/physionet2012"]:
        root = repo_root / root_name
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                sources.append((path, Path(root_name) / path.relative_to(root)))
    for name in ["run_physionet2012_last_run.sh", "run_physionet2012_posthoc.sh", "build_physionet2012_report.sh"]:
        path = repo_root / "scripts" / name
        if path.exists():
            sources.append((path, Path("scripts") / name))
    for name in ["execution_summary.json", "failure_report.json"]:
        path = report_dir / "manifests" / name
        if path.exists():
            sources.append((path, Path("reports/physionet2012/manifests") / name))
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        seen: set[str] = set()
        for source, relative in sources:
            name = relative.as_posix()
            if name not in seen:
                archive.write(source, name)
                seen.add(name)
    with zipfile.ZipFile(archive_path) as archive:
        corrupt = archive.testzip()
        forbidden = [name for name in archive.namelist() if "set-a.zip" in name or "Outcomes-a.txt" in name or name.endswith(".pt")]
        if corrupt or forbidden:
            raise RuntimeError(f"Archive validation failed: corrupt={corrupt}, forbidden={forbidden[:5]}")
        members = len(archive.namelist())
    digest = sha256_file(archive_path)
    sha_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    sha_path.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
    return {"archive": str(archive_path), "sha256": digest, "members": members, "bytes": archive_path.stat().st_size}


def validate_archive(archive_path: Path, output_path: Path) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        corrupt = archive.testzip()
        names = archive.namelist()
        required = [
            "reports/physionet2012/audit/readonly_validation.json",
            "reports/physionet2012/RESULTS_FOR_PAPER.md",
            "artifacts/TABLES/episode_pairwise_stability.parquet",
            "artifacts/TABLES/exhaustive_32_mask_sufficiency.parquet",
            "reports/physionet2012/manifests/execution_summary.json",
            "reports/physionet2012/manifests/failure_report.json",
        ]
        forbidden = [name for name in names if name.endswith(".pt") or "set-a.zip" in name or "Outcomes-a.txt" in name]
    report = {
        "status": "PHYSIONET2012_ARCHIVE_VALIDATION_PASS" if corrupt is None and not forbidden and all(name in names for name in required) else "PHYSIONET2012_ARCHIVE_VALIDATION_FAIL",
        "archive_sha256": sha256_file(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "members": len(names),
        "corrupt_member": corrupt,
        "missing_required": [name for name in required if name not in names],
        "forbidden_members": forbidden,
    }
    report["passed"] = report["status"].endswith("PASS")
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
