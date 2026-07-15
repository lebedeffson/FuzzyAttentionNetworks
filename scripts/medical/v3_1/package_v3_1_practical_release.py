#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copytree_filtered(src: Path, dst: Path) -> None:
    def ignore(_dir, names):
        banned = {"__pycache__", ".pytest_cache"}
        return [n for n in names if n in banned or n.endswith(".zip")]

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def write_sitecustomize(release: Path) -> None:
    (release / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "ROOT = Path(__file__).resolve().parent\n"
        "for p in [ROOT / 'SOURCE' / 'src', ROOT / 'SOURCE']:\n"
        "    if str(p) not in sys.path:\n"
        "        sys.path.insert(0, str(p))\n",
        encoding="utf-8",
    )
    (release / "pytest.ini").write_text("[pytest]\ntestpaths = TESTS\n", encoding="utf-8")
    (release / "TESTS" / "conftest.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "ROOT = Path(__file__).resolve().parents[1]\n"
        "for p in [ROOT / 'SOURCE' / 'src', ROOT / 'SOURCE']:\n"
        "    if str(p) not in sys.path:\n"
        "        sys.path.insert(0, str(p))\n",
        encoding="utf-8",
    )


def repair_fan_bundle_metrics(release: Path) -> None:
    baseline_path = ROOT / "artifacts" / "medical" / "v3_real_final" / "results" / "fan_validation_metrics.csv"
    if not baseline_path.exists():
        return
    baseline = pd.read_csv(baseline_path)
    for manifest_path in (release / "BUNDLES" / "fan_noalpha").glob("seed*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seed = int(manifest["seed"])
        row = baseline[baseline["seed"].astype(int).eq(seed)].iloc[0].to_dict()
        manifest["model_name"] = "FAN-NoAlpha"
        manifest["saved_metrics"] = {
            "seed": seed,
            "model": "FAN-NoAlpha",
            "source": "baseline_results",
            "AUROC": float(row["AUROC"]),
            "AUPRC": float(row["AUPRC"]),
            "F1": float(row.get("F1", float("nan"))),
            "Brier": float(row.get("Brier", float("nan"))),
            "ECE": float(row.get("ECE", float("nan"))),
            "direct_macro_R2": float(row.get("macro_trajectory_r2", row.get("direct_macro_R2", float("nan")))),
            "macro_Pearson": float(row.get("mean_trajectory_pearson", row.get("macro_Pearson", float("nan")))),
            "concept_MAE": float(row.get("concept_MAE", float("nan"))),
            "concept_delta_MAE": float(row.get("concept_delta_MAE", float("nan"))),
            "oracle_noalpha_AUPRC": float(row.get("oracle_noalpha_AUPRC", float("nan"))),
            "predicted_oracle_ratio": float(row.get("predicted_oracle_ratio", float("nan"))),
        }
        manifest["stable_run_diagnostics"] = {
            "source": "RESULTS/fan_stable_metrics.csv",
            "note": "Stable-run metrics are diagnostics only because production_model is FAN-NoAlpha.",
        }
        manifest["metrics_model"] = "FAN-NoAlpha"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fan-stable-output", default="artifacts/medical/v3_1/fan_noalpha_stable")
    parser.add_argument("--release-root", default="artifacts/medical/v3_1_practical_release")
    parser.add_argument("--zip-output-dir", default="artifacts/medical")
    args = parser.parse_args(argv)
    release = Path(args.release_root)
    if release.exists():
        shutil.rmtree(release)
    for sub in ["SOURCE", "CONFIGS", "TESTS", "RESULTS", "BUNDLES", "MANIFESTS", "LIMITATIONS", "TABLES"]:
        (release / sub).mkdir(parents=True, exist_ok=True)
    copytree_filtered(ROOT / "src" / "fan", release / "SOURCE" / "src" / "fan")
    copytree_filtered(ROOT / "src" / "med_circuitbench", release / "SOURCE" / "src" / "med_circuitbench")
    copytree_filtered(ROOT / "scripts" / "medical" / "v2", release / "SOURCE" / "scripts" / "medical" / "v2")
    copytree_filtered(ROOT / "scripts" / "medical" / "v2_2", release / "SOURCE" / "scripts" / "medical" / "v2_2")
    copytree_filtered(ROOT / "scripts" / "medical" / "v3", release / "SOURCE" / "scripts" / "medical" / "v3")
    copytree_filtered(ROOT / "scripts" / "medical" / "v3_1", release / "SOURCE" / "scripts" / "medical" / "v3_1")
    copytree_filtered(ROOT / "scripts" / "cli", release / "SOURCE" / "scripts" / "cli")
    copytree_filtered(ROOT / "configs" / "medical" / "v3_1", release / "CONFIGS" / "medical" / "v3_1")
    copytree_filtered(ROOT / "configs" / "medical" / "v3", release / "CONFIGS" / "medical" / "v3")
    copytree_filtered(ROOT / "configs" / "medical" / "v2", release / "CONFIGS" / "medical" / "v2")
    copytree_filtered(ROOT / "configs" / "medical" / "v2_2", release / "CONFIGS" / "medical" / "v2_2")
    copytree_filtered(ROOT / "tests" / "medical" / "v2", release / "TESTS" / "medical" / "v2")
    copytree_filtered(ROOT / "tests" / "medical" / "v2_2", release / "TESTS" / "medical" / "v2_2")
    copytree_filtered(ROOT / "tests" / "medical" / "v3", release / "TESTS" / "medical" / "v3")
    for package_init in [
        release / "SOURCE" / "scripts" / "__init__.py",
        release / "SOURCE" / "scripts" / "medical" / "__init__.py",
    ]:
        package_init.parent.mkdir(parents=True, exist_ok=True)
        package_init.write_text("", encoding="utf-8")
    copytree_filtered(release / "SOURCE" / "scripts", release / "scripts")
    copytree_filtered(release / "SOURCE" / "src" / "fan", release / "fan")
    copytree_filtered(release / "SOURCE" / "src" / "med_circuitbench", release / "med_circuitbench")
    copytree_filtered(release / "CONFIGS", release / "configs")
    copytree_filtered(ROOT / "docs" / "medical", release / "docs" / "medical")
    copytree_filtered(ROOT / "artifacts" / "medical" / "v3_real", release / "artifacts" / "medical" / "v3_real")
    copytree_filtered(ROOT / "artifacts" / "medical" / "v3_real_final", release / "artifacts" / "medical" / "v3_real_final")
    real_zip = ROOT / "artifacts" / "medical" / "Med_CircuitBench_V3_REAL_RESEARCH_FINAL.zip"
    if real_zip.exists():
        (release / "artifacts" / "medical").mkdir(parents=True, exist_ok=True)
        shutil.copy2(real_zip, release / "artifacts" / "medical" / real_zip.name)
    write_sitecustomize(release)
    fan_out = ROOT / args.fan_stable_output
    if fan_out.exists():
        for name in [
            "fan_stable_gate.json",
            "fan_stable_metrics.csv",
            "fan_cross_seed_stability.csv",
            "resolved_config.yaml",
        ]:
            if (fan_out / name).exists():
                shutil.copy2(fan_out / name, release / "RESULTS" / name)
        if (fan_out / "bundles").exists():
            copytree_filtered(fan_out / "bundles", release / "BUNDLES")
            repair_fan_bundle_metrics(release)
    for src, dst in [
        (ROOT / "AGENTS.md", release / "MANIFESTS" / "AGENTS.md"),
        (ROOT / "AGENTS.md", release / "AGENTS.md"),
        (ROOT / "docs" / "medical" / "PROJECT_STATE.md", release / "MANIFESTS" / "PROJECT_STATE.md"),
        (ROOT / "docs" / "medical" / "KNOWN_LIMITATIONS.md", release / "LIMITATIONS" / "KNOWN_LIMITATIONS.md"),
    ]:
        if src.exists():
            shutil.copy2(src, dst)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    manifest = {
        "status": "SCIENTIFIC_RESEARCH_COMPLETE",
        "code_commit": commit,
        "git_status_clean": status.strip() == "",
        "standalone_release": "VALIDATED_BY_UNPACKED_TESTS_REQUIRED",
        "production_fan": "FAN-NoAlpha",
        "fan_stable_result": "FAN_NOALPHA_BASELINE_RETAINED",
        "sctc_line_status": [
            "SPARSE_FIDELITY_VALIDATED",
            "MECHANISTIC_RECOVERY_NOT_SEMANTICALLY_SPECIFIC",
            "DECODER_COUPLING_BREAKS_FIDELITY",
        ],
        "papers_included": False,
    }
    (release / "MANIFESTS" / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (release / "TABLES" / "med_circuitbench_result_summary.csv").write_text(
        "\n".join(
            [
                "claim,status,evidence",
                "FAN predictive validation,PASS,RESULTS/fan_stable_gate.json and baseline validation metrics",
                "FAN stable replacement,FAIL,FAN_NOALPHA_BASELINE_RETAINED",
                "SCTC semantic specificity,FAIL,decoder-coupled and concept-control diagnostics",
                "Oracle causal evaluator,PASS,repaired evaluator C0 artifacts",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (release / "README_FIRST.md").write_text(
        "\n".join(
            [
                "# Med-CircuitBench V3.1 Practical Release",
                "",
                "This package contains practical code, configs, tests, results, and frozen bundles.",
                "Papers are intentionally not included in this practical release.",
                "",
                "SCTC development is closed as a mixed/negative research line:",
                "- sparse behavioral fidelity is validated;",
                "- mechanistic recovery is not semantically specific;",
                "- decoder-coupled semantic supervision breaks fidelity and remains non-specific.",
            ]
        ),
        encoding="utf-8",
    )
    zip_dir = Path(args.zip_output_dir)
    zip_dir.mkdir(parents=True, exist_ok=True)
    short = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    zip_path = zip_dir / f"Med_CircuitBench_V3_1_RESEARCH_COMPLETE_{short}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(release.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(release))
    sha = sha256_file(zip_path)
    (zip_path.with_suffix(zip_path.suffix + ".sha256")).write_text(f"{sha}  {zip_path.name}\n", encoding="utf-8")
    print(json.dumps({"zip": str(zip_path), "sha256": sha, "size_bytes": zip_path.stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
