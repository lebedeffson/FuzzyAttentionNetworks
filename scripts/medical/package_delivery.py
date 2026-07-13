#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def _run_completed(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_tree(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git_or_value(cmd: list[str], fallback: str) -> str:
    try:
        return _run(cmd)
    except Exception:
        return fallback


def build_delivery(out_dir: Path, commit: str | None = None, branch: str | None = None, run_tests: bool = False) -> tuple[Path, Path]:
    commit = commit or _git_or_value(["git", "rev-parse", "HEAD"], "unknown")
    branch = branch or _git_or_value(["git", "branch", "--show-current"], "unknown")
    commit8 = commit[:8]
    name = f"Med_CircuitBench_SCTC_DELIVERY_V4_{date.today().isoformat()}_{commit8}"
    staging = out_dir / name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    _write(staging / "README_FIRST.txt", "Read DELIVERY_REPORT.md first. This is a V4 lightweight package without raw activations or PhysioNet data.\n")
    _write(
        staging / "DELIVERY_REPORT.md",
        "\n".join(
            [
                "# Delivery Report",
                "",
                f"Commit: {commit}",
                f"Branch: {branch}",
                "Status: V4 code package. Full multi-seed scientific GO requires run_benchmark_pipeline validation/test artifacts.",
                "",
                "V4 changes included: threshold 0.0715, no fallback, measured CLS, h-to-a SCTC, measured fidelity primitives, layer-aware CircuitF1, delivery checksums.",
                "",
                "Included: source, configs, tests, docs, package metadata, current lightweight metrics when present.",
                "Excluded: raw activation Zarr chunks, virtual environments, PhysioNet raw data, secrets.",
            ]
        )
        + "\n",
    )
    _write(staging / "GIT_INFO.txt", f"branch={branch}\ncommit={commit}\n")

    source = staging / "SOURCE"
    _copy_tree(ROOT / "src" / "med_circuitbench", source / "src" / "med_circuitbench")
    _copy_tree(ROOT / "scripts" / "medical", source / "scripts" / "medical")
    _copy_tree(ROOT / "configs" / "medical", source / "configs" / "medical")
    _copy_tree(ROOT / "configs" / "manifests", source / "configs" / "manifests")
    _copy_tree(ROOT / "tests" / "medical", source / "tests" / "medical")
    _copy_tree(ROOT / "docs" / "medical", source / "docs" / "medical")
    for filename in ("requirements.txt", "requirements-lock.txt"):
        if (ROOT / filename).exists():
            shutil.copy2(ROOT / filename, source / filename)
    _write(source / "git_patch.diff", _git_or_value(["git", "diff", "HEAD"], ""))

    env = staging / "ENVIRONMENT"
    _write(env / "python_version.txt", sys.version + "\n")
    _write(env / "system_info.txt", f"platform={platform.platform()}\n")
    try:
        _write(env / "pip_freeze.txt", _run([sys.executable, "-m", "pip", "freeze"]) + "\n")
    except Exception as exc:
        _write(env / "pip_freeze.txt", f"pip freeze failed: {exc}\n")
    try:
        _write(env / "gpu_info.txt", _run(["nvidia-smi"]) + "\n")
    except Exception as exc:
        _write(env / "gpu_info.txt", f"nvidia-smi unavailable: {exc}\n")
    _write(env / "environment.json", json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "python": sys.version, "platform": platform.platform()}, indent=2))

    tests = staging / "TESTS"
    if run_tests:
        started = datetime.now(timezone.utc)
        result = _run_completed([sys.executable, "-m", "pytest", "tests/medical", "-q", "--disable-warnings"])
        ended = datetime.now(timezone.utc)
        _write(tests / "pytest_output.txt", result.stdout)
        _write(tests / "pytest_stderr.txt", result.stderr)
        _write(
            tests / "test_summary.json",
            json.dumps(
                {
                    "status": "PASS" if result.returncode == 0 else "FAIL",
                    "command": f"{sys.executable} -m pytest tests/medical -q --disable-warnings",
                    "exit_code": result.returncode,
                    "started_at": started.isoformat(),
                    "ended_at": ended.isoformat(),
                    "duration_seconds": (ended - started).total_seconds(),
                },
                indent=2,
            ),
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    else:
        _write(tests / "pytest_output.txt", "Not run by package script. Pass --run-tests to execute.\n")
        _write(tests / "pytest_stderr.txt", "")
        _write(tests / "test_summary.json", json.dumps({"status": "NOT_RUN", "command": None, "exit_code": None}, indent=2))
    _write(tests / "end_to_end_smoke_output.txt", "Smoke command documented in SOURCE/docs/medical/README_RUN.md\n")
    _write(tests / "fixtures_report.md", "Fixtures are included under SOURCE/tests/medical/fixtures.\n")

    results = staging / "RESULTS"
    _write(results / "go_no_go_validation.json", json.dumps({"status": "PENDING_FULL_VALIDATION", "reason": "multi_seed_pipeline_required"}, indent=2))
    _write(results / "overall_status.json", json.dumps({"status": "PENDING_FULL_VALIDATION"}, indent=2))
    _write(results / "result_summary.md", "V4 code package built. Full scientific GO/NO-GO requires registered multi-seed validation/test runs.\n")
    for name_csv in ("final_metrics.csv", "benchmark_comparison.csv", "fidelity_summary.csv", "stability_summary.csv", "bootstrap_intervals.csv", "timing_summary.csv"):
        _write(results / name_csv, "metric,value\nstatus,PENDING_FULL_VALIDATION\n")

    tables_src = ROOT / "artifacts" / "medical" / "article" / "tables"
    figures_src = ROOT / "artifacts" / "medical" / "article" / "figures"
    _copy_tree(tables_src, staging / "TABLES")
    if figures_src.exists():
        png = staging / "FIGURES" / "PNG"
        pdf = staging / "FIGURES" / "PDF"
        png.mkdir(parents=True, exist_ok=True)
        pdf.mkdir(parents=True, exist_ok=True)
        for fig in figures_src.glob("*.png"):
            shutil.copy2(fig, png / fig.name)

    _write(staging / "LIMITATIONS" / "known_issues.md", (ROOT / "docs" / "medical" / "KNOWN_LIMITATIONS.md").read_text(encoding="utf-8"))
    _write(staging / "LIMITATIONS" / "failed_runs.md", "Full V4 validation/test is not executed by package_delivery.py.\n")
    _write(staging / "LIMITATIONS" / "deviations_from_tz.md", "See DELIVERY_REPORT.md. No PhysioNet raw data included.\n")

    manifests = staging / "MANIFESTS"
    manifests.mkdir(parents=True, exist_ok=True)
    if (ROOT / "configs" / "manifests" / "article.yaml").exists():
        shutil.copy2(ROOT / "configs" / "manifests" / "article.yaml", manifests / "article_manifest.yaml")
    checksums = []
    for file in sorted(p for p in staging.rglob("*") if p.is_file()):
        rel = file.relative_to(staging)
        if rel == Path("MANIFESTS/checksums.sha256"):
            continue
        checksums.append(f"{_sha256(file)}  {rel.as_posix()}")
    _write(manifests / "checksums.sha256", "\n".join(checksums) + "\n")
    _write(
        manifests / "delivery_manifest.json",
        json.dumps({"name": name, "commit": commit, "created_at": datetime.now(timezone.utc).isoformat(), "files": len(checksums)}, indent=2),
    )

    zip_path = out_dir / f"{name}.zip"
    sha_path = out_dir / f"{name}.zip.sha256"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(p for p in staging.rglob("*") if p.is_file()):
            zf.write(file, file.relative_to(out_dir))
    _write(sha_path, f"{_sha256(zip_path)}  {zip_path.name}\n")
    return zip_path, sha_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "medical")
    parser.add_argument("--commit")
    parser.add_argument("--branch")
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--include-results", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    zip_path, sha_path = build_delivery(args.out_dir, commit=args.commit, branch=args.branch, run_tests=args.run_tests)
    print({"zip": str(zip_path), "sha256": str(sha_path), "size_bytes": zip_path.stat().st_size})


if __name__ == "__main__":
    main()
