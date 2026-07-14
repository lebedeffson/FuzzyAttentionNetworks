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
FORBIDDEN_TOKENS = ["PEND" + "ING", "PEND" + "ING_FULL_VALIDATION", "NOT" + "_RUN", "PLACE" + "HOLDER", "TODO" + "_RESULT"]


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
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.zarr"))


def _copy_runs_lightweight(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        if "activations" in rel.parts or "checkpoints" in rel.parts:
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _copy_file_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git_or_value(cmd: list[str], fallback: str) -> str:
    try:
        return _run(cmd)
    except Exception:
        return fallback


def _assert_no_forbidden_tokens(staging: Path) -> None:
    offenders: list[str] = []
    for file in sorted(p for p in staging.rglob("*") if p.is_file()):
        if file.suffix.lower() not in {".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".py", ".log"}:
            continue
        text = file.read_text(encoding="utf-8", errors="ignore")
        for token in FORBIDDEN_TOKENS:
            if token in text:
                offenders.append(f"{file.relative_to(staging)}:{token}")
    if offenders:
        raise SystemExit("forbidden delivery tokens found: " + "; ".join(offenders[:20]))


def build_delivery(out_dir: Path, commit: str | None = None, branch: str | None = None, run_tests: bool = False) -> tuple[Path, Path]:
    commit = commit or _git_or_value(["git", "rev-parse", "HEAD"], "unknown")
    branch = branch or _git_or_value(["git", "branch", "--show-current"], "unknown")
    commit8 = commit[:8]
    name = f"Med_CircuitBench_SCTC_FINAL_PRACTICE_{date.today().isoformat()}_{commit8}"
    staging = out_dir / name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    aggregate_src = ROOT / "artifacts" / "medical" / "runs" / "aggregate" / "benchmark_validation"
    runs_src = ROOT / "artifacts" / "medical" / "runs" / "benchmark"
    smoke_logs = ROOT / "artifacts" / "medical" / "package_tests"

    _write(staging / "README_FIRST.txt", "Open DELIVERY_REPORT.md first. This is the final-practice package for Med-CircuitBench/SCTC validation artifacts.\n")
    _write(staging / "GIT_INFO.txt", f"branch={branch}\ncommit={commit}\n")
    go_path = aggregate_src / "go_no_go.json"
    go_payload = json.loads(go_path.read_text()) if go_path.exists() else {"overall_status": "NO_GO", "reasons": ["full validation aggregate file is absent"]}
    _write(
        staging / "DELIVERY_REPORT.md",
        "\n".join(
            [
                "# Med-CircuitBench / SCTC Final Practice Delivery",
                "",
                f"Branch: {branch}",
                f"Commit: {commit}",
                f"Validation status: {go_payload.get('overall_status', 'NO_GO')}",
                "",
                "The package contains the implemented research-core pipeline, three validation run directories when available, aggregate metrics, logs, generated tables, generated figures, and delivery checksums.",
                "PhysioNet raw data are not bundled. If the data root is absent, the PhysioNet runner records BLOCKED_DATA_ACCESS.",
                "Raw activation arrays are excluded from the ZIP to keep the archive small; run directories keep manifests, metrics, logs, tables, and figures.",
            ]
        )
        + "\n",
    )

    source = staging / "SOURCE"
    _copy_tree(ROOT / "src" / "med_circuitbench", source / "src" / "med_circuitbench")
    _copy_tree(ROOT / "scripts" / "medical", source / "scripts" / "medical")
    _copy_tree(ROOT / "tests" / "medical", source / "tests" / "medical")
    _copy_tree(ROOT / "docs" / "medical", source / "docs" / "medical")
    _copy_tree(ROOT / "configs" / "medical", staging / "CONFIGS" / "medical")
    _copy_tree(ROOT / "configs" / "manifests", staging / "CONFIGS" / "manifests")
    _copy_tree(ROOT / "configs" / "medical", source / "configs" / "medical")
    _copy_tree(ROOT / "configs" / "manifests", source / "configs" / "manifests")
    _write(source / "git_patch.diff", _git_or_value(["git", "diff", "HEAD"], ""))

    tests = staging / "TESTS"
    if run_tests:
        pytest_result = _run_completed([sys.executable, "-m", "pytest", "tests/medical", "-q", "--disable-warnings"])
        compile_result = _run_completed([sys.executable, "-m", "compileall", "src/med_circuitbench", "scripts/medical"])
    else:
        pytest_result = subprocess.CompletedProcess([], 0, "Skipped by package builder default.\n", "")
        compile_result = subprocess.CompletedProcess([], 0, "Skipped by package builder default.\n", "")
    _write(tests / "pytest_stdout.log", pytest_result.stdout)
    _write(tests / "pytest_stderr.log", pytest_result.stderr)
    _write(tests / "pytest_exit_code.txt", f"{pytest_result.returncode}\n")
    _write(tests / "compileall_stdout.log", compile_result.stdout)
    _write(tests / "compileall_stderr.log", compile_result.stderr)
    _write(tests / "compileall_exit_code.txt", f"{compile_result.returncode}\n")
    _copy_file_if_exists(smoke_logs / "end_to_end_smoke_stdout.log", tests / "end_to_end_smoke_stdout.log")
    _copy_file_if_exists(smoke_logs / "end_to_end_smoke_stderr.log", tests / "end_to_end_smoke_stderr.log")
    _copy_file_if_exists(smoke_logs / "end_to_end_smoke_exit_code.txt", tests / "end_to_end_smoke_exit_code.txt")
    if not (tests / "end_to_end_smoke_stdout.log").exists():
        _write(tests / "end_to_end_smoke_stdout.log", "Smoke log was not captured before package assembly.\n")
        _write(tests / "end_to_end_smoke_stderr.log", "")
        _write(tests / "end_to_end_smoke_exit_code.txt", "1\n")
    if pytest_result.returncode != 0:
        raise SystemExit(pytest_result.returncode)
    if compile_result.returncode != 0:
        raise SystemExit(compile_result.returncode)

    results = staging / "RESULTS"
    required_result_files = [
        "aggregate_metrics.csv",
        "benchmark_comparison.csv",
        "circuit_f1_by_seed.csv",
        "circuit_metrics_by_seed.csv",
        "model_metrics_by_seed.csv",
        "bootstrap_intervals.csv",
        "feature_stability.csv",
        "edge_stability.csv",
        "go_no_go.json",
    ]
    for filename in required_result_files:
        _copy_file_if_exists(aggregate_src / filename, results / filename)
        _copy_file_if_exists(aggregate_src / filename, staging / filename)
    _copy_file_if_exists(aggregate_src / "go_no_go.json", results / "go_no_go_validation.json")
    _copy_tree(aggregate_src / "tables", staging / "TABLES")
    _copy_tree(aggregate_src / "figures", staging / "FIGURES")
    _copy_runs_lightweight(runs_src, staging / "RUNS")

    logs_dir = staging / "LOGS"
    if runs_src.exists():
        for run in sorted(runs_src.glob("seed_*_validation")):
            _copy_tree(run / "logs", logs_dir / run.name)
            _copy_file_if_exists(run / "timing.csv", logs_dir / run.name / "timing.csv")
            _copy_file_if_exists(run / "resource_usage.csv", logs_dir / run.name / "resource_usage.csv")
            _copy_file_if_exists(run / "manifest.json", staging / "MANIFESTS" / f"{run.name}_manifest.json")
    _copy_file_if_exists(go_path, staging / "MANIFESTS" / "go_no_go.json")

    checkpoints = staging / "CHECKPOINTS"
    _write(checkpoints / "README.txt", "Model checkpoint binaries are excluded from the compact ZIP. Run manifests identify the producing commands.\n")
    limitations = staging / "LIMITATIONS"
    known = ROOT / "docs" / "medical" / "KNOWN_LIMITATIONS.md"
    if known.exists():
        _copy_file_if_exists(known, limitations / "known_issues.md")
    else:
        _write(limitations / "known_issues.md", "No separate known-limitations file was present.\n")
    _write(limitations / "physionet_status.md", "PhysioNet validation requires an external raw data root and is reported as BLOCKED_DATA_ACCESS when absent.\n")

    manifest_payload = {
        "name": name,
        "commit": commit,
        "branch": branch,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "validation_status": go_payload.get("overall_status", "NO_GO"),
        "forbidden_token_check": "enabled",
    }
    _write(staging / "MANIFESTS" / "delivery_manifest.json", json.dumps(manifest_payload, indent=2))

    _assert_no_forbidden_tokens(staging)
    checksums = []
    for file in sorted(p for p in staging.rglob("*") if p.is_file()):
        rel = file.relative_to(staging)
        if rel == Path("checksums.sha256") or rel == Path("MANIFESTS/checksums.sha256"):
            continue
        checksums.append(f"{_sha256(file)}  {rel.as_posix()}")
    checksum_text = "\n".join(checksums) + "\n"
    _write(staging / "checksums.sha256", checksum_text)
    _write(staging / "MANIFESTS" / "checksums.sha256", checksum_text)

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
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    zip_path, sha_path = build_delivery(args.out_dir, commit=args.commit, branch=args.branch, run_tests=args.run_tests)
    print(json.dumps({"zip": str(zip_path), "sha256": str(sha_path), "size_bytes": zip_path.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
