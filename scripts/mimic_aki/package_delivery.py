#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
EXCLUDE_NAMES = {
    "mimic-iv-clinical-database-demo-2.2.zip",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def should_include(path: Path) -> bool:
    if path.name in EXCLUDE_NAMES:
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    if path.name.startswith("mimic-iv") and path.suffix == ".zip":
        return False
    return path.is_file()


def iter_files(base: Path) -> list[Path]:
    if not base.exists():
        return []
    if base.is_file():
        return [base] if should_include(base) else []
    return sorted(path for path in base.rglob("*") if should_include(path))


def add_tree(zf: zipfile.ZipFile, source: Path, dest_root: str, manifest: list[dict]) -> None:
    for path in iter_files(source):
        rel = path.relative_to(source.parent if source.is_file() else source)
        arcname = f"{dest_root}/{rel.as_posix()}"
        zf.write(path, arcname)
        manifest.append(
            {
                "path": arcname,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )


def run_tests(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/mimic_aki",
        "tests/fan_attention",
        "tests/medical/v3/test_v3_1_method_improvements.py",
        "-q",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    (output_dir / "pytest_output.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return {"command": " ".join(cmd), "returncode": proc.returncode, "passed": proc.returncode == 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output", default="artifacts/mimic_aki/demo")
    parser.add_argument("--output", default="artifacts/mimic_aki")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args(argv)

    source_output = (ROOT / args.source_output).resolve()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    commit = git_text(["rev-parse", "--short", "HEAD"])
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    zip_path = output / f"MIMIC_AKI_DEMO_DELIVERY_{date}_{commit}.zip"
    sidecar_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    test_report = None
    if args.run_tests:
        test_report = run_tests(output / "mimic_aki_delivery_tests")
        if not test_report["passed"]:
            print(json.dumps({"status": "TESTS_FAILED", "test_report": test_report}, indent=2))
            return 2

    manifest: list[dict] = []
    metadata = {
        "package": "MIMIC_AKI_DEMO_DELIVERY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_text(["rev-parse", "HEAD"]),
        "branch": git_text(["branch", "--show-current"]),
        "source_output": str(source_output.relative_to(ROOT)),
        "clinical_data_included": False,
        "demo_zip_included": False,
        "status": "DEMO_ONLY_NOT_FULL_VALIDATION",
        "test_report": test_report,
    }

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        add_tree(zf, ROOT / "src" / "mimic_aki", "SOURCE/src/mimic_aki", manifest)
        add_tree(zf, ROOT / "src" / "fan" / "attention", "SOURCE/src/fan/attention", manifest)
        add_tree(zf, ROOT / "src" / "fan" / "sae", "SOURCE/src/fan/sae", manifest)
        add_tree(zf, ROOT / "scripts" / "mimic_aki", "SOURCE/scripts/mimic_aki", manifest)
        add_tree(zf, ROOT / "configs" / "mimic_aki", "CONFIGS/configs/mimic_aki", manifest)
        add_tree(zf, ROOT / "tests" / "mimic_aki", "TESTS/tests/mimic_aki", manifest)
        add_tree(zf, ROOT / "tests" / "fan_attention", "TESTS/tests/fan_attention", manifest)
        add_tree(zf, source_output, "RESULTS/demo", manifest)
        if args.run_tests:
            add_tree(zf, output / "mimic_aki_delivery_tests", "TESTS/reports", manifest)
        zf.writestr("README_FIRST.md", README)
        zf.writestr("MANIFEST/package_metadata.json", json.dumps(metadata, indent=2))
        zf.writestr("MANIFEST/file_manifest.json", json.dumps(manifest, indent=2))

    zip_sha = sha256_file(zip_path)
    sidecar_path.write_text(f"{zip_sha}  {zip_path.name}\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
    report = {
        "status": "DELIVERY_PACKAGE_CREATED" if bad is None else "ZIP_TEST_FAILED",
        "zip": str(zip_path),
        "bytes": zip_path.stat().st_size,
        "sha256": zip_sha,
        "sidecar": str(sidecar_path),
        "unzip_test": bad is None,
        "files": len(manifest) + 3,
    }
    print(json.dumps(report, indent=2))
    return 0 if bad is None else 3


README = """# MIMIC-AKI Demo Delivery

This archive contains source code, configs, tests, and demo-derived artifacts
for the MIMIC-AKI FAN/SAE program.

No MIMIC-IV clinical data archive is included. The local demo zip remains
outside the package and is ignored by Git.

The included demo results are a real-format pipeline check only. They are not a
full clinical validation and must not be reported as final scientific evidence.
Full training, SAE, steering, and frozen test remain gated until full MIMIC-IV
access is available.
"""


if __name__ == "__main__":
    raise SystemExit(main())
