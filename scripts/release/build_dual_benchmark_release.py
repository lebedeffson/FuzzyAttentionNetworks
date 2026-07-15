#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


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


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def add_file(zf: zipfile.ZipFile, path: Path, arcname: str, manifest: list[dict]) -> None:
    if not path.exists():
        return
    zf.write(path, arcname)
    manifest.append({"path": arcname, "bytes": path.stat().st_size, "sha256": sha256_file(path)})


def add_tree(zf: zipfile.ZipFile, root: Path, arc_root: str, manifest: list[dict]) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.name.startswith("mimic-iv") or "MIMIC" in path.name and path.suffix == ".csv":
            continue
        arcname = f"{arc_root}/{path.relative_to(root).as_posix()}"
        add_file(zf, path, arcname, manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/release")
    parser.add_argument("--medical-zip", default="artifacts/medical/Med_CircuitBench_V3_1_RESEARCH_COMPLETE_c4820113.zip")
    parser.add_argument("--mimic-zip", default="artifacts/mimic_aki/MIMIC_AKI_DEMO_ENGINEERING_FINAL_5848611.zip")
    args = parser.parse_args(argv)

    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    medical_zip = (ROOT / args.medical_zip).resolve()
    mimic_zip = (ROOT / args.mimic_zip).resolve()
    if not medical_zip.exists():
        raise FileNotFoundError(medical_zip)
    if not mimic_zip.exists():
        raise FileNotFoundError(mimic_zip)

    final_status = {
        "status": "PRACTICE_CLOSED_DUAL_BENCHMARK_RESEARCH_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_text(["rev-parse", "HEAD"]),
        "branch": git_text(["branch", "--show-current"]),
        "roles": {
            "Med-CircuitBench": "scientific validation and negative/mixed mechanistic result",
            "MIMIC-IV Demo": "engineering compatibility only, not clinical performance evidence",
        },
        "registered_decisions": {
            "auprc_0_85_gate": "not used",
            "oracle_first": True,
            "stability": "reported as scientific result, not tuned with new loss",
            "new_hyperparameter_experiments_after_release": "forbidden",
        },
        "med_circuitbench": {
            "source_zip": medical_zip.name,
            "sha256": sha256_file(medical_zip),
            "status": "V3.1_RESEARCH_COMPLETE",
            "production_fan": "FAN-NoAlpha",
            "sctc_line": "closed mixed/negative",
        },
        "mimic_demo": {
            "source_zip": mimic_zip.name,
            "sha256": sha256_file(mimic_zip),
            "status": "PRACTICE_CLOSED_MIMIC_DEMO_END_TO_END",
            "full_mimic_status": "BLOCKED_FULL_MIMIC_ACCESS",
        },
    }

    staging = output / "dual_benchmark_release"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    write_json(staging / "MANIFESTS" / "dual_benchmark_status.json", final_status)
    (staging / "README_FIRST.md").write_text(README, encoding="utf-8")
    (staging / "KNOWN_LIMITATIONS.md").write_text(LIMITATIONS, encoding="utf-8")
    if (ROOT / "docs" / "medical" / "DUAL_BENCHMARK_FINAL_STATUS.md").exists():
        shutil.copy2(ROOT / "docs" / "medical" / "DUAL_BENCHMARK_FINAL_STATUS.md", staging / "DUAL_BENCHMARK_FINAL_STATUS.md")

    commit = git_text(["rev-parse", "--short", "HEAD"])
    zip_path = output / f"DUAL_BENCHMARK_RESEARCH_COMPLETE_{commit}.zip"
    if zip_path.exists():
        zip_path.unlink()
    manifest: list[dict] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        add_tree(zf, staging, ".", manifest)
        add_file(zf, medical_zip, f"MED_CIRCUITBENCH/{medical_zip.name}", manifest)
        add_file(zf, Path(str(medical_zip) + ".sha256"), f"MED_CIRCUITBENCH/{medical_zip.name}.sha256", manifest)
        add_file(zf, mimic_zip, f"MIMIC_AKI_DEMO/{mimic_zip.name}", manifest)
        add_file(zf, Path(str(mimic_zip) + ".sha256"), f"MIMIC_AKI_DEMO/{mimic_zip.name}.sha256", manifest)
        add_tree(zf, ROOT / "docs" / "medical", "DOCS/medical", manifest)
        add_file(zf, ROOT / "AGENTS.md", "MANIFESTS/AGENTS.md", manifest)
        add_file(zf, ROOT / "requirements-lock.txt", "requirements-lock.txt", manifest)
        add_file(zf, ROOT / "pyproject.toml", "pyproject.toml", manifest)
        zf.writestr("MANIFESTS/file_manifest.json", json.dumps(manifest, indent=2))

    sha = sha256_file(zip_path)
    sidecar = zip_path.with_suffix(".zip.sha256")
    sidecar.write_text(f"{sha}  {zip_path.name}\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
    report = {
        "status": "DUAL_BENCHMARK_RELEASE_CREATED" if bad is None else "ZIP_TEST_FAILED",
        "zip": str(zip_path),
        "bytes": zip_path.stat().st_size,
        "sha256": sha,
        "sidecar": str(sidecar),
        "unzip_test": bad is None,
        "final_status": final_status["status"],
    }
    write_json(output / "dual_benchmark_release_report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if bad is None else 3


README = """# Dual Benchmark Research Complete

Open `MANIFESTS/dual_benchmark_status.json` first.

This package separates two roles.

Med-CircuitBench is the scientific validation block.

MIMIC-IV Demo is an engineering compatibility block. It is not clinical
performance evidence and is not a substitute for full MIMIC-IV.
"""


LIMITATIONS = """# Known Limitations

The MIMIC-IV Demo block validates engineering mechanics only.

Full MIMIC-IV remains required for real clinical validation.

The Med-CircuitBench SCTC line is closed as a mixed or negative result. The
release does not claim solved causal discovery.

AUPRC >= 0.85 is not used as a release gate.
"""


if __name__ == "__main__":
    raise SystemExit(main())
