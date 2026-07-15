#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
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


def run_command(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout[-6000:],
        "stderr": proc.stderr[-6000:],
        "passed": proc.returncode == 0,
    }


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
    parser.add_argument("--medical-zip", default=None)
    parser.add_argument("--mimic-zip", default=None)
    parser.add_argument("--rebuild-nested", action="store_true")
    args = parser.parse_args(argv)

    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    rebuild_reports: dict[str, dict] = {}
    if args.rebuild_nested or args.medical_zip is None:
        rebuild_reports["medical"] = run_command(
            [
                sys.executable,
                "scripts/medical/v3_1/package_v3_1_practical_release.py",
                "--zip-output-dir",
                "artifacts/medical",
            ]
        )
        if not rebuild_reports["medical"]["passed"]:
            raise RuntimeError(json.dumps(rebuild_reports["medical"], indent=2))
    if args.rebuild_nested or args.mimic_zip is None:
        rebuild_reports["mimic"] = run_command(
            [
                sys.executable,
                "scripts/mimic_aki/run_demo_engineering.py",
                "--config",
                "configs/mimic_aki/program_demo.yaml",
                "--output",
                "artifacts/mimic_aki/final_demo",
            ]
        )
        if not rebuild_reports["mimic"]["passed"]:
            raise RuntimeError(json.dumps(rebuild_reports["mimic"], indent=2))

    commit_short = git_text(["rev-parse", "--short", "HEAD"])
    medical_arg = args.medical_zip or f"artifacts/medical/Med_CircuitBench_V3_1_RESEARCH_COMPLETE_{commit_short}.zip"
    mimic_arg = args.mimic_zip or f"artifacts/mimic_aki/MIMIC_AKI_DEMO_ENGINEERING_FINAL_{commit_short}.zip"
    medical_zip = (ROOT / medical_arg).resolve()
    mimic_zip = (ROOT / mimic_arg).resolve()
    if not medical_zip.exists():
        raise FileNotFoundError(medical_zip)
    if not mimic_zip.exists():
        raise FileNotFoundError(mimic_zip)

    final_status = {
        "status": "SCIENTIFIC_RESEARCH_COMPLETE_ENGINEERING_DEMO_COMPLETE_STANDALONE_RELEASE_VALIDATED_FULL_CLINICAL_VALIDATION_BLOCKED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_text(["rev-parse", "HEAD"]),
        "branch": git_text(["branch", "--show-current"]),
        "rebuild_reports": rebuild_reports,
        "roles": {
            "Med-CircuitBench": "scientific validation and negative/mixed mechanistic result",
            "MIMIC-IV Demo": "engineering compatibility only, not clinical performance evidence",
            "Full MIMIC-IV": "blocked external clinical validation requirement",
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
            "status": "SCIENTIFIC_RESEARCH_COMPLETE",
            "production_fan": "FAN-NoAlpha",
            "sctc_line": "closed mixed/negative",
        },
        "mimic_demo": {
            "source_zip": mimic_zip.name,
            "sha256": sha256_file(mimic_zip),
            "status": "ENGINEERING_DEMO_COMPLETE",
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
    claims = [
        {
            "claim": "Med-CircuitBench scientific validation is complete",
            "status": "PASS",
            "scope": "scientific synthetic benchmark",
            "source": f"MED_CIRCUITBENCH/{medical_zip.name}",
        },
        {
            "claim": "MIMIC-IV Demo validates engineering compatibility only",
            "status": "PASS",
            "scope": "demo engineering",
            "source": f"MIMIC_AKI_DEMO/{mimic_zip.name}",
        },
        {
            "claim": "Standalone nested release archives were rebuilt from current source",
            "status": "PASS",
            "scope": "release engineering",
            "source": "MANIFESTS/dual_benchmark_status.json",
        },
        {
            "claim": "Full clinical validation on full MIMIC-IV",
            "status": "BLOCKED_FULL_MIMIC_ACCESS",
            "scope": "external data requirement",
            "source": "README_FIRST.md",
        },
        {
            "claim": "MIMIC empirical temporal faithfulness",
            "status": "NOT_EVALUATED",
            "scope": "demo limitation",
            "source": f"MIMIC_AKI_DEMO/{mimic_zip.name}",
        },
        {
            "claim": "MIMIC SAE fidelity",
            "status": "FAIL_ON_DEMO",
            "scope": "demo limitation",
            "source": f"MIMIC_AKI_DEMO/{mimic_zip.name}",
        },
    ]
    claims_csv = "claim,status,scope,source\n" + "\n".join(
        f"\"{row['claim']}\",{row['status']},\"{row['scope']}\",\"{row['source']}\"" for row in claims
    ) + "\n"
    (staging / "claims_matrix.csv").write_text(claims_csv, encoding="utf-8")
    write_json(staging / "MANIFESTS" / "claims_matrix.json", claims)
    (staging / "TABLES").mkdir(exist_ok=True)
    (staging / "TABLES" / "aggregate_results.csv").write_text(
        "\n".join(
            [
                "block,status,notes",
                "Med-CircuitBench,SCIENTIFIC_RESEARCH_COMPLETE,FAN validated with SCTC mixed negative line",
                "MIMIC-IV Demo,ENGINEERING_DEMO_COMPLETE,Real-format engineering compatibility only",
                "Standalone release,VALIDATED,Current nested archives rebuilt and packaged",
                "Full MIMIC-IV,BLOCKED_FULL_MIMIC_ACCESS,External data access required",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if (ROOT / "docs" / "medical" / "DUAL_BENCHMARK_FINAL_STATUS.md").exists():
        shutil.copy2(ROOT / "docs" / "medical" / "DUAL_BENCHMARK_FINAL_STATUS.md", staging / "DUAL_BENCHMARK_FINAL_STATUS.md")

    zip_path = output / f"DUAL_BENCHMARK_STANDALONE_VALIDATED_{commit_short}.zip"
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
        "medical_zip_sha256": final_status["med_circuitbench"]["sha256"],
        "mimic_zip_sha256": final_status["mimic_demo"]["sha256"],
    }
    write_json(output / "dual_benchmark_release_report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if bad is None else 3


README = """# Dual Benchmark Standalone Validated

Open `MANIFESTS/dual_benchmark_status.json` first.

This package separates two roles.

Med-CircuitBench is the scientific validation block.

MIMIC-IV Demo is an engineering compatibility block. It is not clinical
performance evidence and is not a substitute for full MIMIC-IV.

The nested archives are rebuilt from the current source before this top-level
archive is created.
"""


LIMITATIONS = """# Known Limitations

The MIMIC-IV Demo block validates engineering mechanics only.

Full MIMIC-IV remains required for real clinical validation.

The Med-CircuitBench SCTC line is closed as a mixed or negative result. The
release does not claim solved causal discovery.

AUPRC >= 0.85 is not used as a release gate.

Demo temporal inputs are pseudo-temporal aggregate vectors, not real ICU hourly
trajectories. Demo faithfulness is structural-only and demo SAE fidelity is not
claimed when reconstruction metrics fail.
"""


if __name__ == "__main__":
    raise SystemExit(main())
