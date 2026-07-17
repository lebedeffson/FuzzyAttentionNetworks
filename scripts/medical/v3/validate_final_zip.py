#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(zip_path: Path) -> dict:
    findings = []
    if not zip_path.exists():
        return {"passed": False, "findings": [{"reason": "missing_zip"}]}
    if zip_path.stat().st_size >= 524288000:
        findings.append({"reason": "zip_too_large", "size": zip_path.stat().st_size})
    sidecar = zip_path.with_suffix(zip_path.suffix + ".sha256")
    if not sidecar.exists():
        findings.append({"reason": "missing_sidecar_sha256"})
    else:
        recorded = sidecar.read_text(encoding="utf-8").split()[0]
        actual = sha256_file(zip_path)
        if recorded != actual:
            findings.append({"reason": "sidecar_sha_mismatch", "recorded": recorded, "actual": actual})
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        names = zf.namelist()
    if bad:
        findings.append({"reason": "zip_test_failed", "bad_file": bad})
    required_dirs = ["SOURCE/", "TESTS/", "CONFIGS/", "PAPER/", "RESULTS/", "MANIFESTS/"]
    for required in required_dirs:
        if not any("/" + required in name or name.endswith(required) for name in names):
            findings.append({"reason": "missing_required_dir", "dir": required})
    for name in names:
        low = name.lower()
        if "__pycache__" in low or ".pytest_cache" in low or "activation_cache" in low or "/datasets/" in low:
            findings.append({"reason": "forbidden_cache_or_dataset", "path": name})
        if "optimizer" in low or "scheduler" in low:
            findings.append({"reason": "forbidden_training_state", "path": name})
    has_source = any("SOURCE/src/fan/" in name for name in names) and any("SOURCE/scripts/medical/v3/" in name for name in names)
    has_tests = any("TESTS/tests/medical/v3/" in name for name in names)
    if not has_source:
        findings.append({"reason": "source_not_packaged"})
    if not has_tests:
        findings.append({"reason": "tests_not_packaged"})
    return {"passed": not findings, "zip": str(zip_path), "size": zip_path.stat().st_size, "sha256": sha256_file(zip_path), "findings": findings}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    args = parser.parse_args(argv)
    result = validate(Path(args.zip))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
