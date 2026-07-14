#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.medical.v3.finalize_real_research import delivery_manifest, git_commit, package, sha256_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-bytes", type=int, default=524288000)
    args = parser.parse_args(argv)
    source = Path(args.source)
    zip_path = package(source, git_commit())
    target = Path(args.output) / zip_path.name
    if zip_path.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        zip_path.replace(target)
        sidecar = zip_path.with_suffix(zip_path.suffix + ".sha256")
        if sidecar.exists():
            sidecar.replace(target.with_suffix(target.suffix + ".sha256"))
        zip_path = target
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
    result = {"zip": str(zip_path), "size": zip_path.stat().st_size, "sha256": sha256_file(zip_path), "passed": bad is None and zip_path.stat().st_size < args.max_bytes}
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
