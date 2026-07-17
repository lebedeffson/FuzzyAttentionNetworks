#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-glob", required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/medical/validation_freeze_manifest.json"))
    args = parser.parse_args()
    runs = [Path(p) for p in sorted(glob.glob(args.runs_glob))]
    manifests = [run / "manifest.json" for run in runs if (run / "manifest.json").exists()]
    payload = {
        "status": "FROZEN" if manifests else "NO_RUN_MANIFESTS",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": [{"path": str(path), "sha256": _sha(path)} for path in manifests],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
