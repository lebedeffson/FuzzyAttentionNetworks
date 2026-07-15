#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mimic_aki.access import verify_mimic_access  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    status = verify_mimic_access(args.root)
    payload = {"status": status.status, "root": status.root, "dataset_kind": status.dataset_kind, "missing_files": status.missing_files}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if status.status in {"OK", "OK_DEMO"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
