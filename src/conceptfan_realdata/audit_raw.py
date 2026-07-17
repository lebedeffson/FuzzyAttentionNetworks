from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import audit_raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-zip", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_raw(args.set_zip, args.outcomes, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
