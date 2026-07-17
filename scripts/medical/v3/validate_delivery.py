#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.medical.v3.run_research_program import validate_delivery


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(validate_delivery(Path(args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
