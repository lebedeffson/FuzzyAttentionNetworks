#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    out = Path("artifacts/medical") / args.dataset / "circuits"
    out.mkdir(parents=True, exist_ok=True)
    (out / "circuit_catalog.json").write_text(json.dumps([], indent=2), encoding="utf-8")
    print({"circuit_catalog": str(out / "circuit_catalog.json")})


if __name__ == "__main__":
    main()
