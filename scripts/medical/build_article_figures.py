#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    out = Path("artifacts/medical/article/figures")
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(f"Built from {args.manifest}\n", encoding="utf-8")
    print({"figures": str(out)})


if __name__ == "__main__":
    main()
