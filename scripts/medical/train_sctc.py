#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    out = Path("artifacts/medical") / args.dataset / "sctc"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"layer": 0, "selected": True, "lambda_1": 1e-4, "status": "placeholder_ready"}]).to_csv(
        out / "lambda_selection.csv", index=False
    )
    print({"lambda_selection": str(out / "lambda_selection.csv")})


if __name__ == "__main__":
    main()
