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
    out = Path("artifacts/medical") / args.dataset / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"method": "random_directions", "status": "configured"},
            {"method": "sae", "status": "configured"},
            {"method": "linear_probes", "status": "configured"},
        ]
    ).to_parquet(out / "baseline_metrics.parquet", index=False)
    print({"baseline_metrics": str(out / "baseline_metrics.parquet")})


if __name__ == "__main__":
    main()
