#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common import add_config_arg, load_config
from src.med_circuitbench.benchmark.generator import BenchmarkConfig, write_benchmark


def main() -> None:
    parser = add_config_arg()
    args = parser.parse_args()
    cfg_raw = load_config(args.config)
    ds = cfg_raw["dataset"]
    cfg = BenchmarkConfig(
        seed=int(ds["seed"]),
        n_samples=int(ds["n_samples"]),
        sequence_length=int(ds["sequence_length"]),
        input_window=int(ds["window"]),
        prediction_horizon=int(ds["horizon"]),
        target_threshold=float(ds.get("target_threshold", 0.65)),
        allow_target_fallback=bool(ds.get("allow_target_fallback", False)),
        fallback_positive_rate=float(ds.get("fallback_positive_rate", 0.25)),
        infection_prevalence=float(ds.get("infection_prevalence", 0.25)),
        infection_impulse_strength=float(ds.get("infection_impulse_strength", 3.0)),
    )
    out_dir = Path(cfg_raw.get("artifacts", {}).get("root", "artifacts/medical")) / "benchmark"
    files = write_benchmark(out_dir, cfg)
    for name, path in files.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
