#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from conceptfan_realdata.posthoc_attribution import run_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen PhysioNet 2012 post-hoc attribution audit.")
    parser.add_argument("--config", type=Path, default=Path("configs/medical/physionet2012/posthoc_attribution_stability.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/physionet2012/data.yaml"))
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/physionet2012_last_run/prepared/prepared_physionet2012.npz"))
    parser.add_argument("--runs", type=Path, default=Path("artifacts/physionet2012_last_run/runs"))
    parser.add_argument("--output", type=Path, default=Path("reports/physionet2012/posthoc_attribution_audit"))
    parser.add_argument("--map", dest="map_path", type=Path, default=Path("configs/medical/physionet2012/channel_to_proxy_concept_map.csv"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--write-map-only", action="store_true")
    args = parser.parse_args()
    run_audit(
        config_path=args.config,
        data_config_path=args.data_config,
        prepared_path=args.prepared,
        runs_root=args.runs,
        output_dir=args.output,
        map_path=args.map_path,
        device_name=args.device,
        write_map_only=args.write_map_only,
    )


if __name__ == "__main__":
    main()
