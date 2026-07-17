#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from conceptfan_realdata.posthoc_attribution import verify_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only verification of a completed post-hoc attribution audit.")
    parser.add_argument("--report", type=Path, default=Path("reports/physionet2012/posthoc_attribution_audit"))
    parser.add_argument("--runs", type=Path, default=Path("artifacts/physionet2012_last_run/runs"))
    parser.add_argument("--map", dest="map_path", type=Path, default=Path("configs/medical/physionet2012/channel_to_proxy_concept_map.csv"))
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    report = verify_audit(args.report, args.runs, args.map_path, args.output_json)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
