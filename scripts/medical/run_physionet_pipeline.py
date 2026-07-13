#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--data-root")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    data_root = Path(args.data_root or os.environ.get("PHYSIONET2019_ROOT", ""))
    out = Path(cfg.get("artifacts", {}).get("root", "artifacts/medical")) / "physionet2019"
    out.mkdir(parents=True, exist_ok=True)
    if not str(data_root) or not data_root.exists():
        report = {
            "status": "BLOCKED_DATA_ACCESS",
            "reason": "PHYSIONET2019_ROOT/--data-root missing",
            "split": args.split,
            "seeds": args.seeds,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (out / "blocked_data_access.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    raise SystemExit("PhysioNet full runner requires real data preparation; raw data is not bundled.")


if __name__ == "__main__":
    main()
