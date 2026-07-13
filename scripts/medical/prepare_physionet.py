#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common import add_config_arg, load_config
from src.med_circuitbench.data.physionet import prepare_physionet_dir


def main() -> None:
    parser = add_config_arg()
    args = parser.parse_args()
    cfg = load_config(args.config)
    ds = cfg["dataset"]
    result = prepare_physionet_dir(Path(ds["raw_dir"]), Path(ds["prepared_dir"]), int(ds.get("seed", 42)))
    print(result)


if __name__ == "__main__":
    main()
