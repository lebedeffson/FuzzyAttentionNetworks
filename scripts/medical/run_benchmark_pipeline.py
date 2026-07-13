#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log.write(proc.stdout)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--unlock-test")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    root = Path(cfg.get("artifacts", {}).get("root", "artifacts/medical"))
    overall = []
    if args.split == "test" and not args.unlock_test:
        raise SystemExit("test split requires --unlock-test")
    for seed in args.seeds:
        run_id = f"seed_{seed}_{args.split}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        run_dir = root / "benchmark_runs" / run_id
        logs = run_dir / "logs"
        run_dir.mkdir(parents=True, exist_ok=True)
        resolved = dict(cfg)
        resolved["dataset"] = dict(cfg["dataset"])
        resolved["dataset"]["seed"] = seed
        resolved_config = run_dir / "config_resolved.yaml"
        resolved_config.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        stages = [
            ["generate_benchmark.py", "--config", str(resolved_config)],
            ["train_transformer.py", "--dataset", "med_circuitbench", "--config", str(resolved_config)],
            ["extract_activations.py", "--dataset", "med_circuitbench", "--config", str(resolved_config)],
            ["screen_layers.py", "--dataset", "med_circuitbench", "--config", str(resolved_config)],
            ["train_baselines.py", "--dataset", "med_circuitbench", "--config", str(resolved_config)],
            ["train_sctc.py", "--dataset", "med_circuitbench", "--config", str(resolved_config), "--epochs", "1" if args.smoke else str(cfg.get("sctc", {}).get("maximum_epochs", 3))],
            ["build_circuits.py", "--dataset", "med_circuitbench", "--config", str(resolved_config)],
        ]
        status = "SUCCESS"
        completed = []
        for stage in stages:
            cmd = [sys.executable, str(ROOT / "scripts" / "medical" / stage[0]), *stage[1:]]
            code = _run(cmd, logs / f"{Path(stage[0]).stem}.log")
            completed.append({"stage": stage[0], "exit_code": code})
            if code != 0:
                status = "FAILED"
                break
        manifest = {
            "run_id": run_id,
            "seed": seed,
            "split": args.split,
            "status": status,
            "stages": completed,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        overall.append(manifest)
        if status != "SUCCESS":
            raise SystemExit(1)
    print(json.dumps({"runs": overall}, indent=2))


if __name__ == "__main__":
    main()
