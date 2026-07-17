from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from fan.artifacts import AuditConfig, AuditRunner, FANBundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fan")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--seeds", nargs="+", default=["42", "43", "44"])
    audit = sub.add_parser("audit")
    audit.add_argument("--bundle", required=True)
    audit.add_argument("--dataset")
    audit.add_argument("--output", required=True)
    report = sub.add_parser("report")
    report.add_argument("--run", required=True)
    report.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        cmd = [
            sys.executable,
            "scripts/medical/v3_1/run_fan_noalpha_stable.py",
            "--config",
            "configs/medical/v3/full.yaml",
            "--method-config",
            args.config,
            "--seeds",
            *args.seeds,
            "--output",
            args.output,
        ]
        return subprocess.call(cmd)
    if args.command == "audit":
        bundle = FANBundle.load(args.bundle)
        dataset = pd.read_parquet(args.dataset) if args.dataset else None
        audit_report = AuditRunner(AuditConfig()).run(bundle, dataset)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        audit_report.to_json(args.output)
        return 0
    if args.command == "report":
        run = Path(args.run)
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "run": str(run),
            "gate": _read_json(run / "fan_stable_gate.json"),
            "metrics": _read_csv_head(run / "fan_stable_metrics.csv"),
        }
        (out / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0
    return 2


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _read_csv_head(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path).to_dict(orient="records")


if __name__ == "__main__":
    raise SystemExit(main())
