from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import validate_release
from .data import load_prepared
from .pipeline import load_config, prepare_all, resolve_device, run_full_pipeline, run_posthoc
from .reporting import build_report
from .training import train_run


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ConceptFAN PhysioNet 2012 real-data pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    full = subparsers.add_parser("full")
    full.add_argument("--config", type=Path, default=Path("configs/physionet2012/data.yaml"))
    full.add_argument("--set-a-zip", type=Path, required=True)
    full.add_argument("--outcomes", type=Path, required=True)
    full.add_argument("--artifacts-dir", type=Path, required=True)
    full.add_argument("--report-dir", type=Path, default=Path("reports/physionet2012"))
    full.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    full.add_argument("--resume", action="store_true")
    full.add_argument("--dry-run", action="store_true")
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=Path, default=Path("configs/physionet2012/data.yaml"))
    prepare.add_argument("--set-a-zip", type=Path, required=True)
    prepare.add_argument("--outcomes", type=Path, required=True)
    prepare.add_argument("--artifacts-dir", type=Path, required=True)
    prepare.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    prepare.add_argument("--dry-run", action="store_true")
    train = subparsers.add_parser("train")
    train.add_argument("--model", choices=["conceptfan", "pure_nofuzzy", "transformer", "stability_reg", "temporal_cem"], required=True)
    train.add_argument("--config", type=Path, default=Path("configs/physionet2012/data.yaml"))
    train.add_argument("--prepared", type=Path, required=True)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--init-seed", type=int, required=True)
    train.add_argument("--data-order-seed", type=int, required=True)
    train.add_argument("--channels", default="V+M+D")
    train.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    train.add_argument("--dry-run", action="store_true")
    posthoc = subparsers.add_parser("posthoc")
    posthoc.add_argument("--config", type=Path, default=Path("configs/physionet2012/data.yaml"))
    posthoc.add_argument("--prepared", type=Path, required=True)
    posthoc.add_argument("--artifacts-dir", type=Path, required=True)
    posthoc.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    posthoc.add_argument("--dry-run", action="store_true")
    report = subparsers.add_parser("report")
    report.add_argument("--prepared", type=Path, required=True)
    report.add_argument("--artifacts-root", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)
    report.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    report.add_argument("--dry-run", action="store_true")
    readonly = subparsers.add_parser("readonly-audit")
    readonly.add_argument("--artifacts-root", type=Path, required=True)
    readonly.add_argument("--report-dir", type=Path, required=True)
    readonly.add_argument("--output-json", type=Path, required=True)
    readonly.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    readonly.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "full":
        result = run_full_pipeline(
            _root(), args.config, args.set_a_zip, args.outcomes, args.artifacts_dir, args.report_dir, args.device, args.resume, args.dry_run
        )
    elif args.command == "prepare":
        if args.dry_run:
            result = {"status": "DRY_RUN", "command": "prepare"}
        else:
            data = prepare_all(args.set_a_zip, args.outcomes, args.artifacts_dir, load_config(args.config))
            result = {"status": data.metadata["status"], "patients": len(data.record_ids)}
    elif args.command == "train":
        mapping = {
            "conceptfan": "ConceptFAN-NoAlpha",
            "pure_nofuzzy": "PureNoFuzzy",
            "transformer": "PlainTransformer",
            "stability_reg": "ConceptFAN-StabilityReg",
            "temporal_cem": "TemporalCEM",
        }
        if args.dry_run:
            result = {"status": "DRY_RUN", "model_arm": mapping[args.model]}
        else:
            config = load_config(args.config)
            result = train_run(
                mapping[args.model], load_prepared(args.prepared), config, args.run_dir, args.init_seed,
                args.data_order_seed, args.channels, resolve_device(args.device), int(config["training"]["batch_size"]),
                {"lambda_stab": 0.01, "lambda_logit": 0.0} if args.model == "stability_reg" else None,
            )
    elif args.command == "posthoc":
        if args.dry_run:
            result = {"status": "DRY_RUN", "command": "posthoc"}
        else:
            run_posthoc(load_prepared(args.prepared), load_config(args.config), args.artifacts_dir, resolve_device(args.device))
            result = {"status": "POSTHOC_COMPLETE"}
    elif args.command == "report":
        if args.dry_run:
            result = {"status": "DRY_RUN", "command": "report"}
        else:
            result = build_report(load_prepared(args.prepared), args.artifacts_root, args.output_dir)
    else:
        if args.dry_run:
            result = {"status": "DRY_RUN", "command": "readonly-audit"}
        else:
            result = validate_release(args.artifacts_root, args.report_dir, args.output_json)
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
