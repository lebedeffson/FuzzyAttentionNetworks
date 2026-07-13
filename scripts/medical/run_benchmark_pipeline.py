#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log.write(proc.stdout)
    return proc.returncode


def _stage(script: str, *args: str) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / "medical" / script), *args]


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _collect_run_outputs(run_dir: Path, work_root: Path) -> dict:
    dataset_root = work_root / "med_circuitbench"
    outputs = {
        "benchmark_manifest": work_root / "benchmark" / "manifest.json",
        "transformer_metrics": dataset_root / "transformer" / "model_metrics.csv",
        "layer_scores": dataset_root / "layers" / "layer_scores.csv",
        "lambda_selection": dataset_root / "sctc" / "lambda_selection.csv",
        "feature_catalog": dataset_root / "sctc" / "feature_catalog.parquet",
        "edge_catalog": dataset_root / "circuits" / "edge_catalog.parquet",
        "circuit_catalog": dataset_root / "circuits" / "circuit_catalog.json",
        "evaluation_summary": dataset_root / "evaluation_summary.json",
        "graph_metrics": dataset_root / "evaluation" / "graph_metrics.csv",
    }
    copied = {}
    for name, src in outputs.items():
        dst = run_dir / "results" / src.name
        _copy_if_exists(src, dst)
        if dst.exists():
            copied[name] = str(dst.relative_to(run_dir))
    _copy_if_exists(dataset_root / "evaluation", run_dir / "evaluation")
    return copied


def _aggregate(output_root: Path, split: str, manifests: list[dict], criteria: dict) -> dict:
    aggregate_dir = output_root.parent / "aggregate" / f"benchmark_{split}"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for manifest in manifests:
        summary_path = Path(manifest["run_dir"]) / "results" / "evaluation_summary.json"
        if summary_path.exists():
            rows.append(json.loads(summary_path.read_text()))
        else:
            rows.append({"seed": manifest["seed"], "status": manifest["status"]})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(aggregate_dir / "metrics_by_seed.csv", index=False)

    failures = []
    if len(metrics) != len(manifests):
        failures.append("missing_seed_metrics")
    if "validation_auprc" not in metrics or (metrics["validation_auprc"] < float(criteria.get("minimum_validation_auprc", 0.85))).any():
        failures.append("validation_auprc")
    if "CircuitF1" not in metrics or (metrics["CircuitF1"] < float(criteria.get("minimum_circuit_f1", 0.70))).any():
        failures.append("circuit_f1")
    if "accepted_edges" not in metrics or (metrics["accepted_edges"] <= 0).any():
        failures.append("accepted_edges")
    status = "GO" if not failures else "NO_GO"

    intervals = []
    numeric_cols = [c for c in metrics.columns if pd.api.types.is_numeric_dtype(metrics[c])]
    for col in numeric_cols:
        values = metrics[col].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        intervals.append(
            {
                "metric": col,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "ci95_low": float(values.mean() - 1.96 * values.std(ddof=1) / (len(values) ** 0.5)) if len(values) > 1 else float(values.mean()),
                "ci95_high": float(values.mean() + 1.96 * values.std(ddof=1) / (len(values) ** 0.5)) if len(values) > 1 else float(values.mean()),
            }
        )
    pd.DataFrame(intervals).to_csv(aggregate_dir / "bootstrap_intervals.csv", index=False)
    decision = {
        "status": status,
        "failures": failures,
        "split": split,
        "seeds": [m["seed"] for m in manifests],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics_by_seed": str((aggregate_dir / "metrics_by_seed.csv").relative_to(output_root.parent.parent)),
        "bootstrap_intervals": str((aggregate_dir / "bootstrap_intervals.csv").relative_to(output_root.parent.parent)),
    }
    (aggregate_dir / "go_no_go.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--unlock-test")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/medical/runs/benchmark"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.split == "test" and not args.unlock_test:
        raise SystemExit("test split requires --unlock-test")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for seed in args.seeds:
        run_id = f"seed_{seed}_{args.split}"
        run_dir = output_root / run_id
        if run_dir.exists():
            raise SystemExit(f"refusing to overwrite existing run directory: {run_dir}")
        logs = run_dir / "logs"
        work_root = run_dir / "work"
        run_dir.mkdir(parents=True, exist_ok=True)
        resolved = dict(cfg)
        resolved["dataset"] = dict(cfg["dataset"])
        resolved["dataset"]["seed"] = seed
        resolved["artifacts"] = dict(cfg.get("artifacts", {}))
        resolved["artifacts"]["root"] = str(work_root)
        resolved_config = run_dir / "config_resolved.yaml"
        resolved_config.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        epochs = "1" if args.smoke else str(cfg.get("sctc", {}).get("maximum_epochs", 3))
        stages = [
            _stage("generate_benchmark.py", "--config", str(resolved_config)),
            _stage("train_transformer.py", "--dataset", "med_circuitbench", "--config", str(resolved_config)),
            _stage("extract_activations.py", "--dataset", "med_circuitbench", "--config", str(resolved_config)),
            _stage("screen_layers.py", "--dataset", "med_circuitbench", "--config", str(resolved_config)),
            _stage("train_baselines.py", "--dataset", "med_circuitbench", "--config", str(resolved_config)),
            _stage("train_sctc.py", "--dataset", "med_circuitbench", "--config", str(resolved_config), "--epochs", epochs),
            _stage("build_circuits.py", "--dataset", "med_circuitbench", "--config", str(resolved_config)),
            _stage("evaluate_benchmark.py", "--manifest", str(work_root / "med_circuitbench" / "circuits" / "manifest.json")),
        ]
        status = "SUCCESS"
        completed = []
        for cmd in stages:
            stage_name = Path(cmd[1]).stem
            code = _run(cmd, logs / f"{stage_name}.log")
            completed.append({"stage": stage_name, "exit_code": code})
            if code != 0:
                status = f"FAILED_{stage_name}"
                break
        files = _collect_run_outputs(run_dir, work_root)
        manifest = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "seed": seed,
            "split": args.split,
            "status": status,
            "stages": completed,
            "files": files,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifests.append(manifest)
        if status.startswith("FAILED"):
            break

    criteria = {
        "minimum_validation_auprc": 0.0 if args.smoke else 0.85,
        "minimum_circuit_f1": 0.0 if args.smoke else 0.70,
    }
    decision = _aggregate(output_root, args.split, manifests, criteria)
    print(json.dumps({"runs": manifests, "decision": decision}, indent=2))
    if any(m["status"].startswith("FAILED") for m in manifests):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
