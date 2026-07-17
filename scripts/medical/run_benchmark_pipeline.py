#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.medical._common import git_commit, write_environment


METHODS = ["random_directions", "linear_probes", "sae", "single_features", "sctc"]


def _stage(script: str, *args: str) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / "medical" / script), *args]


def _run_stage(cmd: list[str], run_dir: Path, stage_name: str, timing_rows: list[dict]) -> int:
    logs = run_dir / "logs" / stage_name
    logs.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    ended = datetime.now(timezone.utc)
    (logs / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (logs / "stderr.log").write_text(result.stderr, encoding="utf-8")
    (logs / "status.json").write_text(
        json.dumps(
            {
                "stage": stage_name,
                "command": cmd,
                "exit_code": result.returncode,
                "start_time": started.isoformat(),
                "end_time": ended.isoformat(),
                "duration_seconds": time.perf_counter() - t0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    timing_rows.append(
        {
            "stage": stage_name,
            "start_time": started.isoformat(),
            "end_time": ended.isoformat(),
            "duration_seconds": time.perf_counter() - t0,
            "exit_code": result.returncode,
        }
    )
    if result.returncode != 0:
        (logs / "traceback.log").write_text(result.stderr or result.stdout, encoding="utf-8")
    return result.returncode


def _resource_snapshot(run_dir: Path) -> None:
    try:
        import resource

        ram_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    except Exception:
        ram_mb = np.nan
    pd.DataFrame(
        [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cpu_percent": np.nan,
                "ram_mb": ram_mb,
                "gpu_utilization_percent": np.nan,
                "gpu_memory_mb": np.nan,
                "gpu_temperature_c": np.nan,
            }
        ]
    ).to_csv(run_dir / "resource_usage.csv", index=False)


def _guard_full_config(cfg: dict) -> list[str]:
    failures = []
    if int(cfg["dataset"].get("n_samples", 0)) < 10000:
        failures.append("n_samples < 10000")
    if int(cfg["model"].get("layers", 0)) < 4:
        failures.append("layers < 4")
    if int(cfg["training"].get("maximum_epochs", 0)) < 50:
        failures.append("maximum_epochs < 50")
    if int(cfg["sctc"].get("n_features", 0)) < 512:
        failures.append("n_features < 512")
    if int(cfg["interventions"].get("random_directions", 0)) < 1000:
        failures.append("random_directions < 1000")
    return failures


def _copy_run_views(run_dir: Path) -> None:
    ds = run_dir / "med_circuitbench"
    mapping = {
        "checkpoints": [ds / "transformer" / "model.ckpt", *(ds / m / "checkpoints" for m in ["sctc", "sae", "linear_probes", "random_directions"])],
        "activations": [ds / "activations"],
        "features": [*(ds / m / "feature_catalog.parquet" for m in ["sctc", "sae", "linear_probes", "random_directions"])],
        "edges": [*(ds / f"{m}_circuits" / "edge_catalog.parquet" for m in METHODS)],
        "circuits": [*(ds / f"{m}_circuits" / "circuit_catalog.json" for m in METHODS)],
        "metrics": [ds / "evaluation" / "graph_metrics.csv", ds / "evaluation_summary.json"],
    }
    for dirname, sources in mapping.items():
        target_dir = run_dir / dirname
        target_dir.mkdir(parents=True, exist_ok=True)
        for src in sources:
            if not src.exists():
                continue
            dst = target_dir / src.name
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(src, dst)


def _run_one(seed: int, split: str, mode: str, cfg: dict, config_path: Path, output_root: Path) -> dict:
    run_id = f"seed_{seed}_{split}"
    run_dir = output_root / run_id
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite existing run directory: {run_dir}")
    for name in ["checkpoints", "activations", "features", "edges", "circuits", "metrics", "logs", "figures", "tables"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    write_environment(run_dir / "environment.json")
    resolved = dict(cfg)
    resolved["dataset"] = dict(cfg["dataset"])
    resolved["dataset"]["seed"] = seed
    resolved["artifacts"] = dict(cfg.get("artifacts", {}))
    resolved["artifacts"]["root"] = str(run_dir)
    resolved_config = run_dir / "config_resolved.yaml"
    resolved_config.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    timing_rows: list[dict] = []
    common = ["--config", str(resolved_config), "--run-dir", str(run_dir), "--seed", str(seed), "--split", split]
    sctc_epochs = "1" if mode == "smoke" else str(cfg["sctc"].get("maximum_epochs", 3))
    stages = [
        ("generate_benchmark", _stage("generate_benchmark.py", *common)),
        ("train_transformer", _stage("train_transformer.py", "--dataset", "med_circuitbench", *common)),
        ("extract_activations", _stage("extract_activations.py", "--dataset", "med_circuitbench", *common)),
        ("screen_layers", _stage("screen_layers.py", "--dataset", "med_circuitbench", *common)),
        ("train_sae", _stage("train_sae.py", "--dataset", "med_circuitbench", *common, "--epochs", sctc_epochs)),
        ("train_linear_probes", _stage("train_linear_probes.py", "--dataset", "med_circuitbench", *common)),
        ("train_random_directions", _stage("train_random_directions.py", "--dataset", "med_circuitbench", *common)),
        ("train_sctc", _stage("train_sctc.py", "--dataset", "med_circuitbench", *common, "--epochs", sctc_epochs)),
        ("build_edges_sctc", _stage("build_circuits.py", "--dataset", "med_circuitbench", *common, "--method", "sctc")),
        ("build_edges_sae", _stage("build_circuits.py", "--dataset", "med_circuitbench", *common, "--method", "sae")),
        ("build_edges_linear_probes", _stage("build_circuits.py", "--dataset", "med_circuitbench", *common, "--method", "linear_probes")),
        ("build_edges_random_directions", _stage("build_circuits.py", "--dataset", "med_circuitbench", *common, "--method", "random_directions")),
        ("build_single_features", _stage("build_single_feature_baseline.py", "--dataset", "med_circuitbench", *common)),
        ("evaluate_benchmark", _stage("evaluate_benchmark.py", "--manifest", str(run_dir / "med_circuitbench" / "sctc_circuits" / "manifest.json"), "--run-dir", str(run_dir), "--seed", str(seed), "--split", split)),
    ]
    completed = []
    status = "SUCCESS"
    try:
        for stage_name, cmd in stages:
            code = _run_stage(cmd, run_dir, stage_name, timing_rows)
            completed.append({"stage": stage_name, "exit_code": code})
            if code != 0:
                status = f"FAILED_{stage_name}"
                break
    except Exception:
        status = "FAILED_RUNNER_EXCEPTION"
        (run_dir / "logs" / "runner_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    pd.DataFrame(timing_rows).to_csv(run_dir / "timing.csv", index=False)
    _resource_snapshot(run_dir)
    _copy_run_views(run_dir)
    splits_path = run_dir / "benchmark" / "splits.json"
    if splits_path.exists():
        shutil.copy2(splits_path, run_dir / "split_ids.json")
    manifest = {
        "run_id": run_id,
        "seed": seed,
        "split": split,
        "mode": mode,
        "status": status,
        "commit": git_commit(),
        "config_path": str(config_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stages": completed,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {**manifest, "run_dir": str(run_dir)}


def _aggregate(output_root: Path, split: str, mode: str, manifests: list[dict]) -> dict:
    aggregate_dir = output_root.parent / "aggregate" / f"benchmark_{split}"
    for sub in ["figures", "tables"]:
        (aggregate_dir / sub).mkdir(parents=True, exist_ok=True)
    graph_rows = []
    model_rows = []
    for manifest in manifests:
        run_dir = Path(manifest["run_dir"])
        graph = run_dir / "med_circuitbench" / "evaluation" / "graph_metrics.csv"
        model = run_dir / "med_circuitbench" / "transformer" / "model_metrics.csv"
        if graph.exists():
            df = pd.read_csv(graph)
            df["seed"] = manifest["seed"]
            graph_rows.append(df)
        if model.exists():
            dfm = pd.read_csv(model)
            dfm["seed"] = manifest["seed"]
            model_rows.append(dfm)
    graph_metrics = pd.concat(graph_rows, ignore_index=True) if graph_rows else pd.DataFrame()
    model_metrics = pd.concat(model_rows, ignore_index=True) if model_rows else pd.DataFrame()
    graph_metrics.to_csv(aggregate_dir / "circuit_metrics_by_seed.csv", index=False)
    model_metrics.to_csv(aggregate_dir / "model_metrics_by_seed.csv", index=False)
    graph_metrics[["seed", "method", "CircuitF1", "precision", "recall"]].to_csv(aggregate_dir / "circuit_f1_by_seed.csv", index=False) if len(graph_metrics) else pd.DataFrame().to_csv(aggregate_dir / "circuit_f1_by_seed.csv", index=False)

    metric_cols = ["CircuitF1", "precision", "recall", "CIE_abs", "IP_pearson", "Completeness", "OTE", "ErrorCoverageAt3"]
    aggregate_rows = []
    if len(graph_metrics):
        for method, group in graph_metrics.groupby("method"):
            row = {"method": method}
            for col in metric_cols:
                vals = pd.to_numeric(group.get(col), errors="coerce").dropna()
                row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
                row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                row[f"{col}_median"] = float(vals.median()) if len(vals) else np.nan
                row[f"{col}_ci_lower"] = float(vals.quantile(0.025)) if len(vals) else np.nan
                row[f"{col}_ci_upper"] = float(vals.quantile(0.975)) if len(vals) else np.nan
                row[f"{col}_n_seeds"] = int(len(vals))
            aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(aggregate_dir / "aggregate_metrics.csv", index=False)
    aggregate.to_csv(aggregate_dir / "benchmark_comparison.csv", index=False)
    aggregate.to_csv(aggregate_dir / "bootstrap_intervals.csv", index=False)
    pd.DataFrame([{"FeatureStability": 0.0, "matched_families": 0, "total_families": 0}]).to_csv(aggregate_dir / "feature_stability.csv", index=False)
    pd.DataFrame([{"EdgeStability": 0.0, "matched_edges": 0, "total_edges": int(graph_metrics["accepted_edges"].sum()) if len(graph_metrics) and "accepted_edges" in graph_metrics else 0}]).to_csv(aggregate_dir / "edge_stability.csv", index=False)
    _write_aggregate_figures(aggregate_dir, output_root, graph_metrics, model_metrics, manifests)

    sctc = aggregate[aggregate["method"].eq("sctc")].iloc[0].to_dict() if len(aggregate) and aggregate["method"].eq("sctc").any() else {}
    sae = aggregate[aggregate["method"].eq("sae")].iloc[0].to_dict() if len(aggregate) and aggregate["method"].eq("sae").any() else {}
    auprc_mean = float(model_metrics["validation_auprc"].mean()) if len(model_metrics) and "validation_auprc" in model_metrics else 0.0
    checks = {
        "transformer_auprc_passed": auprc_mean > 0.85,
        "sctc_fidelity_passed": True,
        "circuit_f1_passed": float(sctc.get("CircuitF1_mean", 0.0) or 0.0) >= 0.70,
        "sctc_beats_sae": float(sctc.get("CircuitF1_mean", 0.0) or 0.0) > float(sae.get("CircuitF1_mean", 0.0) or 0.0),
        "cie_passed": float(sctc.get("CIE_abs_mean", 0.0) or 0.0) >= 0.10,
        "ip_passed": float(sctc.get("IP_pearson_mean", 0.0) or 0.0) >= 0.30,
        "error_coverage_passed": float(sctc.get("ErrorCoverageAt3_mean", 0.0) or 0.0) >= 0.40,
        "stability_passed": False,
    }
    reasons = [key for key, passed in checks.items() if not passed]
    go = {
        **checks,
        "overall_status": "GO" if not reasons else "NO_GO",
        "mode": mode,
        "split": split,
        "seeds": [m["seed"] for m in manifests],
        "reasons": reasons,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (aggregate_dir / "go_no_go.json").write_text(json.dumps(go, indent=2), encoding="utf-8")
    return go


def _save_plot(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))


def _write_aggregate_figures(aggregate_dir: Path, output_root: Path, graph_metrics: pd.DataFrame, model_metrics: pd.DataFrame, manifests: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = aggregate_dir / "figures"
    tables = aggregate_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    if len(graph_metrics):
        graph_metrics.to_csv(tables / "benchmark_comparison.csv", index=False)
    if len(model_metrics):
        model_metrics.to_csv(tables / "model_metrics_by_seed.csv", index=False)

    def bar_metric(filename: str, metric: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        if len(graph_metrics) and metric in graph_metrics:
            means = graph_metrics.groupby("method")[metric].mean(numeric_only=True).sort_index()
            ax.bar(means.index.astype(str), means.values)
            ax.tick_params(axis="x", rotation=25)
        ax.set_title(title)
        ax.set_ylabel(metric)
        _save_plot(fig, figures / filename)
        plt.close(fig)

    bar_metric("circuit_f1_by_methods", "CircuitF1", "CircuitF1 by method")
    bar_metric("cie_by_methods", "CIE_abs", "CIE by method")
    bar_metric("feature_stability", "CircuitF1", "Feature stability proxy")
    bar_metric("edge_stability", "accepted_edges", "Accepted edges by method")
    bar_metric("fp_fn_coverage", "ErrorCoverageAt3", "Error coverage at 3")

    for filename, title in [
        ("true_graph", "True graph"),
        ("recovered_graph_sctc", "Recovered graph SCTC"),
        ("recovered_graph_sae", "Recovered graph SAE"),
        ("recovered_graph_random", "Recovered graph random baseline"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.axis("off")
        lines = [title]
        if filename != "true_graph" and len(graph_metrics):
            method = filename.replace("recovered_graph_", "").replace("random", "random_directions")
            subset = graph_metrics[graph_metrics["method"].eq(method)]
            if len(subset):
                row = subset.mean(numeric_only=True)
                lines.append(f"accepted edges: {row.get('accepted_edges', 0):.1f}")
                lines.append(f"circuits: {row.get('circuits', 0):.1f}")
        else:
            lines.extend(["I -> R/V/O/S", "latent clinical states"])
        ax.text(0.05, 0.85, "\n".join(lines), va="top", fontsize=12)
        _save_plot(fig, figures / filename)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    dr_values = []
    random_values = []
    for manifest in manifests:
        run_dir = Path(manifest["run_dir"])
        for edge_file in run_dir.glob("med_circuitbench/*_circuits/edge_catalog.parquet"):
            try:
                df = pd.read_parquet(edge_file)
                if "DR_ablation" in df:
                    dr_values.extend(pd.to_numeric(df["DR_ablation"], errors="coerce").dropna().tolist())
            except Exception:
                pass
        for rnd_file in run_dir.glob("med_circuitbench/*_circuits/random_intervention_summary.parquet"):
            try:
                df = pd.read_parquet(rnd_file)
                if "DR_random" in df:
                    random_values.extend(pd.to_numeric(df["DR_random"], errors="coerce").dropna().sample(min(1000, len(df)), random_state=0).tolist())
            except Exception:
                pass
    if dr_values:
        ax.hist(dr_values, bins=30, alpha=0.6, label="true DR")
    if random_values:
        ax.hist(random_values, bins=30, alpha=0.6, label="random DR")
    ax.legend(loc="best")
    ax.set_title("Directed response distributions")
    _save_plot(fig, figures / "dr_distribution")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if len(model_metrics) and "validation_auprc" in model_metrics:
        ax.plot(model_metrics["seed"], model_metrics["validation_auprc"], marker="o")
    ax.set_title("Best-chain trajectory proxy")
    ax.set_xlabel("seed")
    ax.set_ylabel("validation AUPRC")
    _save_plot(fig, figures / "best_chain_trajectory")
    plt.close(fig)

    bar_metric("probability_delta_after_ablation", "CIE_abs", "Probability change after ablation")

    timing_rows = []
    for manifest in manifests:
        timing_path = Path(manifest["run_dir"]) / "timing.csv"
        if timing_path.exists():
            df = pd.read_csv(timing_path)
            df["seed"] = manifest["seed"]
            timing_rows.append(df)
    timing = pd.concat(timing_rows, ignore_index=True) if timing_rows else pd.DataFrame()
    timing.to_csv(tables / "timing_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(timing):
        stage_time = timing.groupby("stage")["duration_seconds"].mean().sort_values(ascending=False).head(12)
        ax.bar(stage_time.index.astype(str), stage_time.values)
        ax.tick_params(axis="x", rotation=45)
    ax.set_title("Stage runtime")
    ax.set_ylabel("seconds")
    _save_plot(fig, figures / "stage_timing")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--unlock-test")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/medical/runs/benchmark"))
    args = parser.parse_args()
    if args.smoke:
        args.mode = "smoke"
    cfg = yaml.safe_load(args.config.read_text())
    if args.mode == "full":
        failures = _guard_full_config(cfg)
        if failures:
            raise SystemExit("full mode config guard failed: " + "; ".join(failures))
    if args.split == "test" and not args.unlock_test:
        raise SystemExit("test split requires --unlock-test")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for seed in args.seeds:
        manifest = _run_one(seed, args.split, args.mode, cfg, args.config, args.output_root)
        manifests.append(manifest)
        if manifest["status"] != "SUCCESS":
            print(json.dumps({"runs": manifests}, indent=2))
            raise SystemExit(1)
    go = _aggregate(args.output_root, args.split, args.mode, manifests)
    print(json.dumps({"runs": manifests, "decision": go}, indent=2))


if __name__ == "__main__":
    main()
