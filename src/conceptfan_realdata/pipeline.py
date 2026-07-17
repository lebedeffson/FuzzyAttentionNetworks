from __future__ import annotations

import gc
import itertools
import json
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from .audit import build_archive, validate_archive, validate_release
from .data import PreparedData, audit_raw, create_split, ensure_outcomes, load_prepared, prepare_dataset
from .leakage import analyze_residual_signal, summarize_leakage
from .models import MODEL_ARMS
from .reporting import build_report
from .robustness import aggregate_robustness, run_robustness
from .stability import aggregate_stability, build_episode_stability
from .statistics import (
    calibration_summary,
    cohort_table,
    collect_run_metrics,
    concept_quality_metrics,
    paired_model_comparisons,
    performance_summary,
    seed_variance_decomposition,
)
from .sufficiency import build_sufficiency, summarize_sufficiency
from .training import select_stability_coefficients, train_run


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def _split_frame_from_files(patient_manifest: pd.DataFrame, split_dir: Path) -> pd.DataFrame:
    mapping: dict[int, str] = {}
    for split in ["train", "validation", "calibration", "test"]:
        for value in (split_dir / f"{split}_ids.txt").read_text(encoding="utf-8").splitlines():
            if value.strip():
                mapping[int(value)] = split
    frame = patient_manifest.rename(columns={"record_id": "RecordID", "icu_type": "ICUType"})
    frame["split"] = frame["RecordID"].map(mapping)
    if frame["split"].isna().any() or len(mapping) != 4000:
        raise ValueError("Frozen split files do not map all 4000 patients")
    return frame[["RecordID", "ICUType", "In-hospital_death", "split"]]


def prepare_all(set_zip: Path, outcomes: Path, artifacts_root: Path, config: dict) -> PreparedData:
    outcomes = ensure_outcomes(outcomes, config["data"]["outcomes_url"])
    raw_dir = artifacts_root / "raw_audit"
    audit_path = raw_dir / "audit.json"
    if not audit_path.exists():
        audit_raw(set_zip, outcomes, raw_dir, int(config["data"]["expected_patients"]))
    else:
        report = json.loads(audit_path.read_text(encoding="utf-8"))
        if report.get("status") != "PHYSIONET2012_RAW_AUDIT_PASS":
            raise RuntimeError("Existing raw audit is not a pass")
    patient_manifest = pd.read_parquet(raw_dir / "patient_manifest.parquet")
    split_dir = artifacts_root / "splits"
    if not next(split_dir.glob("split_seed_*.json"), None):
        split_frame = create_split(patient_manifest, split_dir, int(config["program"]["split_seed"]))
    else:
        split_frame = _split_frame_from_files(patient_manifest, split_dir)
    prepared_path = artifacts_root / "prepared" / "prepared_physionet2012.npz"
    if prepared_path.exists():
        data = load_prepared(prepared_path)
    else:
        data = prepare_dataset(set_zip, outcomes, split_frame, config, artifacts_root / "prepared")
    expected = {"train": 2400, "validation": 600, "calibration": 400, "test": 600}
    if {split: len(data.indices(split)) for split in expected} != expected:
        raise AssertionError("Prepared split counts changed")
    return data


def _preflight(data: PreparedData, config: dict, artifacts_root: Path, device: torch.device, batch_size: int) -> None:
    root = artifacts_root / "preflight"
    statuses = []
    for index, arm in enumerate(MODEL_ARMS):
        run_dir = root / arm
        selected = {"lambda_stab": 0.01, "lambda_logit": 0.0} if arm == "ConceptFAN-StabilityReg" else None
        manifest = train_run(
            arm,
            data,
            config,
            run_dir,
            init_seed=9001 + index,
            data_order_seed=9901,
            channels="V+M+D",
            device=device,
            batch_size=batch_size,
            selected_stability=selected,
            max_epochs_override=2,
            validation_only=True,
        )
        statuses.append(manifest["status"])
        forbidden = [run_dir / "logits_calibration.parquet", run_dir / "logits_test.parquet", run_dir / "metrics_test.json"]
        if any(path.exists() for path in forbidden):
            raise AssertionError(f"Preflight selection leaked a frozen split for {arm}")
    report = {
        "status": "PHYSIONET2012_PREFLIGHT_PASS",
        "created_utc": utc_now(),
        "models": MODEL_ARMS,
        "epochs": 2,
        "evaluation_scope": "train_validation_only",
        "run_statuses": statuses,
    }
    (root / "preflight_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def _main_specs(config: dict) -> list[dict]:
    pairs = list(itertools.product(config["training"]["init_seeds"], config["training"]["data_order_seeds"]))
    return [
        {
            "arm": arm,
            "channels": "V+M+D",
            "init_seed": int(init_seed),
            "data_order_seed": int(order_seed),
            "run_name": f"run_{position:02d}_init_{init_seed}_order_{order_seed}",
            "kind": "main",
        }
        for arm in MODEL_ARMS
        for position, (init_seed, order_seed) in enumerate(pairs, start=1)
    ]


def _ablation_specs(config: dict) -> list[dict]:
    specs = []
    for channels in ["V", "V+M", "V+D"]:
        for position, init_seed in enumerate(config["training"]["leakage_init_seeds"], start=1):
            specs.append(
                {
                    "arm": "ConceptFAN-NoAlpha",
                    "channels": channels,
                    "init_seed": int(init_seed),
                    "data_order_seed": int(config["training"]["leakage_data_order_seed"]),
                    "run_name": f"run_{position:02d}_init_{init_seed}_order_{config['training']['leakage_data_order_seed']}",
                    "kind": "channel_ablation",
                }
            )
    return specs


def _run_dir(artifacts_root: Path, spec: dict) -> Path:
    if spec["kind"] == "main":
        return artifacts_root / "runs" / spec["arm"] / spec["run_name"]
    return artifacts_root / "runs" / "channel_ablation" / spec["channels"].replace("+", "_") / spec["run_name"]


def _progress(artifacts_root: Path, specs: list[dict], failures: list[dict], started: float) -> None:
    completed = 0
    for spec in specs:
        manifest = _run_dir(artifacts_root, spec) / "run_manifest.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")).get("status") == "RUN_COMPLETE":
            completed += 1
    payload = {
        "status": "RUNNING" if completed + len(failures) < len(specs) else "QUEUE_TERMINAL",
        "updated_utc": utc_now(),
        "total": len(specs),
        "completed": completed,
        "failed": len(failures),
        "pending": len(specs) - completed - len(failures),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (artifacts_root / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def execute_queue(
    data: PreparedData,
    config: dict,
    artifacts_root: Path,
    device: torch.device,
    selected_stability: dict[str, float],
) -> list[dict]:
    specs = _main_specs(config) + _ablation_specs(config)
    failures: list[dict] = []
    started = time.perf_counter()
    _progress(artifacts_root, specs, failures, started)
    default_batch = int(config["training"]["batch_size"])
    max_retries = int(config["training"]["max_retries"])
    for queue_position, spec in enumerate(specs, start=1):
        run_dir = _run_dir(artifacts_root, spec)
        existing = run_dir / "run_manifest.json"
        if existing.exists() and json.loads(existing.read_text(encoding="utf-8")).get("status") == "RUN_COMPLETE":
            _progress(artifacts_root, specs, failures, started)
            continue
        batch_size = default_batch
        errors: list[dict] = []
        completed = False
        for attempt in range(max_retries + 1):
            try:
                manifest = train_run(
                    spec["arm"],
                    data,
                    config,
                    run_dir,
                    spec["init_seed"],
                    spec["data_order_seed"],
                    spec["channels"],
                    device,
                    batch_size,
                    selected_stability if spec["arm"] == "ConceptFAN-StabilityReg" else None,
                )
                manifest["queue_position"] = queue_position
                manifest["queue_total"] = len(specs)
                manifest["batch_size"] = batch_size
                manifest["batch_size_reduced_after_oom"] = batch_size < default_batch
                existing.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                completed = True
                break
            except torch.cuda.OutOfMemoryError as exc:
                errors.append({"attempt": attempt + 1, "type": type(exc).__name__, "message": str(exc), "batch_size": batch_size})
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                batch_size = max(1, batch_size // 2)
            except Exception as exc:
                errors.append(
                    {
                        "attempt": attempt + 1,
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
        if not completed:
            failure = {"spec": spec, "run_dir": str(run_dir), "errors": errors}
            failures.append(failure)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        _progress(artifacts_root, specs, failures, started)
    (artifacts_root / "failure_report.json").write_text(
        json.dumps({"status": "NO_RUN_FAILURES" if not failures else "RUN_FAILURES_PRESENT", "failures": failures}, indent=2),
        encoding="utf-8",
    )
    return failures


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    frame.to_csv(path.with_suffix(".csv"), index=False)


def _concept_examples(data: PreparedData, output_path: Path) -> None:
    test = data.indices("test")[:6]
    fig, axes = plt.subplots(len(test), 1, figsize=(10, 11), sharex=True)
    for axis, position in zip(axes, test):
        for concept, name in enumerate(data.concept_names):
            values = np.where(data.concept_mask[position, :, concept] > 0, data.concepts[position, :, concept], np.nan)
            axis.plot(values, label=name if position == test[0] else None, linewidth=1.1)
        axis.set_ylabel(str(data.record_ids[position]))
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.15)
    axes[-1].set_xlabel("Hour")
    axes[0].legend(ncol=3, fontsize=7, frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_posthoc(data: PreparedData, config: dict, artifacts_root: Path, device: torch.device) -> None:
    tables = artifacts_root / "TABLES"
    tables.mkdir(parents=True, exist_ok=True)
    runs_root = artifacts_root / "runs"
    repetitions = int(config["statistics"]["bootstrap_repetitions"])
    predictive, calibration, fit = collect_run_metrics(runs_root)
    if len(fit) != 150:
        raise RuntimeError(f"Post-hoc requires 150 completed main runs, found {len(fit)}")
    _write(predictive, tables / "predictive_metrics.parquet")
    _write(calibration, tables / "calibration_metrics.parquet")
    _write(fit, tables / "checkpoint_fit_metrics.parquet")
    _write(performance_summary(predictive, repetitions), tables / "performance_summary.parquet")
    _write(calibration_summary(predictive, repetitions), tables / "calibration_summary.parquet")
    _write(paired_model_comparisons(predictive), tables / "paired_model_comparisons.parquet")
    _write(seed_variance_decomposition(predictive), tables / "seed_variance_decomposition.parquet")
    _write(concept_quality_metrics(data, runs_root), tables / "concept_quality_metrics.parquet")
    _write(cohort_table(data), tables / "cohort.parquet")
    episode_path = tables / "episode_pairwise_stability.parquet"
    pair_summary_path = tables / "stability_pair_summary.parquet"
    if not episode_path.exists() or not pair_summary_path.exists():
        pair_summary, pair_counts = build_episode_stability(runs_root, episode_path)
        _write(pair_summary, pair_summary_path)
        (tables / "stability_pair_counts.json").write_text(json.dumps(pair_counts, indent=2), encoding="utf-8")
    else:
        pair_summary = pd.read_parquet(pair_summary_path)
    aggregate = aggregate_stability(
        episode_path, pair_summary, int(config["statistics"]["hierarchical_bootstrap_repetitions"])
    )
    _write(aggregate, tables / "stability_aggregate.parquet")
    exhaustive = tables / "exhaustive_32_mask_sufficiency.parquet"
    controls = tables / "sufficiency_controls.parquet"
    if not exhaustive.exists() or not controls.exists():
        build_sufficiency(runs_root, exhaustive, controls)
    exhaustive_summary, controls_summary = summarize_sufficiency(exhaustive, controls)
    _write(exhaustive_summary, tables / "sufficiency_exhaustive_summary.parquet")
    _write(controls_summary, tables / "sufficiency_controls_summary.parquet")
    robustness_raw_path = tables / "robustness_detailed.parquet"
    if robustness_raw_path.exists():
        robustness_raw = pd.read_parquet(robustness_raw_path)
    else:
        robustness_raw = run_robustness(data, config, runs_root, robustness_raw_path, device, int(config["training"]["batch_size"]))
    _write(aggregate_robustness(robustness_raw), tables / "robustness_aggregate.parquet")
    leakage_path = tables / "residual_signal_diagnostics.parquet"
    importance_path = tables / "residual_group_permutation_importance.parquet"
    if leakage_path.exists() and importance_path.exists():
        leakage_raw = pd.read_parquet(leakage_path)
    else:
        leakage_raw, _ = analyze_residual_signal(data, runs_root, leakage_path, importance_path, min(repetitions, 1000))
    _write(summarize_leakage(leakage_raw), tables / "leakage_aggregate.parquet")
    selection_grid = artifacts_root / "stability_selection" / "stability_selection_grid.csv"
    if selection_grid.exists():
        frame = pd.read_csv(selection_grid)
        _write(frame, tables / "stability_selection_grid.parquet")
    _concept_examples(data, artifacts_root / "concepts" / "concept_examples.pdf")


def run_full_pipeline(
    repo_root: Path,
    config_path: Path,
    set_zip: Path,
    outcomes: Path,
    artifacts_root: Path,
    report_dir: Path,
    device_name: str,
    resume: bool,
    dry_run: bool = False,
) -> dict:
    del resume
    config = load_config(config_path)
    device = resolve_device(device_name)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    if dry_run:
        specs = _main_specs(config) + _ablation_specs(config)
        return {"status": "DRY_RUN", "runs": len(specs), "main": 150, "channel_ablation": 30}
    started = time.perf_counter()
    data = prepare_all(set_zip, outcomes, artifacts_root, config)
    _preflight(data, config, artifacts_root, device, int(config["training"]["batch_size"]))
    selected = select_stability_coefficients(
        data, config, artifacts_root / "stability_selection", device, int(config["training"]["batch_size"])
    )
    failures = execute_queue(data, config, artifacts_root, device, selected)
    if failures:
        raise RuntimeError(f"{len(failures)} runs failed after retries; see failure_report.json")
    run_posthoc(data, config, artifacts_root, device)
    report_manifest = build_report(data, artifacts_root, report_dir)
    audit_path = report_dir / "audit" / "readonly_validation.json"
    audit = validate_release(artifacts_root, report_dir, audit_path)
    if not audit["passed"]:
        raise RuntimeError(f"Read-only validation failed: {audit['failed_checks']}")
    (artifacts_root / "audit").mkdir(parents=True, exist_ok=True)
    shutil.copy2(audit_path, artifacts_root / "audit" / "readonly_validation.json")
    archive = repo_root / "reports" / "PHYSIONET2012_CONCEPTFAN_LAST_RUN_e60d9a3.zip"
    archive_result = build_archive(repo_root, artifacts_root, report_dir, archive)
    archive_validation = validate_archive(archive, report_dir / "audit" / "archive_validation.json")
    if not archive_validation["passed"]:
        raise RuntimeError(f"Archive validation failed: {archive_validation}")
    resource = {
        "status": "PHYSIONET2012_LAST_RUN_RESOURCE_REPORT",
        "created_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "main_runs": 150,
        "channel_ablation_runs": 30,
        "run_failures": 0,
        "archive": archive_result,
    }
    (artifacts_root / "resource_report.json").write_text(json.dumps(resource, indent=2), encoding="utf-8")
    final = {
        "status": "PHYSIONET2012_LAST_RUN_COMPLETE",
        "created_utc": utc_now(),
        "report": report_manifest,
        "readonly_audit": audit["status"],
        "archive_validation": archive_validation["status"],
        "archive": archive_result,
        "resource_report": resource,
    }
    (artifacts_root / "FINAL_MANIFEST.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return final
