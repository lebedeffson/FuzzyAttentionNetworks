from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .data import PreparedData
from .metrics import holm_adjust, paired_rank_biserial, summarize_runs
from .models import MODEL_ARMS


PREDICTIVE_METRICS = [
    "AUROC",
    "AUPRC",
    "Brier",
    "NLL",
    "ECE10",
    "ECE15",
    "ECE20",
    "calibration_slope",
    "calibration_intercept",
]


def collect_run_metrics(runs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictive_rows: list[dict] = []
    fit_rows: list[dict] = []
    calibration_rows: list[dict] = []
    for arm in MODEL_ARMS:
        for run_dir in sorted((runs_root / arm).glob("run_*")):
            manifest_path = run_dir / "run_manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "RUN_COMPLETE":
                continue
            test = json.loads((run_dir / "metrics_test.json").read_text(encoding="utf-8"))
            calibration = json.loads((run_dir / "metrics_calibration.json").read_text(encoding="utf-8"))
            primary = test["primary_method"]
            common = {
                "model_arm": arm,
                "run_id": run_dir.name,
                "init_seed": manifest["init_seed"],
                "data_order_seed": manifest["data_order_seed"],
            }
            for method, values in test["metrics"].items():
                predictive_rows.append({**common, "calibration_method": method, "is_primary": method == primary, **values})
            for method, values in calibration["metrics"].items():
                calibration_rows.append(
                    {
                        **common,
                        "calibration_method": method,
                        "is_primary": method == primary,
                        "selection_nll": calibration["selection_nll"].get(method),
                        **values,
                    }
                )
            fit_rows.append(
                {
                    **common,
                    "channels": manifest["channels"],
                    "best_epoch": manifest["best_epoch"],
                    "epochs_completed": manifest["epochs_completed"],
                    "best_validation_AUPRC": manifest["best_validation_AUPRC"],
                    "selected_calibrator": primary,
                    "parameter_sha256": manifest["parameter_sha256"],
                    "checkpoint_sha256": manifest["checkpoint_sha256"],
                    "max_decomposition_error": manifest["max_decomposition_error"],
                    "elapsed_seconds": manifest["elapsed_seconds"],
                    "evaluation_scope": manifest.get("evaluation_scope"),
                    "test_used_for_selection": manifest.get("test_used_for_selection", False),
                    "test_read_after_training_and_calibration_selection": manifest.get(
                        "test_read_after_training_and_calibration_selection", False
                    ),
                }
            )
    return pd.DataFrame(predictive_rows), pd.DataFrame(calibration_rows), pd.DataFrame(fit_rows)


def performance_summary(predictive: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    selected = predictive.loc[predictive["calibration_method"].eq("none") | predictive["is_primary"]].copy()
    selected["reporting_state"] = np.where(selected["calibration_method"].eq("none"), "raw", "primary_calibrated")
    selected = selected.drop_duplicates(["model_arm", "run_id", "reporting_state"], keep="last")
    return summarize_runs(
        selected,
        ["model_arm", "reporting_state"],
        [metric for metric in PREDICTIVE_METRICS if metric in selected],
        repetitions,
    )


def calibration_summary(predictive: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    return summarize_runs(
        predictive,
        ["model_arm", "calibration_method", "is_primary"],
        [metric for metric in ["Brier", "NLL", "ECE10", "ECE15", "ECE20", "calibration_slope", "calibration_intercept"] if metric in predictive],
        repetitions,
    )


def paired_model_comparisons(predictive: pd.DataFrame) -> pd.DataFrame:
    raw = predictive.loc[predictive["calibration_method"].eq("none")]
    rows: list[dict] = []
    for comparator in [arm for arm in MODEL_ARMS if arm != "ConceptFAN-NoAlpha"]:
        left = raw.loc[raw["model_arm"].eq("ConceptFAN-NoAlpha")]
        right = raw.loc[raw["model_arm"].eq(comparator)]
        paired = left.merge(right, on=["init_seed", "data_order_seed"], suffixes=("_conceptfan", "_comparator"))
        for metric in ["AUPRC", "AUROC", "Brier", "NLL"]:
            differences = paired[f"{metric}_conceptfan"].to_numpy() - paired[f"{metric}_comparator"].to_numpy()
            if len(differences) == 0:
                continue
            test = stats.wilcoxon(differences, zero_method="zsplit", alternative="two-sided")
            rows.append(
                {
                    "reference": "ConceptFAN-NoAlpha",
                    "comparator": comparator,
                    "metric": metric,
                    "pairs": len(differences),
                    "median_difference_reference_minus_comparator": float(np.median(differences)),
                    "rank_biserial": paired_rank_biserial(differences),
                    "wilcoxon_statistic": float(test.statistic),
                    "p_value": float(test.pvalue),
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["p_value_holm"] = holm_adjust(frame["p_value"].tolist())
    return frame


def seed_variance_decomposition(predictive: pd.DataFrame) -> pd.DataFrame:
    raw = predictive.loc[predictive["calibration_method"].eq("none")]
    rows: list[dict] = []
    for arm, group in raw.groupby("model_arm"):
        for metric in ["AUPRC", "AUROC", "Brier", "NLL"]:
            table = group.pivot(index="init_seed", columns="data_order_seed", values=metric)
            if table.isna().any().any() or table.shape[0] < 2 or table.shape[1] < 2:
                continue
            values = table.to_numpy(dtype=np.float64)
            grand = values.mean()
            init_effect = values.mean(axis=1) - grand
            order_effect = values.mean(axis=0) - grand
            residual = values - grand - init_effect[:, None] - order_effect[None, :]
            components = {
                "initialization_seed": float(np.var(init_effect, ddof=1)),
                "data_order_seed": float(np.var(order_effect, ddof=1)),
                "interaction_residual": float(np.var(residual, ddof=1)),
            }
            total = sum(components.values())
            for component, variance in components.items():
                rows.append(
                    {
                        "model_arm": arm,
                        "metric": metric,
                        "component": component,
                        "variance": variance,
                        "variance_fraction": variance / total if total > 0 else 0.0,
                        "init_seeds": table.shape[0],
                        "data_order_seeds": table.shape[1],
                    }
                )
    return pd.DataFrame(rows)


def concept_quality_metrics(data: PreparedData, runs_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    lookup = {int(record_id): position for position, record_id in enumerate(data.record_ids)}
    for arm in ["ConceptFAN-NoAlpha", "PureNoFuzzy", "ConceptFAN-StabilityReg", "TemporalCEM"]:
        for run_dir in sorted((runs_root / arm).glob("run_*")):
            path = run_dir / "concepts_test.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            positions = np.asarray([lookup[int(record_id)] for record_id in frame["RecordID"]])
            for concept, name in enumerate(data.concept_names):
                prediction = np.stack(frame[f"concept_{concept}_trajectory"].map(np.asarray)).astype(np.float64)
                target = data.concepts[positions, :, concept].astype(np.float64)
                mask = data.concept_mask[positions, :, concept].astype(bool)
                observed_prediction = prediction[mask]
                observed_target = target[mask]
                correlation = stats.spearmanr(observed_target, observed_prediction).statistic if len(observed_target) > 2 else np.nan
                rows.append(
                    {
                        "model_arm": arm,
                        "run_id": run_dir.name,
                        "concept": name,
                        "observations": int(mask.sum()),
                        "MAE": float(np.mean(np.abs(observed_prediction - observed_target))),
                        "RMSE": float(np.sqrt(np.mean((observed_prediction - observed_target) ** 2))),
                        "Spearman": float(correlation),
                    }
                )
    return pd.DataFrame(rows)


def cohort_table(data: PreparedData) -> pd.DataFrame:
    rows: list[dict] = []
    for split in ["train", "validation", "calibration", "test", "all"]:
        indices = np.arange(len(data.y)) if split == "all" else data.indices(split)
        rows.append(
            {
                "split": split,
                "patients": len(indices),
                "deaths": int(data.y[indices].sum()),
                "mortality_rate": float(data.y[indices].mean()),
                "observed_temporal_fraction": float(data.m[indices].mean()),
                "proxy_concept_observability": float(data.concept_mask[indices].mean()),
            }
        )
    return pd.DataFrame(rows)
