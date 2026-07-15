from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .concepts import CONCEPT_NAMES, compute_window_concepts
from .io import MimicSource
from .splits import split_subjects


FEATURE_COLUMNS = [
    "creatinine_first",
    "creatinine_last",
    "creatinine_mean",
    "creatinine_max",
    "creatinine_min",
    "creatinine_delta",
    "creatinine_slope_per_event",
    "creatinine_count",
    *[f"concept_{name}" for name in CONCEPT_NAMES],
    *[f"concept_mask_{name}" for name in CONCEPT_NAMES],
]


def build_demo_window_feature_table(
    source: MimicSource,
    creatinine: pd.DataFrame,
    windows: pd.DataFrame,
    observation_hours: int = 24,
) -> pd.DataFrame:
    """Build a small real-format feature table from the MIMIC-IV demo.

    This is intentionally simple and demo-only. It verifies cohort, labels,
    split isolation, feature extraction, concepts, and prediction wiring without
    pretending that 100 demo patients are enough for a scientific model.
    """
    stays = source.read_csv("icu/icustays.csv.gz", usecols=["subject_id", "stay_id"])
    windows = windows.merge(stays, on="stay_id", how="left")
    creat = creatinine.copy()
    creat["charttime"] = pd.to_datetime(creat["charttime"])
    rows = []
    for row in windows.itertuples(index=False):
        window_end = pd.Timestamp(row.window_end)
        start = window_end - pd.Timedelta(hours=observation_hours)
        observed = creat[
            creat["stay_id"].eq(row.stay_id)
            & (creat["charttime"] > start)
            & (creat["charttime"] <= window_end)
        ].sort_values("charttime")
        if len(observed) == 0:
            continue
        values = observed["creatinine"].astype(float).to_numpy()
        baseline = max(float(np.nanmin(values)), 1e-6)
        concepts, concept_mask = compute_window_concepts(observed[["creatinine"]], {"baseline_creatinine": baseline})
        feature_row = {
            "stay_id": int(row.stay_id),
            "subject_id": int(row.subject_id),
            "window_end": window_end,
            "label": int(row.label),
            "creatinine_first": float(values[0]),
            "creatinine_last": float(values[-1]),
            "creatinine_mean": float(np.nanmean(values)),
            "creatinine_max": float(np.nanmax(values)),
            "creatinine_min": float(np.nanmin(values)),
            "creatinine_delta": float(values[-1] - values[0]),
            "creatinine_slope_per_event": float((values[-1] - values[0]) / max(len(values) - 1, 1)),
            "creatinine_count": int(len(values)),
        }
        for name, value, mask in zip(CONCEPT_NAMES, concepts, concept_mask, strict=True):
            feature_row[f"concept_{name}"] = float(value)
            feature_row[f"concept_mask_{name}"] = float(mask)
        rows.append(feature_row)
    return pd.DataFrame(rows)


def _binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else None,
        "auroc": None,
        "auprc": None,
        "brier": None,
    }
    if len(y_true) == 0:
        return out
    out["brier"] = float(brier_score_loss(y_true, probability))
    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, probability))
        out["auprc"] = float(average_precision_score(y_true, probability))
    return out


def run_demo_logistic_baseline(feature_table: pd.DataFrame, seed: int = 20260715) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if feature_table.empty:
        raise ValueError("feature_table is empty")
    if "split" in feature_table.columns:
        data = feature_table.copy()
    else:
        split = split_subjects(feature_table["subject_id"].unique(), seed=seed)
        data = feature_table.merge(split, on="subject_id", how="left")
    x = data[FEATURE_COLUMNS].astype("float32").to_numpy()
    y = data["label"].astype(int).to_numpy()
    train_mask = data["split"].eq("train").to_numpy()
    if len(np.unique(y[train_mask])) < 2:
        # Demo data can be tiny after split. Fall back to a calibrated prevalence
        # predictor, and record it explicitly in the metrics.
        prevalence = float(np.mean(y[train_mask])) if train_mask.any() else float(np.mean(y))
        prob = np.full(len(data), prevalence, dtype="float32")
        model_type = "train_prevalence_constant"
    else:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
        )
        model.fit(x[train_mask], y[train_mask])
        prob = model.predict_proba(x)[:, 1].astype("float32")
        model_type = "logistic_regression_balanced"
    pred = data[["stay_id", "subject_id", "window_end", "label", "split"]].copy()
    pred["probability"] = prob
    pred["prediction"] = (pred["probability"] >= 0.5).astype(int)
    metrics_rows = []
    for split_name in ["train", "validation", "test", "all"]:
        mask = np.ones(len(pred), dtype=bool) if split_name == "all" else pred["split"].eq(split_name).to_numpy()
        split_metrics = _binary_metrics(y[mask], prob[mask])
        split_metrics["split"] = split_name
        split_metrics["model"] = model_type
        metrics_rows.append(split_metrics)
    metrics = {
        "status": "DEMO_BASELINE_COMPLETE",
        "model": model_type,
        "feature_rows": int(len(feature_table)),
        "feature_columns": FEATURE_COLUMNS,
        "split_counts": pred["split"].value_counts().to_dict(),
        "label_counts": {str(k): int(v) for k, v in pred["label"].value_counts().sort_index().items()},
        "note": "MIMIC-IV Demo baseline is a real-format wiring check, not a full scientific estimate.",
    }
    return pred, pd.DataFrame(metrics_rows), metrics


def metrics_to_jsonable(metrics: dict) -> dict:
    clean = {}
    for key, value in metrics.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            clean[key] = None
        else:
            clean[key] = value
    return clean
