from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def equal_mass_ece(y: np.ndarray, probability: np.ndarray, n_bins: int = 15) -> float:
    y = np.asarray(y, dtype=np.int8)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    order = np.argsort(probability, kind="mergesort")
    bins = np.array_split(order, min(n_bins, len(order)))
    return float(sum((len(index) / len(y)) * abs(float(y[index].mean()) - float(probability[index].mean())) for index in bins if len(index)))


def calibration_slope_intercept(y: np.ndarray, logits: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=np.int8)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
    if np.unique(y).size < 2 or np.std(logits) < 1e-12:
        return float("nan"), float("nan")
    model = LogisticRegression(C=1e8, solver="lbfgs", max_iter=2000)
    model.fit(logits, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def binary_metrics(y: np.ndarray, probability: np.ndarray, logits: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int8)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    if logits is None:
        logits = np.log(probability / (1.0 - probability))
    slope, intercept = calibration_slope_intercept(y, np.asarray(logits))
    return {
        "AUROC": float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else float("nan"),
        "AUPRC": float(average_precision_score(y, probability)),
        "Brier": float(brier_score_loss(y, probability)),
        "NLL": float(log_loss(y, probability, labels=[0, 1])),
        "ECE10": equal_mass_ece(y, probability, 10),
        "ECE15": equal_mass_ece(y, probability, 15),
        "ECE20": equal_mass_ece(y, probability, 20),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    repetitions: int = 2000,
    seed: int = 20260717,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        estimates[index] = statistic(rng.choice(values, size=len(values), replace=True))
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))


def paired_rank_biserial(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if len(differences) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(differences))
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()
    return float((positive - negative) / (positive + negative))


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    count = len(p)
    for rank, position in enumerate(order):
        candidate = min(1.0, (count - rank) * p[position])
        running = max(running, candidate)
        adjusted[position] = running
    return adjusted.tolist()


def summarize_runs(frame: pd.DataFrame, group_columns: list[str], metric_columns: list[str], repetitions: int = 2000) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = frame.groupby(group_columns, dropna=False, sort=True)
    for group_values, group in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        base = dict(zip(group_columns, group_values))
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            if len(values) == 0:
                continue
            low, high = bootstrap_ci(values, repetitions=repetitions)
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)
