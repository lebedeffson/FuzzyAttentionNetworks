from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score


def binary_classification_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(target).astype(int)
    p = np.asarray(probability).astype(float)
    pred = (p >= 0.5).astype(int)
    return {
        "AUROC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "AUPRC": float(average_precision_score(y, p)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Brier": float(brier_score_loss(y, p)),
        "ECE": float(_ece(y, p)),
    }


def _ece(target: np.ndarray, probability: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total = len(target)
    if total == 0:
        return float("nan")
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probability >= lo) & (probability < hi if hi < 1.0 else probability <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(target[mask].mean()) - float(probability[mask].mean()))
    return ece
