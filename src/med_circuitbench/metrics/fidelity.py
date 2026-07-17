from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def probability_fidelity_metrics(base_prob: np.ndarray, replaced_prob: np.ndarray, y: np.ndarray) -> dict[str, float]:
    base_prob = np.asarray(base_prob, dtype=float)
    replaced_prob = np.asarray(replaced_prob, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) > 1:
        base_auroc = roc_auc_score(y, base_prob)
        repl_auroc = roc_auc_score(y, replaced_prob)
        base_auprc = average_precision_score(y, base_prob)
        repl_auprc = average_precision_score(y, replaced_prob)
    else:
        base_auroc = repl_auroc = 0.5
        base_auprc = repl_auprc = float(np.mean(y))
    return {
        "delta_auroc": float(abs(repl_auroc - base_auroc)),
        "delta_auprc": float(abs(repl_auprc - base_auprc)),
        "probability_error": float(np.mean(np.abs(replaced_prob - base_prob))),
    }
