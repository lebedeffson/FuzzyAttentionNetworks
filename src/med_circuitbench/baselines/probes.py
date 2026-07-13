from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def logistic_probe_auc(x: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x, y)
    return float(roc_auc_score(y, clf.predict_proba(x)[:, 1]))
