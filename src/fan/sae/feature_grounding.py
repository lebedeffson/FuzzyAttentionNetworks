from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def feature_concept_correlations(features: np.ndarray, concepts: np.ndarray, concept_names: list[str]) -> pd.DataFrame:
    rows = []
    for j in range(features.shape[1]):
        for k, name in enumerate(concept_names):
            f, c = features[:, j], concepts[:, k]
            if np.std(f) == 0 or np.std(c) == 0:
                rho, p = np.nan, np.nan
            else:
                res = stats.spearmanr(f, c)
                rho, p = float(res.statistic), float(res.pvalue)
            rows.append({"feature": j, "concept": name, "spearman": rho, "p_value": p})
    return pd.DataFrame(rows)
