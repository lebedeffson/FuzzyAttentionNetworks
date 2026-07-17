from __future__ import annotations

import pandas as pd

from conceptfan_realdata.posthoc_attribution import METRICS, summarize_pairwise


def test_article_summary_contains_declared_statistics() -> None:
    rows = []
    for pair in range(4):
        row = {
            "method": "integrated_gradients", "feature_space": "proxy_concept_v_only", "value_view": "signed",
            "model_a": f"run_{pair}", "model_b": f"run_{pair + 1}",
        }
        row.update({metric: pair / 10 for metric in METRICS})
        rows.append(row)
    summary = summarize_pairwise(pd.DataFrame(rows))
    assert set(summary.metric) == set(METRICS)
    assert {"mean", "std", "median", "iqr", "ci95_low", "ci95_high", "minimum", "maximum"}.issubset(summary.columns)

