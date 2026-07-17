# PhysioNet 2012 Post-hoc Attribution Audit

Technical status: `POSTHOC_ATTRIBUTION_AUDIT_COMPLETE`.

The status records completion of the frozen protocol and does not encode whether
the scientific result is positive or negative. No model was retrained. The audit
uses 30 existing PlainTransformer checkpoints, all 600 frozen test episodes, 435
model pairs, a zero baseline for primary Integrated Gradients, and one fixed
stratified 32-episode background for GradientSHAP.

## Frozen input contract

The task template mentioned `[36,27]`, but the immutable PhysioNet checkpoints
accept `[48,119]`: 37 value, 37 mask, 37 delta-time, and 8 static channels. The
audit follows the checkpoint and preprocessing contract. It does not resample
time, remove channels, alter preprocessing, or retrain a model.

## Article mapping

| Article element | Source | Build command |
|---|---|---|
| Section 3.8 primary stability values | `TABLES/posthoc_article_table.csv` | `python scripts/medical/physionet2012/build_posthoc_article.py ...` |
| Model-level comparison and CI | `TABLES/posthoc_method_comparison.csv` | `python scripts/medical/physionet2012/run_posthoc_attribution_stability.py --device cuda` |
| IG completeness | `TABLES/posthoc_completeness.csv` | same audit command |
| Median-baseline sensitivity | `TABLES/posthoc_baseline_sensitivity.csv` | same audit command |
| Mapping controls | `TABLES/posthoc_mapping_controls.csv` | same audit command |
| Figure 8 | `FIGURES/posthoc_stability_distribution.svg` | same audit command |
| Supplementary heatmap | `FIGURES/posthoc_pairwise_heatmap.svg` | same audit command |

## Reproduce and verify

```bash
PYTHONPATH=src:. .venv/bin/python \
  scripts/medical/physionet2012/run_posthoc_attribution_stability.py \
  --device cuda

PYTHONPATH=src:. .venv/bin/python \
  scripts/medical/physionet2012/verify_posthoc_attribution_audit.py
```

`DATA/posthoc_episode_attributions.parquet` is a partitioned zstd Parquet
dataset with the required long schema. It is retained locally for audit but is
not included in the compact article-submission ZIP.
