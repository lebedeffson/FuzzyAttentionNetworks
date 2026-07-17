# Table and figure map

Retain the synthetic benchmark results as mechanistic evidence and add the real-data items below as external empirical evidence.

| Manuscript item | New source | Recommended placement | Proposed caption or update |
|---|---|---|---|
| Real-data cohort | `../tables/table_cohort.csv` | Methods, after dataset description | Patient-level frozen split and mortality characteristics for PhysioNet 2012 Set A. |
| Predictive results | `../tables/table_performance.csv` | Main results, new real-data subsection | Held-out predictive metrics across 30 crossed initialization and data-order seeds. |
| Calibration | `../tables/table_calibration.csv` | Main results or supplement | Calibration metrics before and after the calibration-split-selected transformation. |
| Explanation stability | `../tables/table_stability.csv` | Interpretability results | Episode-level cross-retraining agreement with hierarchical bootstrap intervals. |
| Robustness | `../tables/table_robustness.csv` | Robustness subsection | Deterministic perturbation sensitivity for ConceptFAN and the pure non-fuzzy ablation. |
| Residual signal | `../tables/table_leakage.csv` | Diagnostic supplement | Associative residual-signal localization across V, V+M, V+D and V+M+D inputs and controls. |
| StabilityReg comparison | `../tables/table_stability_reg.csv` | Interpretability results | Pre-specified AUPRC non-inferiority and contribution-stability comparison. |
| Figure 1 | `../figures/fig_real_pipeline.svg` | Methods | Real-data cohort, frozen split, temporal channels and proxy-concept pipeline. |
| Figure 2 | `../figures/fig_calibration.svg` | Main results | Held-out AUPRC/AUROC and patient-bootstrap reliability curves. |
| Figure 3 | `../figures/fig_stability_distribution.svg` | Interpretability results | Distribution of episode-level Spearman agreement across retraining pairs. |
| Figure 4 | `../figures/fig_fuzzy_robustness.svg` | Robustness subsection | AUPRC degradation under value noise and missingness perturbations. |
| Figure 5 | `../figures/fig_leakage_localization.svg` | Diagnostic supplement | Residual diagnostic AUPRC by input-channel ablation and negative control. |
| Figure 6 | `../figures/fig_stability_pareto.svg` | Interpretability results | Held-out AUPRC versus signed contribution stability for concept-mediated arms. |
