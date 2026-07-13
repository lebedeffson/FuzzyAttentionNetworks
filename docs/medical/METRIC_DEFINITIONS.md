# Metric Definitions

- Fidelity: `delta_auroc`, `delta_auprc`, and `probability_error` compare frozen-transformer base predictions with replacement predictions.
- CIE: mean absolute and signed probability change after circuit intervention.
- IP: Pearson correlation between chain strength and absolute intervention effect.
- Completeness: top-circuit effect divided by all-eligible-feature effect.
- OTE: normalized change in eligible nodes outside the circuit.
- CircuitF1: layer-aware graph recovery on `(layer, feature_id)` nodes.
