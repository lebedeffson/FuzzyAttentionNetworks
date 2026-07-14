# Current Project State

Last updated: 2026-07-14

## Active Branch

```text
fix/med-circuitbench-v2-2-final
```

## Current Commit

```text
d404aabf8d93844e11878263103bbdb5d2ab16bc
```

## Final Status

```text
FAN_FOUNDATION_FAIL
```

## Key Metrics

- Oracle Temporal FAN mean AUPRC: `0.9029`
- Predicted Temporal FAN Strict mean AUPRC: `0.8321`
- Planted CircuitF1: `1.0000`
- Standard SCTC delta AUPRC: `0.0125`
- Standard SCTC probability MAE: `0.0308`

## Completed Stages

- Full-run guard.
- Concept sufficiency audit.
- Concept-mediated FAN diagnostics.
- Real planted raw activations and interventions.
- Representation audit.
- Standard Transformer + SCTC fidelity with replacement forward.
- Three-seed aggregation and delivery validation.

## Skipped Stages

- FAN + SCTC is `SKIPPED_BY_GATE` if FAN Foundation Gate fails.

## Next Scientific Step

Use the raw FAN gate table and Standard+SCTC fidelity outputs to decide whether to improve FAN evidence compression or continue with Standard-only SCTC.
