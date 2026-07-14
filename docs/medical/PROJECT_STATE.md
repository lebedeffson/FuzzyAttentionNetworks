# Current Project State

Last updated: 2026-07-14

## Active Branch

```text
fix/med-circuitbench-v2-2-final
```

## Current Commit

```text
a142bd7cef7f6eac7e5b24c32bdb21135e442207
```

## Final Status

```text
FAN_FOUNDATION_FAIL
```

This status is from the corrected V2.2 runner that trains and evaluates the
neural `TemporalConceptFANModel`, not the earlier Ridge/LogisticRegression
surrogate FAN path.

## Key Metrics

- Oracle Temporal FAN mean AUPRC: `0.1390`
- Predicted Temporal FAN Strict mean AUPRC: `0.6011`
- Planted CircuitF1: `1.0000`
- Standard SCTC delta AUPRC: `0.0004`
- Standard SCTC probability MAE: `0.0009`

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
