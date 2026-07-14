# Current Project State

Last updated: 2026-07-14

## Active Branch

```text
fix/med-circuitbench-v2-2-final
```

## Latest Result Commit

```text
473bd7059131e5275b73a5e95025ab4994e03fb6
```

## Delivery Commit

Recorded by `git rev-parse HEAD`, `GIT_INFO.txt`, and the delivery manifest at packaging time.

## Final Status

```text
FAN_FOUNDATION_FAIL
```

## Key Metrics

- Oracle Temporal FAN mean AUPRC: `0.8369`
- Predicted Temporal FAN Strict mean AUPRC: `0.8190`
- Planted CircuitF1: `1.0000`
- Standard SCTC delta AUPRC: `0.0005`
- Standard SCTC probability MAE: `0.0002`

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
