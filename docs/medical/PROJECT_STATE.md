# Current Project State

Last updated: 2026-07-14

## Active Branch

```text
experiment/med-circuitbench-v2-1
```

## Final Commit

```text
b4bd5661544079db5a9c6d274b24bf05787891be
```

## Final Status

```text
FAN_FOUNDATION_FAIL
```

## Key Metrics

- Oracle Temporal FAN-5 mean AUPRC: `0.9138`
- Predicted Temporal FAN-5 Strict mean AUPRC: `0.3151`
- Predicted Temporal FAN-4 Strict mean AUPRC: `0.1572`
- Concept leakage residual AUPRC: `0.1569`
- Planted CircuitF1: `1.0000`

## Stages Completed

- Temporal concept FAN implementation.
- Concept sufficiency audit.
- Concept leakage diagnostics.
- Membership and FAN weight diagnostics.
- Faithfulness diagnostics.
- Planted control.
- Standard Transformer SCTC diagnostic.
- Three-seed aggregate.

## Stages Skipped

- FAN + SCTC is skipped when FAN Foundation Gate fails.

## Next Scientific Step

If FAN Foundation Gate fails, inspect temporal concept leakage, membership saturation, and alpha-mu evidence compression before running FAN+SCTC.
