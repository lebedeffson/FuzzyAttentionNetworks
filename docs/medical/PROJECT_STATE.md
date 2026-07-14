# Current Project State

Last updated: 2026-07-14

## Active Branch

```text
experiment/med-circuitbench-v2-1
```

## Latest Result Commit

```text
2951dd5a7d2a51e680812ed79749b7bd5116eccb
```

## Current Repository Commit

Recorded by `git rev-parse HEAD` and by `GIT_INFO.txt` in the delivery archive.

## Final Status

```text
FAN_FOUNDATION_FAIL
```

## Key Metrics

- Oracle Temporal FAN-5 mean AUPRC: `0.6456`
- Predicted Temporal FAN-5 Strict mean AUPRC: `0.6677`
- Predicted Temporal FAN-4 Strict mean AUPRC: `0.3552`
- Concept leakage residual AUPRC: `0.3844`
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
