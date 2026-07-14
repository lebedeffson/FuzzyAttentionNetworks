# Current Project State

Last updated: 2026-07-14

## Active Branch

```text
experiment/med-circuitbench-v2
```

## Latest Result Commit

```text
b4bd5661544079db5a9c6d274b24bf05787891be
```

## Current Repository Commit

Recorded by `git rev-parse HEAD` and by `GIT_INFO.txt` in the delivery archive.

## Current Result

```text
FAN_FOUNDATION_FAIL
```

## Completed

- Canonical concept-mediated FAN implemented.
- Oracle FAN and Predicted FAN executed.
- Clean shortcut checks completed.
- Clinical-confounded shortcut checks completed.
- Planted internal circuit passed.
- Three seeds completed.
- Delivery validation passed.

## Key Metrics

- Oracle FAN-5 mean AUPRC: `0.5994`
- Predicted FAN-5 mean AUPRC: `0.8236`
- Predicted FAN-5 macro concept R2: `0.3303`
- Predicted FAN-5 mean concept Pearson: `0.1098`
- Predicted FAN-4 mean AUPRC: `0.8382`
- Planted CircuitF1: `1.0`
- Planted node precision: `1.0`
- Planted node recall: `1.0`
- Planted sign agreement: `1.0`
- Negative-control FPR: `0.0`

## Current Interpretation

The planted mechanistic pipeline works.

The concept-mediated FAN foundation gate failed because:

- Oracle FAN performs worse than Predicted FAN.
- Predicted concept quality is weak.
- Insertion faithfulness does not exceed the random control.
- Explicit concept contributions are not sufficiently stable.

## Next Scientific Task

Investigate why Oracle FAN underperforms Predicted FAN before running Transformer+SCTC or FAN+SCTC.

Priority checks:

1. Verify alignment between `c_target` at hour 35 and the future target at hours 36-41.
2. Test whether `S_35` alone dominates the target.
3. Compare linear and MLP oracle classifiers on true concepts.
4. Verify removal and insertion implementation.
5. Inspect membership saturation.
6. Inspect concept contribution distributions.
7. Test whether `alpha * mu` loses sign or magnitude information.
8. Compare decision heads over `c`, `mu`, `alpha * mu`, and concatenated evidence.

## Blocked Stages

- Representation audit: `SKIPPED_BY_GATE`
- Standard Transformer + SCTC: `SKIPPED_BY_GATE`
- FAN + SCTC: `SKIPPED_BY_GATE`

## Data Restrictions

PhysioNet is outside V2.
No patient data may be committed or packaged.

