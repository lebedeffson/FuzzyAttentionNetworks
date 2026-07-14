# Current Project State

Last updated: 2026-07-14

## Active Branch

```text
fix/med-circuitbench-v2-2-final
```

## Current Program

```bash
python scripts/medical/v3/run_research_program.py \
  --config configs/medical/v3/full.yaml \
  --seeds 42 43 44 \
  --output artifacts/medical/v3 \
  --continue-until-terminal
```

## Latest Terminal Result

```text
V3_FAN_NEGATIVE_SCTC_VALIDATED
```

This result used the corrected neural `TemporalConceptFANModel` path, not the
earlier Ridge/LogisticRegression surrogate FAN evaluation.

## Delivery

```text
artifacts/medical/Med_CircuitBench_V3_FINAL_20260714_08576b4.zip
```

SHA256:

```text
d4214dea585e446cef8f4167d71ab451c3dd747c643f16c28a6c71538329b458
```

ZIP size:

```text
866M
```

## Frozen Prior Results

- V1: `ddcf32ae357dfedd91444c5cba18fac1af5d525e`, status `V1_NO_GO`.
- V2 foundation: `b4bd5661544079db5a9c6d274b24bf05787891be`, status `FAN_FOUNDATION_FAIL`.
- V2.1: `d404aabf8d93844e11878263103bbdb5d2ab16bc`, status `FAN_FOUNDATION_FAIL`.
- V2.2 corrected runner commit: `a142bd7cef7f6eac7e5b24c32bdb21135e442207`.

## Key V3 Metrics

- Primary FAN family selected by validation: `gaussian`.
- Oracle Temporal FAN mean AUPRC: `0.3658`.
- Predicted Temporal FAN Strict mean AUPRC: `0.6013`.
- Predicted Temporal FAN trajectory R2: `0.5303`.
- Predicted Temporal FAN trajectory Pearson: `0.7019`.
- FAN gate pass count: `0 / 3`.
- Planted neural CircuitF1: `1.0000`.
- Planted gate pass count: `3 / 3`.
- Standard SCTC delta AUPRC: `0.00096`.
- Standard SCTC probability MAE: `0.00065`.
- SCTC grid rows: `9`.
- Test opened: `false`.

## Completed V3 Stages

- Corrected FAN execution through `TemporalConceptFANModel`.
- Seed control for V3 program.
- Full three-seed corrected V2.2 core execution.
- FAN family comparison.
- Leakage bootstrap.
- Faithfulness outputs from frozen FAN decision head.
- Planted neural circuit and SCTC grid.
- Standard Transformer + SCTC fidelity.
- Paper source and PDF generation.
- Paper claims validation.
- Delivery validation.
- Final ZIP packaging with source snapshot.

## Interpretation

The FAN gate remains negative after using the real neural FAN path, while the
planted neural control and Standard SCTC fidelity pass. The correct terminal
interpretation is therefore `V3_FAN_NEGATIVE_SCTC_VALIDATED`, not a positive FAN
result and not an implementation-only failure.

## Next Scientific Step

Use the packaged V3 results and paper draft for article-level interpretation.
Do not retune thresholds after this terminal validation result.
