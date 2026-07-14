# Current Project State

Last updated: 2026-07-14

## Active Branch

```text
fix/med-circuitbench-v2-2-final
```

## Current Result

```text
V3_REAL_MIXED_RESULT_REPLICATION_CONFIRMED
```

The previous `Med_CircuitBench_V3_REAL_RESEARCH_FINAL.zip` is only an
intermediate package. The current finalization output is:

```text
artifacts/medical/v3_real_final
```

## Final Command

```bash
.venv/bin/python scripts/medical/v3/finalize_real_research.py \
  --config configs/medical/v3/full.yaml \
  --source-output artifacts/medical/v3_real \
  --output artifacts/medical/v3_real_final \
  --seeds 42 43 44
```

## Final Scientific Status

```text
FAN: FAN_VALIDATED
Planted: PLANTED_VALIDATED_NEGATIVE
Standard SCTC fidelity: PASS
Standard graph recovery: NO_VALIDATED_EDGES
FAN+SCTC: SKIPPED_BY_GATE
Original held-out: PARTIAL_TEST_CONSUMED
Frozen replication: COMPLETED_WITH_FROZEN_MODELS
```

## Final Delivery

```text
ZIP pattern: artifacts/medical/Med_CircuitBench_V3_REAL_FINAL_<date>_<commit>.zip
ZIP size: recorded in external .sha256 and final report
SHA256: recorded in external .sha256 and final report
Delivery validation: PASS
Paper claims validation: PASS
Anti-synthetic validation: PASS
Provenance validation: PASS
```

## Key Results

FAN validation:

```text
Predicted FAN-NoAlpha validation AUPRC by seed:
42: 0.827707
43: 0.797834
44: 0.840722

Direct macro R2 by seed:
42: 0.646900
43: 0.640321
44: 0.648308

Macro Pearson by seed:
42: 0.792612
43: 0.787773
44: 0.793594
```

Planted neural circuit:

```text
Node precision: 1.0 / 1.0 / 1.0
Node recall: 1.0 / 1.0 / 1.0
CircuitF1: 1.0 / 1.0 / 0.9
Gate status: PLANTED_VALIDATED_NEGATIVE
Reason: DEAD_FEATURE_FRACTION_GATE_FAILED
Minimum dead-feature fraction: 0.742188
```

Standard Transformer + SCTC:

```text
Fidelity status: PASS
Mean delta AUPRC: 0.000297
Validated candidate edges: 0
DataGraphAgreementF1: 0.0
Status: NO_VALIDATED_EDGES
```

Partial held-out test:

```text
Status: PARTIAL_TEST_CONSUMED
Standard Transformer test AUPRC:
42: 0.819045
43: 0.822749
44: 0.819370
```

Frozen replication:

```text
Replication generator seed: 20260715
Episodes: 10000
FAN pattern: CONFIRMED
Standard SCTC fidelity pattern: CONFIRMED
Planted recovery pattern: CONFIRMED
Planted dictionary gate: NEGATIVE_PATTERN_CONFIRMED
Selected-edge replication: NO_VALIDATED_EDGES
```

## Interpretation

Multi-set fuzzy concept encoding with a signed additive decision layer remains
validated. Standard SCTC preserves model predictions with low fidelity error.
The planted control recovers nodes and most edges, but the preregistered
dictionary-utilization gate remains negative. Standard Transformer edge
discovery did not validate data-graph edges under the frozen protocol.

## Frozen Prior Results

- V1: `ddcf32ae357dfedd91444c5cba18fac1af5d525e`, status `V1_NO_GO`.
- V2 foundation: `b4bd5661544079db5a9c6d274b24bf05787891be`, status `FAN_FOUNDATION_FAIL`.
- V2.1: `d404aabf8d93844e11878263103bbdb5d2ab16bc`, status `FAN_FOUNDATION_FAIL`.
- Revoked V3 engineering snapshot: `56a160aaa7faefaad4cbaeceff99a23335709f40`, previously reported as `V3_GO`.

## Next Scientific Step

Do not tune this package after partial held-out and replication reporting. Any
next experiment should be explicitly versioned and should focus on improving
SCTC dictionary utilization and intervention-edge discovery without reusing the
consumed test for model selection.
