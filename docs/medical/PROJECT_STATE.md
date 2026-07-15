# Current Project State

Last updated: 2026-07-15

## Active Branch

```text
experiment/mimic-aki-fan-sae
```

## Current Program

```text
Med-CircuitBench V3.1: FROZEN_RESEARCH_COMPLETE
New program: MIMIC_AKI_FAN_SAE
```

MIMIC-AKI is a new real-data research program. It must not rewrite V3.1
results. MIMIC-IV data must not be committed or packaged. If `MIMIC_IV_ROOT`
is absent or incomplete, the only valid runtime status is:

```text
BLOCKED_DATA_ACCESS
```

The codebase must still provide parsers, configs, synthetic-fixture tests,
model code, SAE primitives, CLI scripts, and a gated research runner before
reporting the access block.

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
FAN predictive path: VALIDATED
FAN full gate: FAN_VALIDATED_NEGATIVE
Planted: PLANTED_VALIDATED_NEGATIVE
Standard SCTC fidelity: PASS
Standard graph recovery: LIMITED_VALIDATED_EDGES
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
42: 0.646562
43: 0.640029
44: 0.647805

Macro Pearson by seed:
42: 0.792612
43: 0.787773
44: 0.793594

FAN gate reason:
cross-seed signed-contribution stability failed on the common reference set.
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
Validated candidate edges: limited
DataGraphAgreementF1: 0.333333
Status: VALIDATED_EDGES_FOUND
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
FAN predictive pattern: CONFIRMED
Standard SCTC fidelity pattern: CONFIRMED
Planted recovery pattern: CONFIRMED
Planted dictionary gate: NEGATIVE_PATTERN_CONFIRMED
Selected-edge replication: EVALUATED_SELECTED_EDGES
```

## Interpretation

Multi-set fuzzy concept encoding with a signed additive decision layer preserves
predictive performance, but the full FAN gate remains negative because
cross-seed signed-contribution stability fails. Standard SCTC preserves model
predictions with low fidelity error. The planted control recovers nodes and most
edges, but the preregistered dictionary-utilization gate remains negative.
Standard Transformer edge discovery finds limited validated intervention edges.

## Frozen Prior Results

- V1: `ddcf32ae357dfedd91444c5cba18fac1af5d525e`, status `V1_NO_GO`.
- V2 foundation: `b4bd5661544079db5a9c6d274b24bf05787891be`, status `FAN_FOUNDATION_FAIL`.
- V2.1: `d404aabf8d93844e11878263103bbdb5d2ab16bc`, status `FAN_FOUNDATION_FAIL`.
- Revoked V3 engineering snapshot: `56a160aaa7faefaad4cbaeceff99a23335709f40`, previously reported as `V3_GO`.

## Next Scientific Step

Do not tune the V3 final package after partial held-out and replication
reporting.

The next experiment is explicitly versioned as V3.1 and must not reuse the
consumed V3 held-out test for selection.

V3.1 implementation started:

```text
configs/medical/v3_1/method_improvements.yaml
src/fan/sctc/adaptive.py
src/fan/concept/stability.py
scripts/medical/v3_1/run_planted_adaptive_sctc.py
tests/medical/v3/test_v3_1_method_improvements.py
```

Initial implemented methods:

```text
FAN contribution consistency loss
FAN fuzzy-basis stability penalties
Adaptive SCTC train-activation normalization
Adaptive SCTC reconstruction bias
Top-k annealing
Dead-feature residual resampling
Layer-specific capacity by effective rank
Planted adaptive SCTC runner
```

Smoke check:

```text
seed: 42
layers: first layer only
candidates: first candidate only
epochs: 1
status: SMOKE_PASS
scientific_gate_evaluated: false
reason: single seed, first layer only, one epoch, warm-up top-k
```

This smoke output is a wiring check only, not a scientific V3.1 result.
The planted adaptive scientific gate requires full layers, three seeds, final
top-k metrics, intervention validation, edge recovery, and negative controls.

Full V3.1 Stage A planted adaptive SCTC was run after smoke semantics were
fixed:

```text
output: artifacts/medical/v3_1/planted_adaptive_sctc
status: PLANTED_ADAPTIVE_NEGATIVE
scientific_gate_evaluated: true
seed_pass_count: 0
```

Key Stage A result:

```text
Fidelity gate: PASS for all evaluated candidates
Best final dead-feature fraction: 0.0
Sparsity gate: PASS for selected candidates
Node/edge recovery gate: FAIL
Negative-control FPR: high for non-edge controls
Interpretation: adaptive SCTC improved utilization/fidelity but did not recover
the planted mechanism under strict Hungarian + intervention + negative-control
criteria.
```

V3.1 Joint Interventional Causal SCTC was added and run as a new method arm:

```text
output: artifacts/medical/v3_1/joint_causal_sctc
stages: B0 baseline, B1 compact incoherent, B2 joint transition, B3 interventional causal
status: UNSUPERVISED_CAUSAL_BASIS_NOT_IDENTIFIABLE
seed_pass_count: 0
best_stage: B2_joint_transition
best_config_id: tr0.01
best_CircuitF1: 0.0
```

Key interpretation:

```text
Fidelity remains strong across B1/B2/B3.
Whitening and compact rank-based capacities reduce usage but often make
aggregate L0 too low for the preregistered sparsity gate.
No B-stage recovers accepted planted nodes or edges under the strict
intervention/null criteria.
This supports the negative conclusion that unsupervised reconstruction plus
sparse transition/intervention objectives are still insufficient to identify
the planted causal basis in this setup.
```

V3.1 Concept-Aligned Interventional SCTC was added and run as the weakly
supervised mechanistic arm:

```text
output: artifacts/medical/v3_1/concept_aligned_interventional_sctc
controls: correct concepts, permuted concepts, Gaussian random targets
status: CAUSAL_BASIS_NOT_RECOVERED_WITH_WEAK_ALIGNMENT
reason: CONCEPT_CONTROLS_NOT_SEPARATED
seed_pass_count: 0
correct_mean_CircuitF1: 0.0
control_best_CircuitF1: 0.0
```

Key interpretation:

```text
Weak layer-specific concept alignment preserved reconstruction/prediction
fidelity, but it did not recover accepted planted nodes or edges under the
strict intervention/null protocol. Correct concept targets did not separate
from permuted or random targets, so the weakly supervised mechanistic claim is
not supported by this run.
```

Diagnostic closure was then added to test whether this negative result can be
interpreted as a valid mechanistic failure:

```text
output: artifacts/medical/v3_1/concept_aligned_interventional_sctc/diagnostic_closure
status: EVALUATION_PROTOCOL_INVALID
reason: ORACLE_PLANTED_DIRECTIONS_FAIL_EVALUATOR
oracle node precision/recall: 1.0 / 1.0
oracle edge precision/recall: 0.266667 / 0.8
oracle negative-control FPR: 0.733333
alignment gradient to encoder: PASS
correct concept objective vs controls: PASS
continuous control separation: PASS
replication freeze audit: PASS_WITH_FREEZE_RISK
```

Current interpretation:

```text
The concept-aligned arm is engineered and evaluated, and it preserves fidelity,
but the strong causal-basis negative claim is blocked. The oracle planted
directions fail the current edge/null evaluator because transitive/non-edge
effects are accepted as false edges. The next required work is evaluator/null
protocol repair, not another model or lambda search.
```

C0 repaired the planted causal evaluator:

```text
output: artifacts/medical/v3_1/repaired_causal_evaluator
status: ORACLE_CAUSAL_EVALUATOR_PASS
seed_pass_count: 3
ORACLE_DIRECT_EDGE_F1: 1.0
ORACLE_TOTAL_EFFECT_F1: 1.0
ORACLE_DIRECT_NEGATIVE_CONTROL_FPR: 0.0
ORACLE_TOTAL_NEGATIVE_CONTROL_FPR: 0.0
O_TO_S_ACCEPTED: true
NO_REVERSE_CAUSAL_EFFECTS: true
directional_null_status: NOT_APPLICABLE_LOW_RANK
```

The repaired evaluator uses Pearl-style `do` semantics for node interventions,
separates direct edge recovery from total-effect reachability, and avoids
full-dimensional random-direction nulls as the primary null for the low-rank
planted activations.

C1 then re-evaluated the existing Concept-Aligned Interventional SCTC
checkpoints without retraining:

```text
output: artifacts/medical/v3_1/concept_aligned_interventional_sctc/repaired_recovery
status: CAUSAL_BASIS_NOT_RECOVERED_WITH_WEAK_ALIGNMENT
model_selection_performed: false
correct_mean_direct_F1: 0.879630
control_best_direct_F1: 1.0
correct_mean_total_F1: 0.879630
control_best_total_F1: 1.0
```

Current interpretation after C0/C1:

```text
The evaluator defect is repaired for the planted oracle. Existing
concept-aligned checkpoints show strong repaired direct/total recovery in some
seeds, but controls match or exceed the correct-concepts arm. Therefore the
current weak-alignment configuration does not support a positive mechanistic
claim. A final decoder-coupled concept SCTC is allowed only as the single C2
methodological repair described in the project plan.
```

C2 Decoder-Coupled Concept SCTC was implemented and run as the single final
mechanistic repair:

```text
output: artifacts/medical/v3_1/decoder_coupled_concept_sctc
status: DECODER_COUPLED_FIDELITY_FAIL
scientific_gate_evaluated: true
technical_gate_pass: true
oracle_evaluator_status: ORACLE_CAUSAL_EVALUATOR_PASS
fidelity_gate_pass: false
recovery_gate_pass: true
specificity_gate_pass: false
correct_mean_direct_F1: 0.783069
control_best_direct_F1: 0.888889
```

Correct-concepts seed-level C2 result:

```text
seed 42: delta_AUPRC 0.000002, probability_MAE 0.000451, direct_F1 0.888889
seed 43: delta_AUPRC 0.000056, probability_MAE 0.036568, direct_F1 0.571429
seed 44: delta_AUPRC 0.015455, probability_MAE 0.202461, direct_F1 0.888889
```

Interpretation after C2:

```text
Decoder-coupled semantic supervision reaches the tied sparse dictionary and
frozen probes remain unchanged. However, the C2 arm fails the preregistered
fidelity gate on two of three seeds, and recovered graph scores are not
specific to correct concepts because permuted/random controls match the
correct arm. The SCTC method-improvement line is therefore closed as a
negative/mixed result unless a new explicitly versioned experiment is opened.
```

The final allowed FAN stability experiment was then run:

```text
output: artifacts/medical/v3_1/fan_noalpha_stable
status: FAN_NOALPHA_BASELINE_RETAINED
stable_seed_pass_count: 3
contribution_stability_pass: false
mean_pairwise_contribution_spearman: 0.066667
mean_pairwise_contribution_kendall_tau: 0.066667
mean_top3_jaccard: 0.300000
production_model: FAN-NoAlpha
sctc_development_status: TERMINAL_NEGATIVE_CLOSED
```

FAN-Stable seed-level result:

```text
seed 42: AUPRC 0.828779, baseline drop -0.001072, direct R2 0.646562, Pearson 0.792612
seed 43: AUPRC 0.797737, baseline drop 0.000097, direct R2 0.640029, Pearson 0.787773
seed 44: AUPRC 0.840699, baseline drop 0.000023, direct R2 0.647805, Pearson 0.793594
```

Final practical status:

```text
Med-CircuitBench V3.1 Research Complete
Production FAN: original FAN-NoAlpha
Adaptive SCTC: sparse behavioral fidelity/audit component
Repaired causal evaluator: benchmark tool
Joint/Concept-Aligned/Decoder-Coupled SCTC: archived experimental ablations
SCTC development: closed
Papers: not packaged here
```
