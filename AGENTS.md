# Project Instructions

## Repository

This repository contains historical fuzzy-attention implementations, the current concept-mediated FAN architecture, and Med-CircuitBench mechanistic interpretability experiments.

## Canonical FAN Architecture

The current FAN architecture is concept-mediated:

```text
temporal encoder
-> latent representation
-> concept projection
-> fuzzy memberships
-> FAN concept weights
-> concept contributions
-> prediction
```

Do not reinterpret FAN as a replacement for `torch.nn.MultiheadAttention`.

## Frozen Experiments

- Med-CircuitBench V1: `ddcf32ae357dfedd91444c5cba18fac1af5d525e`, status `V1_NO_GO`.
- Med-CircuitBench V2 foundation: `b4bd5661544079db5a9c6d274b24bf05787891be`, status `FAN_FOUNDATION_FAIL`.
- Med-CircuitBench V2.1: `d404aabf8d93844e11878263103bbdb5d2ab16bc`, status `FAN_FOUNDATION_FAIL`.

Do not overwrite or modify frozen results.

## Active Experiment

Active branch:

```text
fix/med-circuitbench-v2-2-final
```

Current FAN validation command:

```bash
python scripts/medical/v3/run_fan_iteration.py \
  --config configs/medical/v3/full.yaml \
  --seeds 42 43 44 \
  --output artifacts/medical/v3_fan_iterations
```

Current exact faithfulness diagnostic command:

```bash
python scripts/medical/v3/diagnose_fan_faithfulness.py \
  --config configs/medical/v3/full.yaml \
  --iteration-dir artifacts/medical/v3_fan_iterations/runs/iteration_01 \
  --seeds 42 43 44 \
  --output artifacts/medical/v3_fan_iterations/runs/iteration_01/exact_faithfulness_diagnostics
```

Current Oracle alpha-ablation command:

```bash
python scripts/medical/v3/run_oracle_alpha_ablation.py \
  --config configs/medical/v3/full.yaml \
  --seeds 42 43 44 \
  --output artifacts/medical/v3_alpha_ablation
```

Current Predicted FAN strict validation command:

```bash
python scripts/medical/v3/run_predicted_fan_strict.py \
  --config configs/medical/v3/full.yaml \
  --seeds 42 43 44 \
  --output artifacts/medical/v3_predicted_fan_strict
```

Current phase:

```text
V3.1 method-improvement exploration
```

The earlier `V3_GO` package is revoked as a surrogate-heavy engineering
snapshot. The `d8906252...` package is also only an intermediate engineering
snapshot because the finalizer still contained synthetic/placeholder result
construction. The corrected real-practice package is under
`artifacts/medical/v3_real_final` and keeps mixed results instead of forcing a
positive FAN/SCTC interpretation.

The held-out test was opened once after validation freeze in the V3 real
research runner. Do not re-open test or change selected FAN/SCTC configurations
unless a new explicitly versioned experiment is started.

V3.1 is an explicitly versioned method-improvement experiment. It must not
rewrite V3 final results or use the consumed V3 held-out test for selection.
Primary V3.1 work targets FAN contribution stability, adaptive SCTC dictionary
utilization, temporal/active-window intervention search, and concept-aligned
model arms.

Current V3.1 planted adaptive SCTC smoke command:

```bash
.venv/bin/python scripts/medical/v3_1/run_planted_adaptive_sctc.py \
  --config configs/medical/v3/full.yaml \
  --method-config configs/medical/v3_1/method_improvements.yaml \
  --seeds 42 \
  --output artifacts/medical/v3_1_smoke/planted_adaptive_sctc \
  --max-epochs 1 \
  --limit-layers 1 \
  --limit-candidates 1
```

Smoke output is only a wiring check. It must report `SMOKE_PASS` or
`SMOKE_FAIL` with `scientific_gate_evaluated=false`; do not report
`PLANTED_ADAPTIVE_PASS` for limited seeds, limited layers/candidates, one-epoch
runs, or warm-up top-k runs.

Current V3.1 planted adaptive SCTC full Stage A command:

```bash
.venv/bin/python scripts/medical/v3_1/run_planted_adaptive_sctc.py \
  --config configs/medical/v3/full.yaml \
  --method-config configs/medical/v3_1/method_improvements.yaml \
  --seeds 42 43 44 \
  --output artifacts/medical/v3_1/planted_adaptive_sctc \
  --full \
  --run-interventions \
  --run-negative-controls
```

The first full Stage A run is a validated negative:
`PLANTED_ADAPTIVE_NEGATIVE`. Fidelity and dictionary utilization improve, but
strict node/edge recovery and negative-control gates do not pass.

Current V3.1 joint causal SCTC command:

```bash
.venv/bin/python scripts/medical/v3_1/run_joint_causal_sctc.py \
  --config configs/medical/v3/full.yaml \
  --method-config configs/medical/v3_1/method_improvements.yaml \
  --seeds 42 43 44 \
  --baseline artifacts/medical/v3_1/planted_adaptive_sctc \
  --output artifacts/medical/v3_1/joint_causal_sctc \
  --continue-until-terminal
```

The first full B0-B3 run is also negative:
`UNSUPERVISED_CAUSAL_BASIS_NOT_IDENTIFIABLE`. Fidelity remains high, but
compact whitening, incoherence, sparse transitions, and interventional
consistency still do not recover the planted causal basis under strict gates.

Current V3.1 concept-aligned interventional SCTC command:

```bash
.venv/bin/python scripts/medical/v3_1/run_concept_aligned_interventional_sctc.py \
  --config configs/medical/v3/full.yaml \
  --method-config configs/medical/v3_1/method_improvements.yaml \
  --seeds 42 43 44 \
  --baseline-adaptive artifacts/medical/v3_1/planted_adaptive_sctc \
  --baseline-joint artifacts/medical/v3_1/joint_causal_sctc \
  --output artifacts/medical/v3_1/concept_aligned_interventional_sctc \
  --continue-until-terminal
```

The first full weakly supervised run is also negative:
`CAUSAL_BASIS_NOT_RECOVERED_WITH_WEAK_ALIGNMENT`. Fidelity remains strong and
dictionary utilization is controlled, but correct concept alignment does not
separate from permuted/random concept controls and no strict node/edge recovery
is accepted.

Current V3.1 concept-aligned negative-result diagnostic command:

```bash
.venv/bin/python scripts/medical/v3_1/audit_concept_aligned_negative_result.py \
  --config configs/medical/v3/full.yaml \
  --run-dir artifacts/medical/v3_1/concept_aligned_interventional_sctc \
  --output artifacts/medical/v3_1/concept_aligned_interventional_sctc/diagnostic_closure \
  --seeds 42 43 44
```

The diagnostic closure currently reports `EVALUATION_PROTOCOL_INVALID` because
oracle planted directions do not pass the strict edge evaluator. Treat the
concept-aligned mechanistic negative result as provisional until the edge/null
protocol is repaired and the oracle sanity test passes.

Current V3.1 repaired planted causal evaluator command:

```bash
.venv/bin/python scripts/medical/v3_1/repair_planted_causal_evaluator.py \
  --config configs/medical/v3/full.yaml \
  --seeds 42 43 44 \
  --output artifacts/medical/v3_1/repaired_causal_evaluator
```

The repaired evaluator separates direct effects from total reachability and
uses paired bootstrap direct/non-edge controls instead of random directions in
the low-rank planted activation space. Oracle sanity now passes:
`ORACLE_CAUSAL_EVALUATOR_PASS`.

Current V3.1 frozen concept-aligned re-evaluation command:

```bash
.venv/bin/python scripts/medical/v3_1/reevaluate_concept_aligned_with_repaired_evaluator.py \
  --config configs/medical/v3/full.yaml \
  --run-dir artifacts/medical/v3_1/concept_aligned_interventional_sctc \
  --output artifacts/medical/v3_1/concept_aligned_interventional_sctc/repaired_recovery
```

Using the repaired evaluator, existing concept-aligned checkpoints still do not
support a positive mechanistic claim because controls match or exceed the
correct-concepts arm. Status:
`CAUSAL_BASIS_NOT_RECOVERED_WITH_WEAK_ALIGNMENT`.

Current V3.1 decoder-coupled concept SCTC command:

```bash
.venv/bin/python scripts/medical/v3_1/run_decoder_coupled_concept_sctc.py \
  --config configs/medical/v3/full.yaml \
  --method-config configs/medical/v3_1/decoder_coupled_concept_sctc.yaml \
  --evaluator artifacts/medical/v3_1/repaired_causal_evaluator \
  --baseline artifacts/medical/v3_1/concept_aligned_interventional_sctc \
  --seeds 42 43 44 \
  --output artifacts/medical/v3_1/decoder_coupled_concept_sctc \
  --full
```

The single allowed C2 decoder-coupled run is complete. Status:
`DECODER_COUPLED_FIDELITY_FAIL`. Technical coupling passes and the repaired
oracle evaluator remains frozen/pass, but behavioral fidelity fails on two of
three correct-concepts seeds and controls match the correct arm on recovered
direct/total graph scores. Do not add another SCTC lambda/grid after C2; treat
the C2 arm as closed unless the user explicitly starts a new versioned
experiment.

Current V3.1 FAN-NoAlpha-Stable command:

```bash
.venv/bin/python scripts/medical/v3_1/run_fan_noalpha_stable.py \
  --config configs/medical/v3/full.yaml \
  --method-config configs/medical/v3_1/fan_noalpha_stable.yaml \
  --seeds 42 43 44 \
  --output artifacts/medical/v3_1/fan_noalpha_stable
```

The single allowed FAN stability run is complete. Status:
`FAN_NOALPHA_BASELINE_RETAINED`. Stable training preserved AUPRC and
faithfulness on all three seeds, but cross-seed contribution Spearman remained
below gate. Production FAN remains the original `FAN-NoAlpha`; contribution
instability is a documented limitation.

Old Q1 oracle-concept surrogate ablation command:

```bash
.venv/bin/python scripts/medical/v3_1/run_q1_oracle_concept_surrogate_ablation.py \
  --config configs/medical/v3/full.yaml \
  --output artifacts/medical/q1_empirical_extension \
  --runs 30 \
  --n-samples 6000 \
  --package \
  --zip-output-dir artifacts/medical
```

This contour is not a scientific Q1 gate. It is retained only as an
`ORACLE_CONCEPT_SURROGATE_ABLATION` because it uses generator states as
features and shallow surrogate estimators.

Q1 neural pilot command:

```bash
.venv/bin/python scripts/medical/v3_1/run_q1_neural_empirical_extension.py \
  --config configs/medical/v3/full.yaml \
  --output artifacts/medical/q1_neural_empirical_extension \
  --runs 30 \
  --n-samples 6000 \
  --epochs 2 \
  --package \
  --zip-output-dir artifacts/medical
```

The `9715e4b` neural contour is a failed pilot, not a completed Q1 result.
Status: `Q1_NEURAL_FAILED_PILOT`. It must not be reported as
`Q1_EMPIRICAL_EXTENSION_NEURAL_VALIDATED`.

Current canonical 3-seed parity command:

```bash
.venv/bin/python scripts/medical/v3_1/verify_canonical_3_seed_parity.py \
  --output artifacts/medical/canonical_3_seed_parity \
  --package \
  --zip-output-dir artifacts/medical
```

The next Q1 neural grid is forbidden until this gate reports
`CANONICAL_3_SEED_PARITY_PASS`. The gate checks the canonical V3 configuration
(`d=128`, 4 transformer layers, 4 heads, FFN=512, up to 50 epochs), three seeds
42/43/44, existing canonical ConceptFAN-NoAlpha and plain Transformer
checkpoints, and AUPRC parity near the established 0.82 level.

Current Q1 neural final command:

```bash
.venv/bin/python scripts/medical/v3_1/run_q1_neural_final.py \
  --config configs/medical/v3/full.yaml \
  --output artifacts/medical/q1_neural_final \
  --package \
  --zip-output-dir artifacts/medical
```

The final runner must keep the canonical architecture and training protocol,
run the model-arm sanity gate before the 30-run grid, produce 150 canonical
checkpoints, and may report `Q1_EMPIRICAL_EXTENSION_NEURAL_VALIDATED` only for
the complete 30-run grid with sanity enabled. The release must then pass the
separate read-only verifier:

```bash
.venv/bin/python scripts/medical/v3_1/verify_q1_neural_readonly.py \
  --release artifacts/medical/q1_neural_final \
  --config configs/medical/v3/full.yaml \
  --output-json artifacts/medical/q1_neural_final/MANIFESTS/q1_neural_readonly_validation.json
```

Allowed V3 final statuses:

- `V3_REAL_VALIDATED_NEGATIVE`
- `V3_REAL_FAN_VALIDATED_SCTC_NEGATIVE`
- `V3_REAL_MIXED_RESULT_REPLICATION_CONFIRMED`
- `V3_REAL_MIXED_RESULT_REPLICATION_NOT_CONFIRMED`

## Scientific Rules

- Do not tune thresholds after viewing validation results.
- Do not open test before validation is frozen.
- Do not call DataGraphAgreementF1 a model CircuitF1.
- CircuitF1 is reserved for the planted internal circuit.
- Do not use constant push as a null for feature ablation.
- Do not use fewer than 2000 episodes for SCTC unless explicitly justified.
- Negative results must be preserved.
- Do not put PhysioNet data in Git or delivery archives.
- Documentation-only increments are not accepted as scientific work.
- Smoke output must not be packaged as final output.

## Required Commands

Run tests:

```bash
python -m pytest tests/medical/v3 tests/medical/v2_2 tests/medical/v2 -q
```

Run V3:

```bash
.venv/bin/python scripts/medical/v3/finalize_real_research.py \
  --config configs/medical/v3/full.yaml \
  --source-output artifacts/medical/v3_real \
  --output artifacts/medical/v3_real_final \
  --seeds 42 43 44
```

## MIMIC-AKI Demo Engineering Closure

Canonical branch:

```text
experiment/mimic-aki-fan-sae
```

Canonical MIMIC-AKI paths:

```text
first FAN canonical path: src/fan/attention/
ConceptFAN canonical path: src/fan/concept/temporal.py
SAE canonical path: src/fan/sae/
MIMIC source path: src/mimic_aki/
```

Demo command:

```bash
.venv/bin/python scripts/mimic_aki/run_demo_engineering.py \
  --config configs/mimic_aki/program_demo.yaml \
  --output artifacts/mimic_aki/final_demo
```

MIMIC-IV Demo may be used only for engineering validation of ingestion,
model wiring, SAE mechanics, steering mechanics, bundles, CLI wrappers, and
reports. It must not be described as clinical or scientific performance
evidence. Full MIMIC-IV remains an external access requirement for scientific
validation.

Terminal demo practice status:

```text
PRACTICE_CLOSED_MIMIC_DEMO_END_TO_END
```

Full-data scientific status remains:

```text
BLOCKED_FULL_MIMIC_ACCESS
```

## Dual Benchmark Release

Final dual-benchmark status:

```text
SCIENTIFIC_RESEARCH_COMPLETE
ENGINEERING_DEMO_COMPLETE
STANDALONE_RELEASE_VALIDATED
FULL_CLINICAL_VALIDATION_BLOCKED
```

Med-CircuitBench is the scientific validation block. MIMIC-IV Demo is only the
engineering compatibility block. Do not report MIMIC-IV Demo metrics as
clinical or scientific performance evidence. Do not use an artificial
`AUPRC >= 0.85` gate for the final practice status. Do not run additional
hyperparameter experiments after the dual release unless the user explicitly
starts a new versioned project.

Standalone release command:

```bash
.venv/bin/python scripts/release/build_dual_benchmark_release.py \
  --output artifacts/release \
  --rebuild-nested
```

The release builder must rebuild nested Med-CircuitBench and MIMIC Demo
archives from current source. Do not aggregate stale nested ZIPs.
