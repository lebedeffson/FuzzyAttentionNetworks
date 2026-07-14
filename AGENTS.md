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
