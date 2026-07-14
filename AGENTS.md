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

Current executable program:

```bash
python scripts/medical/v3/run_research_program.py \
  --config configs/medical/v3/full.yaml \
  --seeds 42 43 44 \
  --output artifacts/medical/v3 \
  --continue-until-terminal
```

V3 must use the terminal-result workflow and must continue independent stages
after a negative FAN gate.

Allowed V3 final statuses:

- `V3_GO`
- `V3_FAN_VALIDATED_SCTC_NEGATIVE`
- `V3_FAN_NEGATIVE_SCTC_VALIDATED`
- `V3_VALIDATED_NEGATIVE`
- `V3_MIXED_RESULT`

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
python scripts/medical/v3/run_research_program.py \
  --config configs/medical/v3/full.yaml \
  --seeds 42 43 44 \
  --output artifacts/medical/v3 \
  --continue-until-terminal
```
