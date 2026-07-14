# Project Instructions

## Repository

This repository contains the historical fuzzy-attention implementations, the current concept-mediated FAN architecture, and the Med-CircuitBench mechanistic interpretability experiments.

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

Med-CircuitBench V1 is frozen at:

```text
ddcf32ae357dfedd91444c5cba18fac1af5d525e
```

Status:

```text
V1_NO_GO
```

Do not overwrite or modify V1 results.

## Active Experiment

Active branch:

```text
experiment/med-circuitbench-v2
```

V2 must use the single-program gated workflow.

Allowed final statuses:

- `V2_GO`
- `FAN_FOUNDATION_FAIL`
- `PLANTED_CONTROL_FAIL`
- `SCTC_RECOVERY_FAIL`
- `CLEAN_ONLY_SUCCESS`
- `V2_NO_GO`

## Scientific Rules

- Do not tune thresholds after viewing validation results.
- Do not open test before validation is frozen.
- Do not call DataGraphAgreementF1 a model CircuitF1.
- CircuitF1 is reserved for the planted internal circuit.
- Do not use constant push as a null for feature ablation.
- Do not use fewer than 2000 episodes for SCTC unless explicitly justified.
- Negative results must be preserved.
- Do not put PhysioNet data in Git or delivery archives.

## Required Commands

Run tests:

```bash
python -m pytest tests/medical tests/medical/v2 -q
```

Run V2:

```bash
python scripts/medical/v2/run_v2_program.py \
  --config configs/medical/v2/program.yaml \
  --seeds 42 43 44 \
  --mode full \
  --output artifacts/medical/v2
```

## Delivery

The final delivery must be:

```text
Med_CircuitBench_V2_FINAL_<date>_<commit>.zip
```

It must contain code, configs, logs, actual results, gate decisions, tables, figures, manifests, and SHA-256 checksums.

Do not submit skeletons, pilot-only packages, or documentation-only increments.

