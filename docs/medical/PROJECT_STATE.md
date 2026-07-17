# Current Project State

Last updated: 2026-07-15

## Branch

```text
experiment/mimic-aki-fan-sae
```

## Final Practical Status

```text
SCIENTIFIC_RESEARCH_COMPLETE
ENGINEERING_DEMO_COMPLETE
STANDALONE_RELEASE_VALIDATED
FULL_CLINICAL_VALIDATION_BLOCKED
```

## Dataset Roles

```text
Med-CircuitBench:
scientific validation and controlled mechanistic tests

MIMIC-IV Demo:
engineering compatibility only

Full MIMIC-IV:
external data requirement for clinical validation
```

MIMIC-IV Demo results must not be described as clinical or scientific
performance evidence. The demo validates ingestion, cohort/window construction,
model wiring, structural ConceptFAN checks, bundle/report generation, SAE
mechanics, and steering mechanics.

## Canonical Paths

```text
first FAN canonical path: src/fan/attention/
ConceptFAN canonical path: src/fan/concept/temporal.py
SAE canonical path: src/fan/sae/
MIMIC-AKI source path: src/mimic_aki/
Med-CircuitBench source path: src/med_circuitbench/
```

## Med-CircuitBench Result

Production FAN remains:

```text
FAN-NoAlpha
```

The single FAN-NoAlpha-Stable run preserved predictive quality but did not pass
the contribution-stability gate:

```text
FAN_NOALPHA_BASELINE_RETAINED
```

The SCTC line is closed:

```text
SPARSE_FIDELITY_VALIDATED
MECHANISTIC_RECOVERY_NOT_SEMANTICALLY_SPECIFIC
DECODER_COUPLING_BREAKS_FIDELITY
```

The repaired planted causal evaluator passes oracle sanity:

```text
ORACLE_CAUSAL_EVALUATOR_PASS
```

## MIMIC-IV Demo Result

Canonical demo command:

```bash
.venv/bin/python scripts/mimic_aki/run_demo_engineering.py \
  --config configs/mimic_aki/program_demo.yaml \
  --output artifacts/mimic_aki/final_demo
```

The demo package uses a pseudo-temporal scaled aggregate vector to exercise
temporal model code paths. It does not validate real ICU hourly dynamics.

Concept targets unavailable in the demo are masked:

```text
renal_function_trajectory: available from creatinine
oliguria_burden: masked unavailable
hemodynamic_instability: masked unavailable unless vitals/lactate are present
volume_imbalance: masked unavailable unless fluid/urine are present
systemic_stress: masked unavailable unless vitals are present
```

Faithfulness status on MIMIC Demo:

```text
STRUCTURAL_FAITHFULNESS_MECHANICS_PASS_EMPIRICAL_NOT_EVALUATED
```

SAE/steering status on MIMIC Demo:

```text
SAE and steering mechanics: completed
SAE fidelity: reported separately and may fail on demo
steering: mechanics only, not a semantic intervention claim
```

## Final Release Command

```bash
.venv/bin/python scripts/release/build_dual_benchmark_release.py \
  --output artifacts/release \
  --rebuild-nested
```

The release builder rebuilds both nested archives from current source before
creating the top-level standalone archive.

## Restrictions

- Do not include raw MIMIC-IV data, protected rows, or patient identifiers in
  release archives.
- Do not use `AUPRC >= 0.85` as an artificial final gate.
- Do not run new FAN/SCTC/SAE hyperparameter experiments for this release.
- Do not claim MIMIC-IV clinical validation until full MIMIC-IV is available
  and the frozen full-data protocol is executed.
