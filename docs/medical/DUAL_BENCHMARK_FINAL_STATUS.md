# Dual Benchmark Final Status

Final status:

```text
SCIENTIFIC_RESEARCH_COMPLETE
ENGINEERING_DEMO_COMPLETE
STANDALONE_RELEASE_VALIDATED
FULL_CLINICAL_VALIDATION_BLOCKED
```

## Roles

Med-CircuitBench is the scientific validation block. It contains the registered
FAN and SCTC results, oracle-calibrated causal evaluation, faithfulness
diagnostics, controls, replication evidence, and the closed SCTC specificity
line.

MIMIC-IV Demo is the engineering compatibility block. It verifies real-format
ingestion, cohort and incident AKI window construction, six model paths,
ConceptFAN structural invariants, fuzzy-attention wiring, SAE mechanics,
steering mechanics, bundles, CLI wrappers, reports, and release packaging.

Full MIMIC-IV is still required for clinical validation. Demo metrics must not
be used as clinical performance evidence.

## Final Registered Interpretation

```text
Med-CircuitBench:
scientific validation complete

MIMIC-IV Demo:
engineering compatibility complete

Standalone software release:
validated from rebuilt nested archives

Full MIMIC-IV:
blocked external data requirement
```

The release does not use an artificial `AUPRC >= 0.85` final gate. Stability is
reported as a scientific result, not tuned away by additional losses. New
hyperparameter experiments are forbidden unless a new explicitly versioned
project is opened.
