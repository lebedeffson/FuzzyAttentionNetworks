# Dual Benchmark Final Status

Status:

```text
PRACTICE_CLOSED_DUAL_BENCHMARK_RESEARCH_COMPLETE
```

The final practice release has two separate roles.

Med-CircuitBench is the scientific validation block. It contains the registered
FAN and SCTC results, including the retained FAN-NoAlpha production model and
the closed mixed or negative SCTC line. The release does not use a fixed AUPRC
threshold such as 0.85 as a scientific gate. The result is judged by the
registered experiment structure, oracle checks, faithfulness diagnostics,
controls, and stability reporting.

MIMIC-IV Demo is the engineering compatibility block. It verifies real-format
ingestion, cohort construction, labels, model wiring, SAE mechanics, steering
mechanics, bundles, CLI wrappers, and reporting. It is not clinical performance
evidence and does not replace full MIMIC-IV.

Final registered interpretation:

```text
Med-CircuitBench: scientific validation
MIMIC-IV Demo: engineering compatibility
Full MIMIC-IV: external data requirement for future clinical validation
```

After this release, new hyperparameter experiments are forbidden unless a new
explicitly versioned project is opened.
