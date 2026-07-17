# Known Limitations

## Scope

This release has two separate benchmark roles:

```text
Med-CircuitBench: scientific validation
MIMIC-IV Demo: engineering compatibility
```

Full MIMIC-IV clinical validation is not included because full MIMIC-IV access
is an external requirement.

## Med-CircuitBench

- The production FAN remains `FAN-NoAlpha`; the stability-trained FAN did not
  pass the cross-seed contribution-stability gate.
- SCTC is validated as sparse behavioral fidelity, not as solved causal
  discovery.
- Mechanistic recovery was not semantically specific: concept controls matched
  or exceeded correct-concept arms in key recovery checks.
- Decoder-coupled concept supervision transmitted semantic gradients but broke
  behavioral fidelity and did not resolve specificity.

## MIMIC-IV Demo

- Demo results are not clinical performance evidence.
- Demo temporal inputs are pseudo-temporal scaled aggregate vectors, not real
  ICU hourly trajectories.
- Concept targets unavailable in the demo are masked and must not be treated as
  measured normal states.
- Demo faithfulness is structural-only: exact decomposition and subset hooks are
  checked, but empirical MIMIC removal/insertion curves are not evaluated.
- SAE and steering are mechanics-only. SAE fidelity is reported separately and
  may fail on the demo.
- Raw MIMIC-IV data, protected rows, and patient identifiers are intentionally
  excluded from release archives.
