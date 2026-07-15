# Final Code Reuse Audit

| Component | Current path | Capabilities | Tests | Defects | Decision |
| --- | --- | --- | --- | --- | --- |
| Legacy fuzzy attention | src/fuzzy_attention.py | FuzzyMembership, FuzzyAttentionHead, MultiHeadFuzzyAttention | characterization tests | legacy interface not batch-first temporal encoder | kept as legacy; canonical temporal path in src/fan/attention |
| Advanced multimodal FAN | src/advanced_fan_model.py | AdvancedFuzzyAttention for text/image/cross attention | audit only | multimodal dependency surface not needed for MIMIC-AKI | archived, not active canonical |
| Universal FAN | src/universal_fan_model.py | SimpleFuzzyAttention | audit only | multimodal example, not clinical temporal model | archived, not active canonical |
| Canonical fuzzy temporal attention | src/fan/attention | batch-first temporal fuzzy attention encoder | tests/fan_attention | none blocking | active canonical first-FAN package |
| ConceptFAN | src/fan/concept/temporal.py | MultiSetAdditiveTemporalConceptFANModel | tests/medical/v3/test_v3_contract.py | no demo training runner before this closure | reused for demo engineering models |
| SAE | src/fan/sae | TopKSAE, dictionary health, steering helpers | tests/mimic_aki | replacement only checked in demo closure | active sparse demo component |

Runtime commit is recorded in the generated report and release manifest.
