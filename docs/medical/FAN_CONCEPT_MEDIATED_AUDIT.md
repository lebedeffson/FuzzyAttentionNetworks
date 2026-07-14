# Concept-Mediated FAN Audit for Med-CircuitBench V2

Commit audited: `ddcf32ae357dfedd91444c5cba18fac1af5d525e`

Purpose: determine whether the repository currently implements the concept-mediated FAN architecture described in the current article:

```text
temporal encoder
-> latent representation
-> concept alignment layer
-> concept activations c
-> fuzzy memberships mu(c)
-> FAN relevance scores and softmax weights alpha
-> concept contributions alpha * mu
-> prediction
```

## Executive Result

The current tracked repository does **not** contain an implementation of the concept-mediated FAN architecture from the article.

What exists:

- early fuzzy-attention modules that modulate token/feature attention scores;
- multimodal text-image FAN-style models based on BERT/ResNet feature extraction;
- the Med-CircuitBench/SCTC medical pipeline with a standard temporal transformer;
- a `FANAdapter` that wraps an arbitrary model but is not used as a research model.

What is missing for the current article architecture:

- no explicit `encoder -> concept projection -> concepts` model;
- no concept-level FAN block over `c in [0,1]^K`;
- no saved concept activations, memberships, attention weights, or concept contributions;
- no four-part article loss `L_task + lambda_c L_concept + lambda_a L_align + lambda_s L_sparse`;
- no training script that reproduces the article's concept-mediated FAN tables.

Therefore, V2 should not try to "find" the article FAN in the existing medical code. It should first implement or restore the concept-mediated FAN as the canonical previous model, then compare it with SCTC-based post-hoc circuit discovery.

## Evidence

### Early Fuzzy-Attention Code

`src/fuzzy_attention.py` implements fuzzy attention over query/key/value projections:

- `FuzzyMembership` learns Gaussian membership functions.
- `FuzzyAttentionHead` computes standard attention scores.
- It computes fuzzy scores from memberships of Q and K.
- It combines standard and fuzzy scores.
- It applies softmax and attends to V.

Relevant code:

```text
src/fuzzy_attention.py:52-59   Q/K/V projections and fuzzy_q/fuzzy_k
src/fuzzy_attention.py:79-87   standard scores + fuzzy scores combination
src/fuzzy_attention.py:96-100  softmax attention and matmul with V
src/fuzzy_attention.py:111-127 pairwise fuzzy memberships and t-norm aggregation
```

This is an early fuzzy self-attention / fuzzy attention formulation. It is not the concept-mediated FAN from the article.

### Multimodal FAN Code

`src/advanced_fan_model.py` and `src/universal_fan_model.py` implement multimodal models:

```text
BERT / ResNet encoders
-> fuzzy attention over text/image feature vectors
-> fusion
-> classifier
```

Relevant code:

```text
src/advanced_fan_model.py:13-105    AdvancedFuzzyAttention
src/advanced_fan_model.py:199-204   text/image/cross fuzzy attention modules
src/advanced_fan_model.py:253-275   fuzzy attention over text/image features and fusion
src/advanced_fan_model.py:277-285   direct classifier output
```

This code has membership functions and fuzzy attention weights, but it does not introduce explicit semantic concept targets or concept-level aggregation.

### Medical Pipeline Code

`src/med_circuitbench/models/transformer.py` implements a standard temporal transformer:

```text
input projection
-> torch.nn.MultiheadAttention
-> FFN
-> mean pooling
-> binary head
```

Relevant code:

```text
src/med_circuitbench/models/transformer.py:21-41  ClinicalTransformerLayer
src/med_circuitbench/models/transformer.py:24     torch.nn.MultiheadAttention
src/med_circuitbench/models/transformer.py:69     mean pooling head
```

The training script uses only binary classification loss:

```text
scripts/medical/train_transformer.py:120      ClinicalTransformer construction
scripts/medical/train_transformer.py:155      binary_cross_entropy_with_logits
```

There is no concept prediction loss, alignment loss, or concept sparsity loss in the medical transformer training.

### FANAdapter

`src/med_circuitbench/integration/fan_adapter.py` only wraps a passed model and calls it:

```text
src/med_circuitbench/integration/fan_adapter.py:8-23
```

It does not instantiate FAN, does not enforce a concept layer, and does not participate in the benchmark runner.

### README Mismatch

The repository README currently describes the public lineage as fuzzy attention / fuzzy transformer:

```text
README.md:187-193
```

The current article direction described by the user is different: FAN is concept-mediated and applied after the encoder to semantically defined concepts. The tracked repository does not yet encode that newer architecture as runnable source.

## Article-Architecture Checklist

| Required concept-mediated FAN component | Present in tracked code | Evidence |
| --- | --- | --- |
| Temporal encoder | Partially | `ClinicalTransformer`, but it is used as standalone classifier |
| Latent representation | Partially | residual/FFN activations saved for SCTC, not FAN concept pipeline |
| Concept projection `z -> c` | No | no model module or script found |
| Concept targets | Partially | Med-CircuitBench hidden states exist, CLS uses median labels diagnostically |
| Membership functions over concepts | No | memberships exist only in early fuzzy-attention modules |
| FAN relevance scores over concepts | No | no `s_k = r(mu_k, c)` implementation found |
| Softmax concept weights `alpha` | No | only token/feature attention softmax in early modules |
| Concept contributions `alpha * mu` | No | no saved contribution table |
| Decision from concept evidence | No | medical model predicts from pooled residual stream |
| `L_task` | Yes | BCE in medical transformer |
| `L_concept` | No | not implemented for FAN/medical |
| `L_align` | No | not implemented for FAN/medical |
| `L_sparse` / entropy over concept weights | No | SCTC sparsity exists, not FAN concept entropy |
| Removal/insertion over explicit concepts | No | current interventions target SCTC features, not explicit FAN concepts |

## Interpretation

The corrected lineage should be treated as:

```text
early fuzzy attention
-> concept-oriented FAN from current article
-> FAN + mechanistic circuit discovery
```

But the middle step is missing as code in the current repository snapshot. V2 must make that step explicit.

## Required V2 Architecture Decision

V2 should define concept-mediated FAN as a first-class model:

```text
TemporalEncoder
-> ConceptProjector
-> ConceptMembership
-> FuzzyConceptAggregator
-> DecisionHead
```

Required outputs:

```text
logit
probability
concepts
memberships
concept_weights
concept_contributions
latent
```

Required losses:

```text
L_task
L_concept
L_align
L_sparse
L_total
```

Med-CircuitBench can provide exact concept targets:

```text
I, R, V, O, S
```

This creates a valid comparison:

```text
explicit supervised concept organization via FAN
vs
post-hoc discovered sparse features and circuits via SCTC
```

## Immediate Recommendation

Do not implement V2 as "replace MultiheadAttention with FAN". That would re-open the earlier fuzzy-transformer interpretation and miss the current article's architecture.

The next implementation branch should instead be:

```text
experiment/med-circuitbench-v2
```

with the first milestone:

```text
reproduce concept-mediated FAN on Med-CircuitBench using true hidden states as concept targets
```

