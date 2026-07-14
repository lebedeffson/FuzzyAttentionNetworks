# Technical Assignment V2: Gated Concept-Mediated FAN + SCTC Program

Status: staged V2 plan after V1 `NO_GO` and concept-mediated FAN audit.

Frozen V1 commit:

```text
ddcf32ae357dfedd91444c5cba18fac1af5d525e
```

New work must start from:

```text
experiment/med-circuitbench-v2
```

Do not overwrite V1 artifacts, thresholds, or conclusions.

## 1. Core Decision

The current repository does not yet implement the current article's concept-mediated FAN. Therefore V2 must not start with another SCTC run.

The correct development line is:

```text
early fuzzy attention
-> concept-mediated FAN
-> FAN + mechanistic circuit discovery
```

The immediate practical goal is:

```text
implement and validate concept-mediated FAN as a standalone model
before launching full SCTC or FAN+SCTC experiments
```

The final V2 experiment may compare explicit FAN concepts with post-hoc SCTC concepts, but only after two positive controls pass:

```text
Control A: explicit FAN concepts are faithful
Control B: planted internal circuit is recoverable
```

## 2. Repository Organization

Old code is not deleted. It must be logically separated from the current concept-mediated architecture.

Target structure:

```text
src/fan/
├── legacy/
│   ├── fuzzy_attention.py
│   ├── advanced_fan_model.py
│   └── universal_fan_model.py
├── concept/
│   ├── memberships.py
│   ├── concept_projector.py
│   ├── fuzzy_concept_aggregator.py
│   ├── losses.py
│   └── model.py
└── common/
    ├── outputs.py
    └── diagnostics.py
```

The old public README must not be rewritten as if concept-mediated FAN already has results. First implement code, tests, and pilot evidence. Then update claims.

Required tags:

```text
fan-legacy-baseline
med-circuitbench-v1-no-go
```

## 3. Concept-Mediated FAN Model

Implement one canonical model:

```text
TemporalEncoder
-> latent z
-> ConceptProjector
-> concepts c
-> MembershipLayer
-> memberships mu
-> FuzzyConceptAggregator
-> weights alpha
-> contributions alpha * mu
-> DecisionHead
```

The model must return a structured output:

```python
{
    "logit": ...,
    "probability": ...,
    "latent": ...,
    "concepts": ...,
    "memberships": ...,
    "concept_weights": ...,
    "concept_contributions": ...,
}
```

The direct path:

```text
latent -> decision
```

is forbidden in the canonical FAN. The final decision must be produced from concept evidence:

```text
alpha * mu
```

Otherwise the concept layer is decorative and the model is not the article architecture.

## 4. Membership Functions

Implement:

```text
Gaussian
bell
sigmoid
mixed
```

Width must be positive:

```text
delta_k = softplus(rho_k) + epsilon
```

Mixed membership:

```text
mu_k =
omega_G * mu_G_k
+ omega_B * mu_B_k
+ omega_S * mu_S_k

omega = softmax(u)
```

Log membership parameters and mixture weights for every run.

## 5. Loss Function

The FAN loss is:

```text
L =
L_task
+ lambda_c * L_concept
+ lambda_a * L_align
+ lambda_s * L_sparse
```

Each component must be logged independently:

```text
L_task
L_concept
L_align
L_sparse
L_total
```

Saving only `total_loss` is not acceptable.

## 6. Med-CircuitBench Concepts

Med-CircuitBench hidden states are exact concept targets:

```text
I: infection burden
R: inflammatory response
V: vascular dysfunction
O: organ dysfunction
S: shock risk
```

Use continuous concept targets by default:

```text
c = (I, R, V, O, S)
```

Binary concept labels are allowed only for secondary AUC diagnostics.

## 7. Stage 1: FAN Pilot Before SCTC

Do not train SCTC in Stage 1.

The first pilot must validate concept-mediated FAN on Med-CircuitBench Clean.

### 7.1 FAN Variants

Run four variants:

```text
Oracle FAN-5
Predicted FAN-5
Oracle FAN-4 without S
Predicted FAN-4 without S
```

Definitions:

```text
Oracle FAN:
  receives true concepts directly and tests only membership, fuzzy aggregation, and decision head.

Predicted FAN:
  encoder predicts concepts; FAN aggregates predicted concepts.

FAN-5:
  concepts = I, R, V, O, S.

FAN-4:
  concepts = I, R, V, O.
```

Reason for FAN-4:

```text
S is already close to shock risk and may almost directly encode the target.
```

If FAN-5 works but FAN-4 collapses, that is a scientific result: the model mostly uses current shock-risk state rather than reconstructing the causal clinical history.

### 7.2 Stage 1 Gates

Stage 2 may start only if all are true:

```text
Oracle FAN learns and beats random baseline.
Concept weights are finite and sum to one.
Concept contributions are finite and reproducible.
Top-concept removal has stronger effect than random concept removal.
Top-concept insertion has stronger effect than random insertion.
Predicted FAN reaches at least 90% of Oracle FAN AUPRC.
Predicted concepts have measurable decodability and calibration.
```

No absolute administrative `AUPRC > 0.85` gate is used here. The main comparison is against Oracle FAN.

### 7.3 Stage 1 Required Outputs

```text
fan_model_metrics.csv
fan_loss_components.csv
concept_metrics.csv
concept_calibration.csv
concept_contributions.parquet
removal_insertion_results.csv
membership_parameters.json
fan_stage1_go_no_go.json
```

## 8. Stage 2: Reproduce Article Mechanism on Med-CircuitBench

Full SWaT/FD001 reproduction is not required before Med-CircuitBench, but the mechanism must be reproduced.

Required comparisons:

```text
CBM without fuzzy aggregation
FAN Gaussian
FAN bell
FAN sigmoid
FAN mixed
FAN without L_concept
FAN without L_sparse
Classifier over concepts without FAN
```

Required faithfulness tests:

```text
removal top-1
removal top-2
insertion top-1
insertion top-2
random concept removal
random concept insertion
```

Gate:

```text
dominant FAN concept removal must affect prediction more than random concept removal
dominant FAN concept insertion must retain/recover prediction better than random insertion
```

The exact article numbers are not required because this is a different dataset. The mechanism must be qualitatively validated.

## 9. Stage 3: Benchmark Modes

Implement three Med-CircuitBench regimes.

### 9.1 Clean

```text
no informative missingness
no state-dependent treatment
treatment absent or randomized independently of hidden states
fixed measurement graph
```

Purpose: test the intended latent dynamics without shortcut channels.

### 9.2 Clinical-Confounded

```text
informative missingness
state-dependent treatment
masks
delta-time
observation noise
```

Purpose: measure shortcut and confounding effects.

### 9.3 Planted Internal Circuit

This is not merely a data generator with known causal graph. It must include a model with a known internal computational graph:

```text
input
-> planted I node
-> planted R node
-> planted V node
-> planted O/S nodes
-> output
```

Requirements:

```text
nodes pinned to specific layers
orthogonal directions
known edge signs
known edge strengths
no residual bypass
no hidden direct connections
```

This model is the positive control for SCTC recovery of model-internal circuits.

## 10. Stage 4: Shortcut Audit

For each benchmark regime, train/evaluate input-subset models:

```text
observations only
masks only
delta-time only
treatments only
observations + masks
observations + treatments
full input
```

Main output:

| Regime | Input subset | AUPRC | Fraction of Full |
| --- | --- | ---: | ---: |
| Clean | observations | fact | fact |
| Clean | masks | fact | fact |
| Confounded | masks | fact | fact |
| Confounded | treatments | fact | fact |

Shortcut criterion:

```text
AUPRC_subset >= 0.8 * AUPRC_full
```

This threshold is frozen before the audit.

## 11. Stage 5: Representation Audit

Do not train SCTC across every possible combination.

For these trained models:

```text
Standard Transformer
CBM
Predicted FAN
```

save:

```text
residual_pre
attention_output
residual_mid
mlp_output
residual_post
```

For every layer and state, report:

```text
R2
Pearson
Spearman
AUROC
AUPRC
best temporal lag
```

Pooling variants:

```text
token-level
mean
last
CLS
learned attention pooling
```

SCTC is then allowed only on:

```text
two best capture points
two best pooling schemes
no more than four layers
```

The representation audit decides where SCTC is trained.

## 12. Stage 6: Pooling Audit

Pooling variants are separate trained models, not post-hoc switches:

```text
mean
last
CLS
learned attention pooling
```

Rules:

```text
CLS requires a trainable token before encoder blocks.
attention pooling requires a trainable query.
last is an ablation only.
mean remains the baseline.
```

Pooling is selected by validation before SCTC.

## 13. Stage 7: SCTC Training

V1-style training on 128 episodes is forbidden.

Main mode:

```text
all training episodes
```

Minimum allowed mode:

```text
2000 stratified episodes
```

Stratify by:

```text
target class
infection onset time
S quartile
treatment pattern
missingness quartile
```

Compare:

```text
128 features
256 features
512 features
```

But first run this comparison on the planted internal circuit model. The chosen feature count is then frozen and transferred to Standard Transformer and FAN.

### 13.1 Decoder Normalization

Decoder columns must have unit norm:

```text
||d_j||_2 = 1
```

Without this, `lambda_1` has no stable meaning because scale can move between encoder and decoder.

### 13.2 Activity Control

Log:

```text
L0 per token
dead feature fraction
support quantiles
fidelity
explained variance
state correlation
```

Target activity range:

```text
8-32 active features per token
```

The final range must be confirmed by planted-circuit control, not chosen for aesthetics.

## 14. Stage 8: Oracle Diagnostics Before Chain Search

For each true state:

```text
1. find most correlated SCTC feature
2. find strongest linear direction
3. run ablation
4. run push
5. check downstream state
6. compare with true edge
```

Diagnostic outcomes:

```text
ORACLE_FAIL
ORACLE_PASS_SCTC_FAIL
SCTC_PASS_CLEAN_ONLY
SCTC_PASS
```

Meaning:

```text
ORACLE_FAIL:
  model or intervention is wrong.

ORACLE_PASS_SCTC_FAIL:
  sparse features fail despite an available direction.

SCTC_PASS_CLEAN_ONLY:
  confounding disrupts recovery.

SCTC_PASS:
  method is ready for full comparison.
```

## 15. Stage 9: Matched Intervention Nulls

For true ablation:

```text
a_int = a - z_i * d_i
```

For random ablation:

```text
c_t = <a_t, d_rand>
a_int = a - c_t * d_rand
```

Therefore the random null removes the actual projection onto the random direction. It must not be a constant random push.

Push has a separate matched push-null with the same scale.

Requirements:

```text
same windows
same coefficient definition
same direction norm
same downstream target
same forward path
```

## 16. Stage 10: Threshold Calibration

Do not set `DR = 0.05` because `0.10` failed in V1.

Compute thresholds from controls:

```text
tau_DR = Q_0.99(|DR_negative|)
tau_CIE = Q_0.99(CIE_negative)
```

Final threshold:

```text
tau = max(tau_null, tau_practical)
```

The negative set and practical minimum must be defined before validation results are viewed.

## 17. Stage 11: Two-Step Significance Test

For all candidates:

```text
1000 random interventions
```

For candidates that pass screening:

```text
10000 random interventions
```

The hypothesis family is fixed in advance by layer pair.

## 18. Main V2 Tables

The final V2 should answer four questions, not compare everything with everything.

### Table A: Does Concept-Mediated FAN Work?

Models:

```text
Standard Transformer
CBM
Concept FAN
Oracle Concept FAN
```

Metrics:

```text
AUPRC
concept R2
concept calibration
removal effect
insertion effect
contribution stability
```

### Table B: Where Are the Concepts?

Index:

```text
model x layer x capture point x pooling
```

Metrics:

```text
R2
Pearson
best lag
patching effect
```

### Table C: Does SCTC Work?

Rows:

```text
planted circuit
standard clean
FAN clean
standard confounded
FAN confounded
```

Metrics:

```text
node recovery
CircuitF1
CIE
IP
stability
```

### Table D: Explicit Concepts vs Discovered Features

Rows:

```text
explicit concept
best SCTC feature
activation correlation
intervention agreement
contribution correlation
seed stability
```

## 19. Outcomes

### Best Outcome

```text
FAN is correctly implemented.
Explicit concepts are predicted well.
FAN faithfulness is confirmed.
Planted circuit is recovered.
SCTC recovers part of concepts and edges.
FAN+SCTC is more stable than Standard+SCTC.
```

Interpretation:

```text
explicit concept organization improves mechanistic recoverability
```

### Middle Outcome

```text
FAN works.
Planted circuit is recovered.
SCTC does not recover freely trained transformer circuits.
```

Interpretation:

```text
SCTC can recover known internal circuits, but free temporal models do not have to organize computation according to the data graph.
```

### Negative Outcome

```text
FAN works.
Planted circuit is not recovered.
```

Interpretation:

```text
problem is in SCTC or intervention pipeline, not in Med-CircuitBench.
```

### Hard Stop Outcome

```text
Oracle FAN fails removal/insertion.
```

Interpretation:

```text
FAN implementation is wrong or not faithful; full experiment must not start.
```

## 20. Immediate Deliverable: V2_PILOT_FOUNDATIONS

Do not run full V2 yet.

The next code stage must produce only:

```text
1. canonical concept-mediated FAN implementation
2. Oracle FAN and Predicted FAN on Clean Med-CircuitBench
3. planted internal circuit model and description
4. shortcut audit for Clean, Clinical-confounded, and Planted regimes
```

Pilot archive name:

```text
V2_PILOT_FOUNDATIONS
```

Required contents:

```text
FAN unit tests
FAN loss tests
Oracle FAN metrics
Predicted FAN metrics
removal/insertion results
planted circuit description
shortcut audit
resolved configs
three seeds
GO/NO-GO for pilot gates
```

Only if this pilot passes should a final implementation assignment be written for:

```text
representation audit
SCTC
full FAN + SCTC comparison
final V2 package
```

## 21. Non-Goals

Do not implement in the pilot:

```text
LLM annotation
VAE counterfactuals
clinical treatment recommendation
new web UI
full SCTC chain search
unregistered threshold lowering
claims of biological causality
```

Do not require FAN to:

```text
replace torch.nn.MultiheadAttention
act as fuzzy self-attention
produce token-level attention explanations
serve as residual-stream replacement
```

The V2 FAN target is concept-mediated FAN.
