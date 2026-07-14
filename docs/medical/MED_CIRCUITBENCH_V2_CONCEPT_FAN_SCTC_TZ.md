# Technical Assignment Draft V2: Concept-Mediated FAN + SCTC

Status: architectural draft after V1 `NO_GO`.

V1 commit remains frozen:

```text
ddcf32ae357dfedd91444c5cba18fac1af5d525e
```

New work must start from a new branch:

```text
experiment/med-circuitbench-v2
```

V1 artifacts and thresholds must not be overwritten.

## 1. Correct Scientific Line

The repository contains a developing Fuzzy Attention Network line. The current normative FAN architecture for this project is the concept-mediated FAN described in the current article:

```text
temporal encoder
-> latent representation
-> concept alignment layer
-> fuzzy membership functions
-> FAN weighting of concepts
-> decision
```

V2 is not a fuzzy-self-attention replacement study. V2 is a comparison between:

```text
explicit concept organization through FAN
vs
post-hoc mechanistic circuit discovery through SCTC
```

Main research question:

```text
Can automatically discovered sparse mechanistic features and circuits approach, recover, or extend the explicit clinical concept structure built into concept-mediated FAN?
```

## 2. Required Models

### 2.1 Standard Transformer

Purpose: black-box temporal baseline.

```text
temporal transformer
-> pooling
-> prediction
```

Pooling variants:

```text
mean
last
learnable CLS token
learned attention pooling
```

### 2.2 Concept Bottleneck Model

Purpose: isolate the effect of an explicit concept layer without fuzzy aggregation.

```text
encoder
-> concept projection
-> classifier over concepts
```

Outputs:

```text
logit
probability
concepts
latent
```

Loss:

```text
L_task + lambda_c L_concept
```

### 2.3 Concept-Mediated FAN

Purpose: canonical current FAN architecture.

```text
encoder
-> concept projection c
-> memberships mu(c)
-> relevance scores s
-> softmax weights alpha
-> contributions alpha * mu
-> decision head
```

Outputs:

```text
logit
probability
latent
concepts
memberships
concept_weights
concept_contributions
```

Loss:

```text
L_total =
L_task
+ lambda_c L_concept
+ lambda_a L_align
+ lambda_s L_sparse
```

Required membership families:

```text
gaussian
bell
sigmoid
mixed
```

Required interpretability tests:

```text
concept removal
concept insertion
top-k concept retention
concept calibration
contribution stability between seed
```

### 2.4 Transformer + SCTC

Purpose: post-hoc mechanistic circuit discovery on a black-box temporal transformer.

```text
standard transformer internals
-> SCTC
-> sparse features
-> interventions
-> circuits
```

### 2.5 FAN + SCTC

Purpose: compare discovered SCTC features against explicitly supervised FAN concepts.

```text
FAN encoder internals
-> SCTC
-> discovered features
-> matching with I,R,V,O,S and FAN concept contributions
```

Required comparison:

```text
SCTC feature <-> explicit concept activation
SCTC feature <-> membership value
SCTC feature <-> concept contribution
SCTC circuit <-> concept-level FAN decision pathway
```

## 3. Med-CircuitBench Modes

V2 must support three benchmark modes.

### 3.1 Clean

Purpose: test whether models can learn the intended latent dynamics without shortcuts.

Requirements:

```text
no informative missingness
treatment absent or randomized independently of hidden states
prediction requires latent state dynamics
```

### 3.2 Clinical-Confounded

Purpose: realistic shortcut/confounding regime.

Requirements:

```text
informative missingness enabled
treatment policy depends on hidden states
observation noise enabled
```

### 3.3 Planted-Circuit Positive Control

Purpose: distinguish data-generating graph recovery from model-internal circuit recovery.

Requirements:

```text
known internal nodes
known layers
known directions
known edges
known signs
```

If SCTC fails here, the SCTC/intervention method is not validated.

## 4. Concepts

Med-CircuitBench hidden states are exact concept targets:

```text
I: infection burden
R: inflammatory response
V: vascular dysfunction
O: organ dysfunction
S: shock risk
```

For concept-supervised models:

```text
c = (I, R, V, O, S)
```

Concept targets should be continuous by default. Binary labels may be used only for secondary AUC diagnostics.

## 5. Representation Audit Before SCTC

Before training SCTC, run representation audit for every trained model:

```text
model:
  standard transformer
  concept bottleneck
  concept-mediated FAN

extraction point:
  residual_pre
  attention_output
  residual_mid / h_ffn
  mlp_output / a_ffn
  residual_post

pooling:
  token-level
  mean
  last
  CLS
  attention pooling

state:
  I
  R
  V
  O
  S
```

Required output:

```text
representation_audit.parquet
```

Required metrics:

```text
linear R2 for continuous state value
Pearson correlation
Spearman correlation
AUC after pre-registered binarization
time lag of maximum decodability
patching/intervention effect
```

This audit decides where SCTC should be trained.

## 6. SCTC Training Changes

V2 must not train SCTC on 128 windows.

Required options:

```text
max_training_windows: all
minimum_training_episodes: 2000
n_features: [128, 256, 512]
early_stopping: enabled
decoder_norm_control: enabled
separate_sctc_validation: enabled
```

Sampling must be stratified by:

```text
target class
infection onset time
hidden state levels
treatment exposure
severity / shock score
```

Report:

```text
L0 per token
dead feature fraction
support distribution
fidelity AUROC/AUPRC/probability error
explained variance
state correlation matrix
concept contribution correlation matrix for FAN runs
```

## 7. Matched Intervention Nulls

The V1 issue must be fixed:

```text
true edge: activation-scaled ablation
random null: constant push
```

V2 requires matched nulls:

```text
ablation edge -> ablation random null
push edge -> push random null
same windows
same coefficient scale
same direction norm
same downstream target
same forward path
```

Random directions:

```text
1000 for screening
10000 for final candidate family or sequential permutation test
```

Thresholds for DR/CIE/IP must be pre-registered from:

```text
random null distribution
positive controls
power analysis
minimum stable measurable effect
```

Do not lower thresholds after viewing validation results.

## 8. Positive Controls

### Control A: Explicit FAN Concepts

Known:

```text
concept activations
membership values
concept weights
concept contributions
```

Tests:

```text
concept removal/insertion faithfulness
concept calibration
top-k concept sufficiency
concept contribution stability
```

### Control B: Planted Internal Circuit

Known:

```text
internal nodes
layers
directions
edges
signs
```

Tests:

```text
SCTC node recovery
SCTC edge recovery
CircuitF1
sign agreement
intervention effect
```

## 9. Required Outputs

Per run:

```text
model_metrics.csv
concept_metrics.csv
representation_audit.parquet
concept_contributions.parquet
sctc_feature_catalog.parquet
edge_catalog.parquet
circuit_catalog.json
intervention_effects.parquet
go_no_go.json
```

Aggregate:

```text
standard_vs_cbm_vs_fan.csv
explicit_fan_vs_sctc.csv
representation_audit_summary.csv
concept_alignment_summary.csv
circuit_recovery_summary.csv
shortcut_ablation_summary.csv
```

Figures:

```text
concept pathway diagram
concept contribution heatmaps
representation audit heatmap
FAN concept removal/insertion curves
SCTC feature-to-concept matching graph
planted circuit recovery graph
clean vs clinical-confounded shortcut comparison
```

## 10. GO/NO-GO Logic

V2 should not use only AUPRC > 0.85 as the main gate.

Required gates:

```text
FAN concept prediction R2/correlation passes pre-registered threshold
FAN removal/insertion faithfulness passes explicit concept control
standard/CBM/FAN predictive performance is reported against oracle baselines
planted-circuit SCTC recovery passes minimum CircuitF1
SCTC fidelity passes model-behavior preservation
matched-null intervention p-values are valid
FAN+SCTC matching to explicit concepts exceeds random/SAE baselines
```

If the explicit FAN works but SCTC fails:

```text
result = explicit concepts are recoverable when supervised, but not recovered post-hoc by current SCTC
```

If planted circuit fails:

```text
result = SCTC/intervention pipeline not validated for mechanistic recovery
```

If planted circuit passes but trained transformer/FAN internal circuits do not match the data graph:

```text
result = data-generating graph is not necessarily the learned computational graph
```

## 11. Non-Goals

Do not implement in V2:

```text
LLM annotation
VAE counterfactuals
clinical treatment recommendation
new web UI
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

