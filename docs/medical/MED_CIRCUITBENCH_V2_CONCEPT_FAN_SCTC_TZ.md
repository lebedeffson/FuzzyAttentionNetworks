# Med-CircuitBench V2 Final Execution Contract

Work starts from commit `299c822d65ece1bac09b66173ad68a4af18f86cc` on branch
`experiment/med-circuitbench-v2`. V1 at
`ddcf32ae357dfedd91444c5cba18fac1af5d525e` is frozen and must not be rewritten.

This document is the final executable contract for V2. It is not a pilot,
architecture note, or audit document. The next delivery is code, three-seed
execution, gate decisions, and a final ZIP.

## Required Program

The V2 program implements and runs:

```text
Concept-mediated FAN
Oracle and Predicted FAN
CBM and Standard Transformer
Clean and Clinical-confounded Med-CircuitBench
shortcut audit
planted internal circuit
representation audit
gate-controlled SCTC stages
aggregation
figures
delivery ZIP
```

The main command is:

```bash
python scripts/medical/v2/run_v2_program.py \
  --config configs/medical/v2/program.yaml \
  --seeds 42 43 44 \
  --mode full \
  --output artifacts/medical/v2
```

The program must continue after scientific gates fail. Dependent stages are
recorded as `SKIPPED_BY_GATE` with a concrete reason. The final status is one of:

```text
V2_GO
FAN_FOUNDATION_FAIL
PLANTED_CONTROL_FAIL
SCTC_RECOVERY_FAIL
CLEAN_ONLY_SUCCESS
V2_NO_GO
```

## Concept-Mediated FAN

The canonical FAN architecture is:

```text
TemporalEncoder
-> latent z
-> ConceptProjector
-> concepts c
-> MembershipLayer
-> memberships mu
-> FAN relevance scores
-> softmax weights alpha
-> contributions alpha * mu
-> DecisionHead
```

The direct path `latent -> prediction` is forbidden. The decision head consumes
only concept contributions.

Required outputs:

```text
logit
probability
latent
concepts
memberships
concept_weights
concept_contributions
```

Required membership families:

```text
gaussian
bell
sigmoid
mixed
```

Membership widths use `softplus(raw_delta) + 1e-6`. Mixed membership weights use
softmax. FAN concept weights use `softmax(scores / temperature)` and must sum to
one within `1e-6`.

## FAN Loss

The predicted FAN loss is:

```text
L_total = L_task + lambda_c L_concept + lambda_a L_align + lambda_s L_sparse
```

where task loss is binary cross entropy, concept loss is MSE to concept targets,
alignment loss is MSE between latent and concept cosine-similarity matrices, and
sparsity loss is entropy of concept weights. Each component is logged separately.

## Concept Targets

Med-CircuitBench hidden states define concept targets at the final observed hour:

```text
FAN-5: I_35, R_35, V_35, O_35, S_35
FAN-4: I_35, R_35, V_35, O_35
```

Mean states and future hours 36-41 are not concept targets.

## Required Models

The V2 program trains and evaluates:

```text
Standard Transformer
CBM
Oracle FAN-5
Predicted FAN-5
Oracle FAN-4
Predicted FAN-4
FAN Gaussian
FAN Bell
FAN Sigmoid
FAN Mixed
FAN loss ablations
```

SCTC stages are gate-controlled and must not be interpreted unless FAN and
planted controls pass.

## Faithfulness

For each FAN output, contribution is `alpha * membership`. Required
interventions:

```text
top-1 removal
top-2 removal
top-1 insertion
top-2 insertion
random removal
random insertion
bottom-k removal
permuted ranking
```

Removal zeroes selected contributions without renormalization. Insertion keeps
only selected contributions. Metrics include probability change, signed change,
sufficiency, comprehensiveness, AUPRC change, and F1 change where applicable.

## Benchmark Modes

Clean mode disables informative missingness and state-dependent treatments:

```text
masks = 1
delta-time = 0
treatment channels absent or randomized independently
missingness independent of severity
```

Clinical-confounded mode keeps informative missingness, state-dependent
treatment, masks, delta-time, treatment channels, and observation noise.

## Shortcut Audit

For Clean and Clinical-confounded modes, Standard Transformer is trained on:

```text
observations only
masks only
delta-time only
treatments only
observations + masks
observations + treatments
full input
```

A shortcut is flagged when subset AUPRC is at least `0.8` of full-input AUPRC.
In Clean mode, masks-only, delta-only, and treatment-only models must remain at
or near prevalence.

## Planted Internal Circuit

The planted control has known model-internal nodes:

```text
layer 0: I
layer 1: R
layer 2: V
layer 3: O and S
```

Known edges:

```text
I -> R
R -> V
V -> O
V -> S
O -> S
```

The program saves the planted specification, node directions, and graph. This
control is evaluated by node precision/recall, CircuitF1, sign agreement,
negative-control false-positive rate, and directed response against matched
negative controls.

## Gates

FAN Foundation Gate passes when at least two of three seeds satisfy:

```text
Oracle FAN AUPRC >= 95 percent of oracle predictor AUPRC
Predicted FAN AUPRC >= 90 percent of Oracle FAN AUPRC
macro concept R2 >= 0.50
mean concept Pearson >= 0.65
weights finite
alpha sum error <= 1e-6
top-1 removal beats random removal with positive lower CI
top-1 insertion beats random insertion with positive lower CI
contribution-rank stability >= 0.50
```

Planted Circuit Gate passes when at least two of three seeds satisfy:

```text
node precision >= 0.80
node recall >= 0.80
CircuitF1 >= 0.80
edge sign agreement >= 0.90
negative-control FPR <= 0.05
true-edge DR exceeds Q99 matched negative control
```

Test is opened only after FAN, planted, and SCTC fidelity gates pass.

## Delivery

The final archive is named:

```text
Med_CircuitBench_V2_FINAL_<date>_<commit>.zip
```

It contains protocol, source, configs, tests, per-seed runs, aggregate results,
tables, figures, logs, manifests, checkpoints, and limitations. Stopped stages
use only `SKIPPED_BY_GATE` with reasons. Empty placeholder result tokens are not
allowed in the delivery archive.
