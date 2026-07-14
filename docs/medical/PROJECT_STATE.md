# Current Project State

Last updated: 2026-07-14

## Active Branch

```text
fix/med-circuitbench-v2-2-final
```

## Frozen Results

- V1: `ddcf32ae357dfedd91444c5cba18fac1af5d525e`, status `V1_NO_GO`.
- V2 foundation: `b4bd5661544079db5a9c6d274b24bf05787891be`, status `FAN_FOUNDATION_FAIL`.
- V2.1: `d404aabf8d93844e11878263103bbdb5d2ab16bc`, status `FAN_FOUNDATION_FAIL`.
- V2.2 prior delivery: `70ab6cf1bba6cabd5fb292e69884e8c5bdd8947c`, status `FAN_FOUNDATION_FAIL`.

## Current Correction

The prior V2.2 runner evaluated FAN with a Ridge/LogisticRegression surrogate
instead of training and evaluating the existing neural `TemporalConceptFANModel`.

This has been corrected in the V2.2 runner:

- Oracle and Predicted FAN now instantiate `TemporalConceptFANModel`.
- Strict Predicted FAN uses concept-stage training, freezes encoder/projector,
  then trains only the temporal aggregator, membership layer, FAN aggregator and
  decision head.
- Static and temporal FAN variants use the same neural FAN path.
- Gaussian, bell, sigmoid and mixed memberships are supported by the runner.
- Faithfulness uses the frozen FAN `DecisionHead` on altered `alpha * mu`
  contributions and does not retrain a classifier after intervention.
- FAN gate reads measured `alpha` and temporal `beta` normalization errors.
- FAN encoder depth, heads and FFN width are now configurable from V2.2 config.

## Verified Checks For Current Correction

```text
python -m compileall -q src/fan/concept scripts/medical/v2_2 tests/medical/v2_2
python -m pytest tests/medical/v2_2 -q
python -m pytest tests/medical/v2_2 tests/medical/v2 -q
python scripts/medical/v2_2/run_v2_2_program.py --config configs/medical/v2_2/smoke.yaml --seeds 42 --mode smoke --output artifacts/medical/v2_2_real_fan_smoke
```

Smoke output is a technical verification only and is not a scientific V2.2
three-seed result.

## Current Interpretation

The previous statement "FAN Foundation Gate failed" remains preserved as a
frozen result for the prior V2.2 delivery, but it should not be interpreted as
a valid failure of neural concept-mediated FAN because that delivery did not
actually use the neural FAN classes in its main FAN evaluation path.

The corrected runner must be used for any new full V2.2 scientific run.

## Next Scientific Step

Run the corrected full V2.2 program on seeds `42`, `43`, and `44`, then package
and validate a new delivery ZIP whose commit hash includes this correction.
