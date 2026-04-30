# ICP Review Experiments

This folder contains the reviewer-requested experimental runner for the
concept-oriented FAN paper:

- multi-seed reporting with mean, standard deviation, and 95% confidence interval;
- CBM baseline with the same encoder and concept predictor;
- Transformer temporal baseline;
- removal- and insertion-based concept faithfulness tests.

Run examples:

```bash
python experiments/icp/run_review_experiments.py \
  --dataset swat \
  --data-dir /path/to/ICP \
  --models fan cbm transformer cnn \
  --seeds 42 43 44 \
  --out-dir results/icp

python experiments/icp/run_review_experiments.py \
  --dataset fd001 \
  --data-dir /path/to/ICP \
  --models fan cbm transformer cnn \
  --seeds 42 43 44 \
  --out-dir results/icp
```

Outputs are written as CSV and JSON under `results/icp/`.
The runner also writes LaTeX table fragments:

- `results/icp/<dataset>_summary.tex`
- `results/icp/<dataset>_faithfulness.tex`

Quick implementation smoke check:

```bash
python experiments/icp/run_review_experiments.py \
  --dataset swat \
  --data-dir . \
  --models fan cbm transformer cnn \
  --seeds 42 \
  --epochs 1 \
  --limit-train 128 \
  --limit-test 64 \
  --smoke-test \
  --out-dir results/icp_smoke
```
