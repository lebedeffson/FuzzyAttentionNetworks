# Review Fix Checklist

Branch: `fix-review-q2-package`

Addressed reviewer requirements:

- Conclusion cleaned: removed recursive/duplicated wording and rewrote the limitation paragraph.
- Full text scan: removed stale artifact phrases and shortened figure captions.
- Figures replaced: generated reproducible publication-style schematic figures in `paper/figures/`.
- Weak direct-download conference PDF reference replaced with a DOI-backed journal article reference.
- Multi-seed evaluation added: SWaT and FD001 are reported over seeds `42, 43, 44` with mean and 95% CI.
- External baselines added: CNN, CBM, and Transformer are implemented in `experiments/icp/run_review_experiments.py`.
- Faithfulness added: removal- and insertion-based concept interventions are reported for FAN and CBM.
- Reproducibility added: raw CSV/JSON outputs are stored in `paper/results/`; LaTeX table fragments are in `paper/tables/`.

Current empirical summary:

- SWaT: Proposed FAN achieves the best F1-score: `0.9363 +/- 0.0099`.
- FD001: Proposed FAN outperforms CNN/CBM in F1-score; Transformer remains higher: `FAN 0.7288 +/- 0.0230`, `Transformer 0.7606 +/- 0.0594`.
- Faithfulness: removing top-ranked FAN concepts strongly reduces F1 on both datasets.

Run commands are documented in `experiments/icp/README.md`.
