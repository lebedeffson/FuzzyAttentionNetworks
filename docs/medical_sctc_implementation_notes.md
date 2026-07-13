# Med-CircuitBench / SCTC Implementation Notes

Branch: `feature/med-circuitbench-sctc`

Normative source reviewed:

- `Med_CircuitBench_SCTC_Protocol_TZ_ONE_SHOT_FINAL.docx`
- user technical assignment in this task

Environment observed on 2026-07-13:

- Python: 3.14.6
- PyTorch: 2.11.0+cu128
- CUDA available: yes
- CUDA version: 12.8
- NumPy: 2.3.5
- Pandas: 2.3.3
- SciPy: 1.17.1
- scikit-learn: 1.7.2
- PyArrow: 24.0.0
- Zarr: installed in local venv during implementation
- PyYAML: 6.0.3
- Matplotlib: 3.10.8
- NetworkX: 3.6.1
- pytest: 9.0.2
- tqdm: 4.67.3
- RTK: `/home/lebedeffson/.local/bin/rtk`, version 0.43.0

Implementation boundary:

- Existing FAN source files were not rewritten.
- New code is isolated under `src/med_circuitbench`.
- Existing FAN integration is represented by `src/med_circuitbench/integration/fan_adapter.py`.
- `artifacts/medical/` is ignored by Git.

Clarified contradictions and implementation choices:

- The Word document describes a broader system including API, React UI, VAE counterfactuals and optional LLM annotation. The user assignment excludes those from MVP. This branch implements the research-core MVP first.
- The Word document mentions Python 3.11 or compatible. The available local environment is Python 3.14.6 and successfully runs the current tests.
- The Word document contains broader baseline ambitions than the MVP acceptance list. This branch includes baseline entry points and core baseline primitives; full GAM/EBM/LIME/SHAP integration remains a later extension.
- V4 replaces the incompatible `0.65` target threshold with fixed `target_threshold=0.0715`. Fallback is disabled in benchmark and smoke configs. Infection input remains restricted to `I`.

Current verified commands:

```bash
/home/lebedeffson/Code/venv/bin/python -m pytest tests/medical -q

/home/lebedeffson/Code/venv/bin/python scripts/medical/generate_benchmark.py \
  --config configs/medical/benchmark.yaml

/home/lebedeffson/Code/venv/bin/python scripts/medical/train_transformer.py \
  --dataset med_circuitbench \
  --config configs/medical/benchmark.yaml

/home/lebedeffson/Code/venv/bin/python scripts/medical/extract_activations.py \
  --dataset med_circuitbench \
  --config configs/medical/benchmark.yaml

/home/lebedeffson/Code/venv/bin/python scripts/medical/train_sctc.py \
  --dataset med_circuitbench \
  --config configs/medical/benchmark.yaml

/home/lebedeffson/Code/venv/bin/python scripts/medical/build_circuits.py \
  --dataset med_circuitbench \
  --config configs/medical/benchmark.yaml

/home/lebedeffson/Code/venv/bin/python scripts/medical/evaluate_benchmark.py \
  --manifest artifacts/medical/med_circuitbench/circuits/manifest.json

/home/lebedeffson/Code/venv/bin/python scripts/medical/build_article_tables.py \
  --manifest configs/manifests/article.yaml

/home/lebedeffson/Code/venv/bin/python scripts/medical/build_article_figures.py \
  --manifest configs/manifests/article.yaml
```

Known limitation of this increment:

- Med-CircuitBench V4 generator now uses fixed threshold `0.0715`, no fallback, and seed 42/43/44 positive rates were checked in the required 0.20--0.30 range.
- SCTC training now maps `h_l -> z_l -> a_hat_l`, computes behavior through frozen-transformer replacement, and measures validation fidelity rather than filling constants.
- Circuit construction currently uses decoder/encoder direction association and random-direction significance. Full downstream intervention propagation through later transformer layers remains the next required strengthening step before strong mechanistic claims.
- PhysioNet preparation and command entry points are present, but a full PhysioNet run requires external raw data, which must stay outside Git.
