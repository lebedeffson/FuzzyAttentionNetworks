# Med-CircuitBench / SCTC Run Notes

This branch contains the isolated medical research module under `src/med_circuitbench`.

Recommended checks:

```bash
python -m pytest tests/medical -q --disable-warnings
python -m compileall src/med_circuitbench scripts/medical
```

Smoke pipeline:

```bash
python scripts/medical/generate_benchmark.py --config configs/medical/smoke.yaml
python scripts/medical/train_transformer.py --dataset med_circuitbench --config configs/medical/smoke.yaml
python scripts/medical/extract_activations.py --dataset med_circuitbench --config configs/medical/smoke.yaml
python scripts/medical/screen_layers.py --dataset med_circuitbench --config configs/medical/smoke.yaml
python scripts/medical/train_sctc.py --dataset med_circuitbench --config configs/medical/smoke.yaml --epochs 1
python scripts/medical/build_circuits.py --dataset med_circuitbench --config configs/medical/smoke.yaml
```

V4 status: Med-CircuitBench uses fixed `target_threshold=0.0715`, no fallback, and infection input only into `I`.
