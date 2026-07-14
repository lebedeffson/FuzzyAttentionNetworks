#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import yaml

from scripts.medical._common import add_run_context_args, git_commit
from scripts.medical.mechanistic_baseline_common import (
    benchmark_states,
    feature_rows,
    load_array,
    ridge_decoder,
    save_checkpoint,
    selected_layers,
    train_probe_weights,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, required=True)
    add_run_context_args(parser)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    root = Path(cfg.get("artifacts", {}).get("root", "artifacts/medical"))
    seed = int(cfg["dataset"].get("seed", 42))
    activation_dir = root / args.dataset / "activations"
    h_layers = load_array(activation_dir / "h_ffn").astype(np.float32)
    a_layers = load_array(activation_dir / "a_ffn").astype(np.float32)
    split = load_array(activation_dir / "split")
    states = benchmark_states(root, split)
    train = split == 0
    labels = (states.mean(axis=1) >= np.median(states[train].mean(axis=1), axis=0)).astype(int)
    out = root / args.dataset / "linear_probes"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for layer in selected_layers(root, args.dataset, h_layers.shape[0]):
        x_train = h_layers[layer, train].mean(axis=1)
        weight, bias = train_probe_weights(x_train, labels[train])
        z = np.maximum(h_layers[layer] @ weight.T + bias, 0.0)
        dec = ridge_decoder(z[train], a_layers[layer, train], seed=seed)
        save_checkpoint(out, layer, weight, bias, dec, "h_ffn")
        rows.extend(feature_rows("linear_probes", seed, layer, z[train], dec))
    pd.DataFrame(rows).to_parquet(out / "feature_catalog.parquet", index=False)
    pd.DataFrame([{"method": "linear_probes", "accepted": True, "layer": r["layer"]} for r in rows]).to_csv(out / "lambda_selection.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({"dataset": args.dataset, "method": "linear_probes", "seed": seed, "commit": git_commit()}, indent=2), encoding="utf-8")
    print({"method": "linear_probes", "features": len(rows), "out": str(out)})


if __name__ == "__main__":
    main()
