#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.med_circuitbench.models.hooks import collect_ffn_activations, max_prediction_delta_with_hooks, write_activation_store_zarr
from src.med_circuitbench.models.transformer import ClinicalTransformer, TransformerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["med_circuitbench", "physionet2019"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    root = Path(cfg.get("artifacts", {}).get("root", "artifacts/medical"))
    ckpt = torch.load(root / args.dataset / "transformer" / "model.ckpt", map_location="cpu")
    model = ClinicalTransformer(TransformerConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state"])
    if args.dataset == "med_circuitbench":
        df = pd.read_parquet(root / "benchmark" / "episodes.parquet")
        splits = json.loads((root / "benchmark" / "splits.json").read_text())
        selected_ids = np.asarray(splits["train"] + splits["validation"], dtype=np.int64)
        df = df.iloc[selected_ids].copy()
        x = np.stack([np.stack(v).astype(np.float32) for v in df["model_input"]])
        y = df["target"].to_numpy(dtype=np.float32)
        window_ids = df["episode_id"].to_numpy(dtype=np.int64)
        split = np.asarray([0] * len(splits["train"]) + [1] * len(splits["validation"]), dtype=np.int8)
    else:
        prep = Path(cfg["dataset"]["prepared_dir"])
        x = np.concatenate([np.load(prep / "train_x.npy"), np.load(prep / "validation_x.npy")]).astype(np.float32)
        y = np.concatenate([np.load(prep / "train_y.npy"), np.load(prep / "validation_y.npy")]).astype(np.float32)
        window_ids = np.arange(len(y), dtype=np.int64)
        split = np.asarray([0] * len(np.load(prep / "train_y.npy")) + [1] * len(np.load(prep / "validation_y.npy")), dtype=np.int8)
    delta = max_prediction_delta_with_hooks(model, torch.tensor(x[:16]))
    if delta >= 1e-6:
        raise SystemExit(f"hooks changed prediction: {delta}")
    store = collect_ffn_activations(model, DataLoader(TensorDataset(torch.tensor(x), torch.tensor(y)), batch_size=64))
    store["window_id"] = window_ids
    store["split"] = split
    out_dir = root / args.dataset / "activations"
    write_activation_store_zarr(store, out_dir)
    print({"activation_dir": str(out_dir), "max_hook_delta": delta})


if __name__ == "__main__":
    main()
