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

from scripts.medical._common import add_run_context_args, git_commit
from scripts.medical.mechanistic_baseline_common import feature_rows, load_array, selected_layers
from src.med_circuitbench.sctc.transcoder import SparseClinicalTranscoder


def _train(a_train: np.ndarray, n_features: int, lambda_1: float, epochs: int, batch_size: int, device: str) -> tuple[SparseClinicalTranscoder, float, np.ndarray]:
    model = SparseClinicalTranscoder(d_model=a_train.shape[-1], n_features=n_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(torch.tensor(a_train)), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for (ab,) in loader:
            ab = ab.to(device)
            out = model(ab)
            loss = torch.nn.functional.mse_loss(out["a_hat"], ab) + float(lambda_1) * out["z"].mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(a_train, device=device))
        rec = torch.nn.functional.mse_loss(out["a_hat"], torch.tensor(a_train, device=device)).item()
        z = out["z"].detach().cpu().numpy()
    return model, float(rec), z


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    add_run_context_args(parser)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    root = Path(cfg.get("artifacts", {}).get("root", "artifacts/medical"))
    seed = int(cfg["dataset"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    activation_dir = root / args.dataset / "activations"
    a_layers = load_array(activation_dir / "a_ffn").astype(np.float32)
    split = load_array(activation_dir / "split")
    train = split == 0
    out = root / args.dataset / "sae"
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    feature_catalog = []
    n_features = int(cfg["sctc"]["n_features"])
    epochs = int(args.epochs if args.epochs is not None else cfg["sctc"].get("maximum_epochs", 3))
    batch_size = int(cfg["training"]["batch_size"])
    for layer in selected_layers(root, args.dataset, a_layers.shape[0]):
        best = None
        for lambda_1 in cfg["sctc"]["lambda_1"]:
            model, rec, z_train = _train(a_layers[layer, train], n_features, float(lambda_1), epochs, batch_size, args.device)
            support = (z_train > 1e-8).mean(axis=(0, 1))
            eligible = (support >= float(cfg["sctc"]["minimum_support"])) & (support <= float(cfg["sctc"]["maximum_support"]))
            row = {
                "method": "sae",
                "layer": int(layer),
                "lambda_1": float(lambda_1),
                "accepted": bool(eligible.sum() >= int(cfg["sctc"].get("minimum_eligible_features", 1))),
                "L_rec": rec,
                "eligible_features": int(eligible.sum()),
                "delta_auroc": 0.0,
                "delta_auprc": 0.0,
                "probability_error": 0.0,
            }
            rows.append(row)
            if row["accepted"] and (best is None or row["L_rec"] < best[0]["L_rec"]):
                best = (row, model, z_train)
        if best is None:
            continue
        _, model, z_train = best
        torch.save(
            {
                "model_state": model.state_dict(),
                "layer": int(layer),
                "d_model": int(a_layers.shape[-1]),
                "n_features": n_features,
                "input_kind": "a_ffn",
            },
            ckpt_dir / f"layer_{layer}.ckpt",
        )
        feature_catalog.extend(feature_rows("sae", seed, layer, z_train, model.decoder.weight.detach().cpu().numpy()))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "lambda_selection.csv", index=False)
    pd.DataFrame(feature_catalog).to_parquet(out / "feature_catalog.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps({"dataset": args.dataset, "method": "sae", "seed": seed, "commit": git_commit()}, indent=2), encoding="utf-8")
    print({"method": "sae", "features": len(feature_catalog), "out": str(out)})


if __name__ == "__main__":
    main()
