#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.med_circuitbench.models.transformer import ClinicalTransformer, TransformerConfig


def _load_benchmark(root: Path):
    df = pd.read_parquet(root / "benchmark" / "episodes.parquet")
    splits = yaml.safe_load((root / "benchmark" / "splits.json").read_text()) if False else None
    import json

    splits = json.loads((root / "benchmark" / "splits.json").read_text())
    x = np.stack([np.stack(v).astype(np.float32) for v in df["model_input"]])
    y = df["target"].to_numpy(dtype=np.float32)
    return x, y, splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["med_circuitbench", "physionet2019"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    root = Path(cfg.get("artifacts", {}).get("root", "artifacts/medical"))
    if args.dataset == "med_circuitbench":
        x, y, splits = _load_benchmark(root)
    else:
        prep = Path(cfg["dataset"]["prepared_dir"])
        x = np.load(prep / "train_x.npy")
        y = np.load(prep / "train_y.npy")
        splits = {"train": list(range(len(y))), "validation": list(range(len(y)))}
    model_cfg = TransformerConfig(
        input_dim=int(cfg["model"]["input_dim"]),
        layers=int(cfg["model"]["layers"]),
        d_model=int(cfg["model"]["d_model"]),
        heads=int(cfg["model"]["heads"]),
        d_ffn=int(cfg["model"]["d_ffn"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    model = ClinicalTransformer(model_cfg)
    train_ids = splits["train"][: min(len(splits["train"]), 2048)]
    val_ids = splits["validation"][: min(len(splits["validation"]), 512)]
    loader = DataLoader(
        TensorDataset(torch.tensor(x[train_ids]), torch.tensor(y[train_ids])),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    for _ in range(max(1, min(2, int(cfg["training"]["maximum_epochs"])))):
        model.train()
        for xb, yb in loader:
            out = model(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logit"], yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(x[val_ids]))["probability"].numpy()
    metrics = {
        "validation_auprc": float(average_precision_score(y[val_ids], probs)) if len(np.unique(y[val_ids])) > 1 else 0.0,
        "validation_auroc": float(roc_auc_score(y[val_ids], probs)) if len(np.unique(y[val_ids])) > 1 else 0.5,
    }
    run_dir = root / args.dataset / "transformer"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": model_cfg.__dict__, "metrics": metrics}, run_dir / "model.ckpt")
    pd.DataFrame([metrics]).to_csv(run_dir / "model_metrics.csv", index=False)
    print(metrics)


if __name__ == "__main__":
    main()
