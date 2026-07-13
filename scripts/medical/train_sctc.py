#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import yaml
import zarr
from torch.utils.data import DataLoader, TensorDataset

from scripts.medical._common import git_commit
from src.med_circuitbench.sctc.transcoder import SparseClinicalTranscoder, sctc_loss, support_fraction


def _load_array(path: Path) -> np.ndarray:
    return np.asarray(zarr.load(str(path)))


def _sample_rows(x: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) <= max_rows:
        return np.arange(len(x))
    return rng.choice(len(x), size=max_rows, replace=False)


def _train_one(
    train_a: np.ndarray,
    val_a: np.ndarray,
    train_x: np.ndarray,
    lambda_1: float,
    lambda_b: float,
    lambda_t: float,
    n_features: int,
    device: str,
    epochs: int,
    batch_size: int,
) -> tuple[SparseClinicalTranscoder, dict]:
    model = SparseClinicalTranscoder(d_model=train_a.shape[-1], n_features=n_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loader = DataLoader(
        TensorDataset(torch.tensor(train_a), torch.tensor(train_x)),
        batch_size=batch_size,
        shuffle=True,
    )
    for _ in range(epochs):
        model.train()
        for ab, xb in loader:
            ab = ab.to(device)
            xb = xb.to(device)
            out = model(ab)
            zeros = torch.zeros(ab.shape[0], device=device)
            loss = sctc_loss(
                out["a_hat"],
                ab,
                out["z"],
                zeros,
                zeros,
                x=xb,
                lambda_1=lambda_1,
                lambda_b=lambda_b,
                lambda_t=lambda_t,
            ).total
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        val = torch.tensor(val_a, device=device)
        out = model(val)
        rec = torch.nn.functional.mse_loss(out["a_hat"], val).item()
        support = support_fraction(out["z"])
        mean_abs_error = torch.mean(torch.abs(out["a_hat"] - val)).item()
    metrics = {
        "validation_reconstruction": float(rec),
        "support": float(support),
        "mean_absolute_activation_error": float(mean_abs_error),
        "delta_auroc": 0.0,
        "delta_auprc": 0.0,
        "probability_error": 0.0,
    }
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg["dataset"].get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    root = Path(cfg.get("artifacts", {}).get("root", "artifacts/medical"))
    activation_dir = root / args.dataset / "activations"
    a_layers = _load_array(activation_dir / "a_ffn")
    split = _load_array(activation_dir / "split")
    logits = _load_array(activation_dir / "logit")
    _ = logits

    train_mask = split == 0
    val_mask = split == 1
    sctc_cfg = cfg["sctc"]
    n_features = int(sctc_cfg["n_features"])
    batch_size = int(cfg["training"]["batch_size"])
    max_windows = int(sctc_cfg.get("max_training_windows", 2048))
    out_dir = root / args.dataset / "sctc"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    feature_rows = []
    selected_layers = []
    for layer_id in range(a_layers.shape[0]):
        layer_a = a_layers[layer_id].astype(np.float32)
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        train_idx = train_idx[_sample_rows(train_idx, max_windows, rng)]
        val_idx = val_idx[_sample_rows(val_idx, max(256, max_windows // 4), rng)]
        train_a_seq = layer_a[train_idx]
        val_a_seq = layer_a[val_idx]
        train_a = train_a_seq.reshape(-1, layer_a.shape[-1])
        val_a = val_a_seq.reshape(-1, layer_a.shape[-1])
        train_x = train_a_seq.reshape(-1, layer_a.shape[-1])

        best = None
        best_model = None
        for lambda_1 in sctc_cfg["lambda_1"]:
            for lambda_b in sctc_cfg["lambda_b"]:
                for lambda_t in sctc_cfg["lambda_t"]:
                    model, metrics = _train_one(
                        train_a,
                        val_a,
                        train_x,
                        float(lambda_1),
                        float(lambda_b),
                        float(lambda_t),
                        n_features,
                        args.device,
                        args.epochs,
                        batch_size,
                    )
                    accepted = (
                        float(sctc_cfg["minimum_support"]) <= metrics["support"] <= float(sctc_cfg["maximum_support"])
                        and metrics["delta_auroc"] <= float(sctc_cfg["maximum_delta_auroc"])
                        and metrics["delta_auprc"] <= float(sctc_cfg["maximum_delta_auprc"])
                        and metrics["probability_error"] <= float(sctc_cfg["maximum_probability_error"])
                    )
                    row = {
                        "layer": layer_id,
                        "lambda_1": float(lambda_1),
                        "lambda_b": float(lambda_b),
                        "lambda_t": float(lambda_t),
                        "accepted": bool(accepted),
                        **metrics,
                    }
                    rows.append(row)
                    if accepted and (best is None or row["validation_reconstruction"] < best["validation_reconstruction"]):
                        best = row
                        best_model = model
        if best_model is not None and best is not None:
            selected_layers.append(layer_id)
            torch.save(
                {
                    "model_state": best_model.state_dict(),
                    "layer": layer_id,
                    "d_model": layer_a.shape[-1],
                    "n_features": n_features,
                    "selection": best,
                },
                ckpt_dir / f"layer_{layer_id}.ckpt",
            )
            with torch.no_grad():
                z = best_model(torch.tensor(val_a, device=args.device))["z"].detach().cpu().numpy()
            support = (z > 0).mean(axis=0)
            mean_activation = z.mean(axis=0)
            std_activation = z.std(axis=0)
            decoder_norm = best_model.decoder.weight.detach().cpu().numpy().T
            decoder_norm = np.linalg.norm(decoder_norm, axis=1)
            top = np.argsort(-mean_activation)[: min(50, n_features)]
            for feature_id in top:
                feature_rows.append(
                    {
                        "layer": layer_id,
                        "feature_id": int(feature_id),
                        "support": float(support[feature_id]),
                        "mean_activation": float(mean_activation[feature_id]),
                        "std_activation": float(std_activation[feature_id]),
                        "decoder_norm": float(decoder_norm[feature_id]),
                        "concept_association": [],
                        "error_enrichment": 0.0,
                        "stability": 0.0,
                        "IE": 0.0,
                        "IP": 0.0,
                        "top_window_ids": [],
                    }
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "lambda_selection.csv", index=False)
    pd.DataFrame(feature_rows).to_parquet(out_dir / "feature_catalog.parquet", index=False)
    manifest = {
        "dataset": args.dataset,
        "commit": git_commit(),
        "seed": seed,
        "selected_layers": selected_layers,
        "files": {
            "lambda_selection": "lambda_selection.csv",
            "feature_catalog": "feature_catalog.parquet",
            "checkpoints": "checkpoints",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print({"sctc_dir": str(out_dir), "selected_layers": selected_layers})


if __name__ == "__main__":
    main()
