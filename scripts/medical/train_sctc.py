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

from scripts.medical._common import add_run_context_args, git_commit
from src.med_circuitbench.metrics.fidelity import probability_fidelity_metrics
from src.med_circuitbench.models.transformer import ClinicalTransformer, TransformerConfig
from src.med_circuitbench.sctc.transcoder import SparseClinicalTranscoder, sctc_loss


def _load_array(path: Path) -> np.ndarray:
    return np.asarray(zarr.load(str(path)))


def _sample_indices(indices: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if len(indices) <= max_rows:
        return indices
    return rng.choice(indices, size=max_rows, replace=False)


def _load_transformer(root: Path, dataset: str, device: str) -> ClinicalTransformer:
    ckpt = torch.load(root / dataset / "transformer" / "model.ckpt", map_location=device)
    model = ClinicalTransformer(TransformerConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _predict_with_replacement(
    transformer: ClinicalTransformer,
    sctc: SparseClinicalTranscoder,
    x: np.ndarray,
    h: np.ndarray,
    layer_id: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    probs = []
    loader = DataLoader(TensorDataset(torch.tensor(x), torch.tensor(h)), batch_size=batch_size, shuffle=False)
    sctc.eval()
    with torch.no_grad():
        for xb, hb in loader:
            xb = xb.to(device)
            hb = hb.to(device)
            a_hat = sctc(hb)["a_hat"]
            out = transformer(xb, replacements={layer_id: a_hat})
            probs.append(out["probability"].detach().cpu().numpy())
    return np.concatenate(probs)


def _train_one(
    transformer: ClinicalTransformer,
    layer_id: int,
    train_h: np.ndarray,
    train_a: np.ndarray,
    train_x: np.ndarray,
    train_x_obs: np.ndarray,
    train_logit: np.ndarray,
    val_h: np.ndarray,
    val_a: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_base_prob: np.ndarray,
    lambda_1: float,
    lambda_b: float,
    lambda_t: float,
    n_features: int,
    device: str,
    epochs: int,
    batch_size: int,
) -> tuple[SparseClinicalTranscoder, dict, np.ndarray]:
    model = SparseClinicalTranscoder(d_model=train_h.shape[-1], n_features=n_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loader = DataLoader(
        TensorDataset(torch.tensor(train_h), torch.tensor(train_a), torch.tensor(train_x), torch.tensor(train_x_obs), torch.tensor(train_logit)),
        batch_size=batch_size,
        shuffle=True,
    )
    last_parts = None
    for _ in range(epochs):
        model.train()
        for hb, ab, xb, xob, lb in loader:
            hb = hb.to(device)
            ab = ab.to(device)
            xb = xb.to(device)
            xob = xob.to(device)
            lb = lb.to(device)
            out = model(hb)
            replaced = transformer(xb, replacements={layer_id: out["a_hat"]})
            parts = sctc_loss(
                out["a_hat"],
                ab,
                out["z"],
                replaced["logit"],
                lb.detach(),
                x=xob,
                lambda_1=lambda_1,
                lambda_b=lambda_b,
                lambda_t=lambda_t,
            )
            opt.zero_grad()
            parts.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_parts = parts
    model.eval()
    with torch.no_grad():
        vh = torch.tensor(val_h, device=device)
        va = torch.tensor(val_a, device=device)
        out = model(vh)
        rec = torch.nn.functional.mse_loss(out["a_hat"], va).item()
        z = out["z"].detach().cpu().numpy()
    replaced_prob = _predict_with_replacement(transformer, model, val_x, val_h, layer_id, batch_size, device)
    fidelity = probability_fidelity_metrics(val_base_prob, replaced_prob, val_y)
    support = (z > 1e-8).mean(axis=(0, 1))
    eligible = (support >= 0.01) & (support <= 0.30)
    metrics = {
        "L_rec": float(rec),
        "L_sparse": float(last_parts.sparse.detach().cpu().item()) if last_parts is not None else 0.0,
        "L_behavior": float(last_parts.behavior.detach().cpu().item()) if last_parts is not None else 0.0,
        "L_temporal": float(last_parts.temporal.detach().cpu().item()) if last_parts is not None else 0.0,
        "eligible_features": int(eligible.sum()),
        "eligible_fraction": float(eligible.mean()),
        **fidelity,
    }
    return model, metrics, z


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    add_run_context_args(parser)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg["dataset"].get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    root = Path(cfg.get("artifacts", {}).get("root", "artifacts/medical"))
    activation_dir = root / args.dataset / "activations"
    h_layers = _load_array(activation_dir / "h_ffn").astype(np.float32)
    a_layers = _load_array(activation_dir / "a_ffn").astype(np.float32)
    x = _load_array(activation_dir / "x").astype(np.float32)
    x_observed = _load_array(activation_dir / "x_observed").astype(np.float32)
    split = _load_array(activation_dir / "split")
    logit = _load_array(activation_dir / "logit").astype(np.float32)
    probability = _load_array(activation_dir / "probability").astype(np.float32)
    target = _load_array(activation_dir / "target").astype(np.float32)

    transformer = _load_transformer(root, args.dataset, args.device)
    train_idx = np.where(split == 0)[0]
    val_idx = np.where(split == 1)[0]
    sctc_cfg = cfg["sctc"]
    n_features = int(sctc_cfg["n_features"])
    batch_size = int(cfg["training"]["batch_size"])
    max_windows = int(sctc_cfg.get("max_training_windows", 2048))
    epochs = int(args.epochs if args.epochs is not None else sctc_cfg.get("maximum_epochs", 3))
    selected_layers = list(range(min(3, h_layers.shape[0])))
    selection_path = root / args.dataset / "layers" / "layer_selection.json"
    if selection_path.exists():
        selected_layers = json.loads(selection_path.read_text())["selected_layers"][:3]

    out_dir = root / args.dataset / "sctc"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    feature_rows = []
    accepted_layers = []

    for layer_id in selected_layers:
        train_sample = _sample_indices(train_idx, max_windows, rng)
        val_sample = _sample_indices(val_idx, max(16, min(len(val_idx), max_windows // 2)), rng)
        best = None
        best_model = None
        best_z = None
        for lambda_1 in sctc_cfg["lambda_1"]:
            for lambda_b in sctc_cfg["lambda_b"]:
                for lambda_t in sctc_cfg["lambda_t"]:
                    model, metrics, z = _train_one(
                        transformer,
                        int(layer_id),
                        h_layers[layer_id, train_sample],
                        a_layers[layer_id, train_sample],
                        x[train_sample],
                        x_observed[train_sample],
                        logit[train_sample],
                        h_layers[layer_id, val_sample],
                        a_layers[layer_id, val_sample],
                        x[val_sample],
                        target[val_sample],
                        probability[val_sample],
                        float(lambda_1),
                        float(lambda_b),
                        float(lambda_t),
                        n_features,
                        args.device,
                        epochs,
                        batch_size,
                    )
                    rejection = []
                    if metrics["eligible_features"] < int(sctc_cfg.get("minimum_eligible_features", 32)):
                        rejection.append("eligible_features")
                    if metrics["eligible_fraction"] < float(sctc_cfg.get("minimum_eligible_fraction", 0.10)):
                        rejection.append("eligible_fraction")
                    if metrics["delta_auroc"] > float(sctc_cfg["maximum_delta_auroc"]):
                        rejection.append("delta_auroc")
                    if metrics["delta_auprc"] > float(sctc_cfg["maximum_delta_auprc"]):
                        rejection.append("delta_auprc")
                    if metrics["probability_error"] > float(sctc_cfg["maximum_probability_error"]):
                        rejection.append("probability_error")
                    row = {
                        "layer": int(layer_id),
                        "lambda_1": float(lambda_1),
                        "lambda_b": float(lambda_b),
                        "lambda_t": float(lambda_t),
                        "accepted": not rejection,
                        "rejection_reason": ";".join(rejection),
                        **metrics,
                    }
                    rows.append(row)
                    if row["accepted"] and (
                        best is None
                        or (row["L_rec"], -row["lambda_1"], row["probability_error"]) < (best["L_rec"], -best["lambda_1"], best["probability_error"])
                    ):
                        best = row
                        best_model = model
                        best_z = z
        if best_model is None or best is None or best_z is None:
            continue
        accepted_layers.append(int(layer_id))
        torch.save(
            {
                "model_state": best_model.state_dict(),
                "layer": int(layer_id),
                "d_model": int(h_layers.shape[-1]),
                "n_features": n_features,
                "input_kind": "h_ffn",
                "selection": best,
            },
            ckpt_dir / f"layer_{layer_id}.ckpt",
        )
        support = (best_z > 1e-8).mean(axis=(0, 1))
        mean_activation = best_z.mean(axis=(0, 1))
        std_activation = best_z.std(axis=(0, 1))
        decoder = best_model.decoder.weight.detach().cpu().numpy().T
        decoder_norm = np.linalg.norm(decoder, axis=1)
        eligible = (support >= float(sctc_cfg["minimum_support"])) & (support <= float(sctc_cfg["maximum_support"]))
        for feature_id in np.where(eligible)[0]:
            feature_rows.append(
                {
                    "run_id": f"{args.dataset}_seed_{seed}",
                    "seed": seed,
                    "method": "SCTC",
                    "layer": int(layer_id),
                    "feature_id": int(feature_id),
                    "eligible": True,
                    "support": float(support[feature_id]),
                    "mean_activation": float(mean_activation[feature_id]),
                    "std_activation": float(std_activation[feature_id]),
                    "decoder_norm": float(decoder_norm[feature_id]),
                    "top_window_ids": [],
                    "state_correlations": [],
                    "stability": 0.0,
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "lambda_selection.csv", index=False)
    pd.DataFrame(feature_rows).to_parquet(out_dir / "feature_catalog.parquet", index=False)
    manifest = {
        "dataset": args.dataset,
        "commit": git_commit(),
        "seed": seed,
        "selected_layers": accepted_layers,
        "status": "PASS" if accepted_layers else "NO_GO_NO_ACCEPTED_SCTC_LAYER",
        "files": {"lambda_selection": "lambda_selection.csv", "feature_catalog": "feature_catalog.parquet", "checkpoints": "checkpoints"},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print({"sctc_dir": str(out_dir), "selected_layers": accepted_layers, "status": manifest["status"]})


if __name__ == "__main__":
    main()
