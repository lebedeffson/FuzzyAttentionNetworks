from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.med_circuitbench.sctc.transcoder import SparseClinicalTranscoder


def load_array(path: Path) -> np.ndarray:
    return np.asarray(zarr.load(str(path)))


def selected_layers(root: Path, dataset: str, n_layers: int) -> list[int]:
    path = root / dataset / "layers" / "layer_selection.json"
    if path.exists():
        return [int(x) for x in json.loads(path.read_text())["selected_layers"][:3]]
    return list(range(min(3, n_layers)))


def benchmark_states(root: Path, split: np.ndarray) -> np.ndarray:
    episodes = pd.read_parquet(root / "benchmark" / "episodes.parquet")
    splits = json.loads((root / "benchmark" / "splits.json").read_text())
    ids = np.asarray(splits["train"] + splits["validation"], dtype=np.int64)
    states = np.stack([np.stack(v).astype(np.float32) for v in episodes.iloc[ids]["states"]])[:, :36, :]
    return states


def ridge_decoder(z: np.ndarray, target: np.ndarray, alpha: float = 1e-3, max_rows: int = 20000, seed: int = 42) -> np.ndarray:
    z2 = z.reshape(-1, z.shape[-1]).astype(np.float64)
    y2 = target.reshape(-1, target.shape[-1]).astype(np.float64)
    if len(z2) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(z2), size=max_rows, replace=False)
        z2 = z2[idx]
        y2 = y2[idx]
    lhs = z2.T @ z2 + alpha * np.eye(z2.shape[1])
    rhs = z2.T @ y2
    return np.linalg.solve(lhs, rhs).T.astype(np.float32)


def save_checkpoint(
    out_dir: Path,
    layer: int,
    encoder_weight: np.ndarray,
    encoder_bias: np.ndarray,
    decoder_weight: np.ndarray,
    input_kind: str,
) -> SparseClinicalTranscoder:
    n_features, d_model = encoder_weight.shape
    model = SparseClinicalTranscoder(d_model=d_model, n_features=n_features)
    with torch.no_grad():
        model.encoder[0].weight.copy_(torch.tensor(encoder_weight, dtype=torch.float32))
        model.encoder[0].bias.copy_(torch.tensor(encoder_bias, dtype=torch.float32))
        model.decoder.weight.copy_(torch.tensor(decoder_weight, dtype=torch.float32))
        model.decoder.bias.zero_()
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "layer": int(layer),
            "d_model": int(d_model),
            "n_features": int(n_features),
            "input_kind": input_kind,
        },
        ckpt_dir / f"layer_{layer}.ckpt",
    )
    return model


def feature_rows(method: str, seed: int, layer: int, z: np.ndarray, decoder_weight: np.ndarray) -> list[dict]:
    support = (z > 1e-8).mean(axis=(0, 1))
    mean_activation = z.mean(axis=(0, 1))
    std_activation = z.std(axis=(0, 1))
    decoder_norm = np.linalg.norm(decoder_weight.T, axis=1)
    rows = []
    for feature_id in range(z.shape[-1]):
        rows.append(
            {
                "run_id": f"{method}_seed_{seed}",
                "seed": seed,
                "method": method,
                "layer": int(layer),
                "feature_id": int(feature_id),
                "eligible": bool(0.01 <= support[feature_id] <= 0.30),
                "support": float(support[feature_id]),
                "mean_activation": float(mean_activation[feature_id]),
                "std_activation": float(std_activation[feature_id]),
                "decoder_norm": float(decoder_norm[feature_id]) if feature_id < len(decoder_norm) else 0.0,
                "top_window_ids": [],
                "state_correlations": [],
                "stability": 0.0,
            }
        )
    return rows


def train_probe_weights(x_train: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = []
    biases = []
    for k in range(labels.shape[1]):
        y = labels[:, k]
        if len(np.unique(y)) < 2:
            weights.append(np.zeros(x_train.shape[1], dtype=np.float32))
            biases.append(-10.0)
            continue
        scaler = StandardScaler().fit(x_train)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        clf.fit(scaler.transform(x_train), y)
        weight = clf.coef_[0] / (scaler.scale_ + 1e-8)
        bias = clf.intercept_[0] - float(np.sum(clf.coef_[0] * scaler.mean_ / (scaler.scale_ + 1e-8)))
        weights.append(weight.astype(np.float32))
        biases.append(float(bias))
    return np.stack(weights).astype(np.float32), np.asarray(biases, dtype=np.float32)
