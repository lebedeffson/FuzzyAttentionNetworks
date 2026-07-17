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
import zarr
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.medical._common import add_run_context_args
from src.med_circuitbench.models.transformer import ClinicalTransformer, TransformerConfig
from src.med_circuitbench.sctc.layer_screening import compute_cls, sparsity_score


def _load_array(path: Path) -> np.ndarray:
    return np.asarray(zarr.load(str(path)))


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return 0.5 if len(np.unique(y)) < 2 else float(roc_auc_score(y, score))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(pearsonr(x, y).statistic)


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo, hi = float(values.min()), float(values.max())
    if abs(hi - lo) < 1e-12:
        return np.full_like(values, np.nan, dtype=float)
    return (values - lo) / (hi - lo)


def _load_benchmark_concepts(root: Path, split: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    episodes = pd.read_parquet(root / "benchmark" / "episodes.parquet")
    splits = json.loads((root / "benchmark" / "splits.json").read_text())
    ordered_ids = np.asarray(splits["train"] + splits["validation"], dtype=np.int64)
    ordered = episodes.iloc[ordered_ids]
    states = np.stack([np.stack(v).astype(np.float32) for v in ordered["states"]])[:, :36, :]
    train_states = states[split == 0]
    thresholds = np.median(train_states.mean(axis=1), axis=0)
    labels = (states.mean(axis=1) >= thresholds).astype(int)
    threshold_dict = {name: float(value) for name, value in zip(["I", "R", "V", "O", "S"], thresholds)}
    return states, labels, threshold_dict


def _decodability(a_layers: np.ndarray, split: np.ndarray, concept_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_layers = a_layers.shape[0]
    n_concepts = concept_labels.shape[1]
    aucs = np.zeros((n_layers, n_concepts), dtype=float)
    valid = np.zeros((n_layers, n_concepts), dtype=bool)
    train = split == 0
    val = split == 1
    for layer in range(n_layers):
        x_train = a_layers[layer, train].mean(axis=1)
        x_val = a_layers[layer, val].mean(axis=1)
        for k in range(n_concepts):
            y_train = concept_labels[train, k]
            y_val = concept_labels[val, k]
            if len(np.unique(y_train)) < 2:
                aucs[layer, k] = 0.5
                continue
            clf = LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear")
            pipe = (StandardScaler(), clf)
            xtr = pipe[0].fit_transform(x_train)
            clf.fit(xtr, y_train)
            score = clf.predict_proba(pipe[0].transform(x_val))[:, 1]
            aucs[layer, k] = _safe_auc(y_val, score)
            valid[layer, k] = len(np.unique(y_val)) > 1
    return aucs, valid


def _stability(a_layers: np.ndarray, split: np.ndarray, concept_labels: np.ndarray, repeats: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_layers = a_layers.shape[0]
    train = split == 0
    val_indices = np.where(split == 1)[0]
    out = np.zeros(n_layers, dtype=float)
    valid_counts = np.zeros(n_layers, dtype=int)
    for layer in range(n_layers):
        x_train = a_layers[layer, train].mean(axis=1)
        trained = []
        for k in range(concept_labels.shape[1]):
            y_train = concept_labels[train, k]
            if len(np.unique(y_train)) < 2:
                trained.append(None)
                continue
            scaler = StandardScaler().fit(x_train)
            clf = LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear")
            clf.fit(scaler.transform(x_train), y_train)
            trained.append((scaler, clf))
        corrs = []
        for _ in range(repeats):
            shuffled = val_indices.copy()
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            if half == 0:
                continue
            vectors = []
            for part in (shuffled[:half], shuffled[half:]):
                auc_vec = []
                x_part = a_layers[layer, part].mean(axis=1)
                for k, fitted in enumerate(trained):
                    if fitted is None:
                        auc_vec.append(0.5)
                        continue
                    scaler, clf = fitted
                    score = clf.predict_proba(scaler.transform(x_part))[:, 1]
                    auc_vec.append(_safe_auc(concept_labels[part, k], score))
                vectors.append(np.asarray(auc_vec))
            corr = _safe_corr(vectors[0], vectors[1])
            if corr is not None:
                corrs.append(corr)
        if corrs:
            out[layer] = float(np.median(corrs))
            valid_counts[layer] = len(corrs)
    return out, valid_counts


def _gradient_sensitivity(root: Path, dataset: str, x: np.ndarray, split: np.ndarray, batch_size: int, device: str, trim: float) -> np.ndarray:
    ckpt = torch.load(root / dataset / "transformer" / "model.ckpt", map_location=device)
    model = ClinicalTransformer(TransformerConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    val_x = x[split == 1][: min(128, int((split == 1).sum()))]
    by_layer = [[] for _ in range(model.cfg.layers)]
    for start in range(0, len(val_x), batch_size):
        xb = torch.tensor(val_x[start : start + batch_size], device=device, requires_grad=False)
        out = model(xb, return_activations=True)
        grads = torch.autograd.grad(out["logit"].sum(), out["a_ffn"], retain_graph=False, allow_unused=True)
        for layer, grad in enumerate(grads):
            if grad is None:
                continue
            norm = torch.linalg.vector_norm(grad.reshape(grad.shape[0], -1), dim=1) / (grad.shape[1] * grad.shape[2]) ** 0.5
            by_layer[layer].extend(norm.detach().cpu().numpy().tolist())
    values = []
    for norms in by_layer:
        arr = np.asarray(norms, dtype=float)
        if arr.size == 0:
            values.append(0.0)
            continue
        cutoff = np.quantile(arr, 1.0 - trim)
        values.append(float(np.median(arr[arr <= cutoff])))
    return np.asarray(values, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_run_context_args(parser)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    root = Path(cfg.get("artifacts", {}).get("root", "artifacts/medical"))
    activation_dir = root / args.dataset / "activations"
    a = _load_array(activation_dir / "a_ffn").astype(np.float32)
    x = _load_array(activation_dir / "x").astype(np.float32)
    split = _load_array(activation_dir / "split")
    seed = int(cfg["dataset"].get("seed", 42))
    if args.dataset != "med_circuitbench":
        raise SystemExit("CLS V4 implementation currently supports med_circuitbench concepts")
    _, concept_labels, thresholds = _load_benchmark_concepts(root, split)
    auc, valid_probes = _decodability(a, split, concept_labels)
    a_raw = np.maximum(0.0, 2.0 * auc - 1.0).mean(axis=1)
    s_raw = np.asarray([sparsity_score(a[layer, split == 1]) for layer in range(a.shape[0])])
    g_raw = _gradient_sensitivity(
        root,
        args.dataset,
        x,
        split,
        int(cfg["training"]["batch_size"]),
        args.device,
        float(cfg.get("cls", {}).get("trim_gradient_top_fraction", 0.01)),
    )
    r_raw, valid_stability = _stability(
        a,
        split,
        concept_labels,
        int(cfg.get("cls", {}).get("stability_repeats", 20)),
        seed,
    )
    result = compute_cls(
        auc,
        s_raw,
        g_raw,
        r_raw,
        threshold_ratio=float(cfg.get("cls", {}).get("relative_threshold", 0.4)),
        max_layers=int(cfg.get("cls", {}).get("max_layers", 3)),
    )
    selected_zero_based = [int(layer - 1) for layer in result.selected_report]
    ranking_report = [int(layer) for layer in result.ranking_report]
    out_dir = root / args.dataset / "layers"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    norms = {"A_norm": _minmax(a_raw), "S_norm": _minmax(s_raw), "G_norm": _minmax(g_raw), "R_norm": _minmax(r_raw)}
    for layer in range(a.shape[0]):
        rows.append(
            {
                "seed": seed,
                "layer": layer,
                "A_raw": float(a_raw[layer]),
                "S_raw": float(s_raw[layer]),
                "G_raw": float(g_raw[layer]),
                "R_raw": float(r_raw[layer]),
                "A_norm": float(norms["A_norm"][layer]) if not np.isnan(norms["A_norm"][layer]) else np.nan,
                "S_norm": float(norms["S_norm"][layer]) if not np.isnan(norms["S_norm"][layer]) else np.nan,
                "G_norm": float(norms["G_norm"][layer]) if not np.isnan(norms["G_norm"][layer]) else np.nan,
                "R_norm": float(norms["R_norm"][layer]) if not np.isnan(norms["R_norm"][layer]) else np.nan,
                "CLS": float(result.cls[layer]),
                "selected": bool(layer in selected_zero_based),
                "valid_probes": int(valid_probes[layer].sum()),
                "valid_stability_repeats": int(valid_stability[layer]),
                "exclusion_reason": "" if layer in selected_zero_based else "below_top3_or_threshold",
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "layer_scores.csv", index=False)
    (out_dir / "concept_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    (out_dir / "layer_selection.json").write_text(
        json.dumps({"ranking_report": ranking_report, "selected_layers": selected_zero_based}, indent=2),
        encoding="utf-8",
    )
    print({"layer_scores": str(out_dir / "layer_scores.csv"), "selected_layers": selected_zero_based})


if __name__ == "__main__":
    main()
