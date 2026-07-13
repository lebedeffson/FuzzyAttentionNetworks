#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from scripts.medical._common import git_commit
from src.med_circuitbench.models.transformer import ClinicalTransformer, TransformerConfig


def _stack_inputs(values: pd.Series) -> np.ndarray:
    return np.stack([np.stack(v).astype(np.float32) for v in values])


def _load_benchmark(root: Path):
    df = pd.read_parquet(root / "benchmark" / "episodes.parquet")
    splits = json.loads((root / "benchmark" / "splits.json").read_text())
    x = _stack_inputs(df["model_input"])
    y = df["target"].to_numpy(dtype=np.float32)
    ids = df["episode_id"].to_numpy(dtype=np.int64)
    return x, y, ids, splits


def _load_physionet(cfg: dict):
    prep = Path(cfg["dataset"]["prepared_dir"])
    train_x = np.load(prep / "train_x.npy").astype(np.float32)
    train_y = np.load(prep / "train_y.npy").astype(np.float32)
    val_x = np.load(prep / "validation_x.npy").astype(np.float32)
    val_y = np.load(prep / "validation_y.npy").astype(np.float32)
    x = np.concatenate([train_x, val_x], axis=0)
    y = np.concatenate([train_y, val_y], axis=0)
    train = list(range(len(train_y)))
    validation = list(range(len(train_y), len(train_y) + len(val_y)))
    ids = np.arange(len(y), dtype=np.int64)
    return x, y, ids, {"train": train, "validation": validation}


def _metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    pred = (prob >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return {
        "auprc": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "auroc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.5,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate": float(np.mean(y_true)),
    }


def _predict(model: ClinicalTransformer, x: np.ndarray, y: np.ndarray, ids: np.ndarray, batch_size: int, device: str) -> tuple[np.ndarray, pd.DataFrame]:
    loader = DataLoader(TensorDataset(torch.tensor(x), torch.tensor(y), torch.tensor(ids)), batch_size=batch_size, shuffle=False)
    probs, logits, targets, item_ids = [], [], [], []
    model.eval()
    with torch.no_grad():
        for xb, yb, ib in loader:
            out = model(xb.to(device))
            probs.append(out["probability"].detach().cpu().numpy())
            logits.append(out["logit"].detach().cpu().numpy())
            targets.append(yb.numpy())
            item_ids.append(ib.numpy())
    prob = np.concatenate(probs)
    frame = pd.DataFrame(
        {
            "id": np.concatenate(item_ids),
            "target": np.concatenate(targets).astype(float),
            "logit": np.concatenate(logits).astype(float),
            "probability": prob.astype(float),
        }
    )
    return prob, frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["med_circuitbench", "physionet2019"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg["dataset"].get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    root = Path(cfg.get("artifacts", {}).get("root", "artifacts/medical"))
    if args.dataset == "med_circuitbench":
        x, y, ids, splits = _load_benchmark(root)
    else:
        x, y, ids, splits = _load_physionet(cfg)

    model_cfg = TransformerConfig(
        input_dim=int(cfg["model"]["input_dim"]),
        layers=int(cfg["model"]["layers"]),
        d_model=int(cfg["model"]["d_model"]),
        heads=int(cfg["model"]["heads"]),
        d_ffn=int(cfg["model"]["d_ffn"]),
        dropout=float(cfg["model"]["dropout"]),
        sequence_length=int(cfg["dataset"]["window"]),
    )
    device = args.device
    model = ClinicalTransformer(model_cfg).to(device)
    train_ids = np.asarray(splits["train"], dtype=np.int64)
    val_ids = np.asarray(splits["validation"], dtype=np.int64)
    batch_size = int(cfg["training"]["batch_size"])
    loader = DataLoader(
        TensorDataset(torch.tensor(x[train_ids]), torch.tensor(y[train_ids])),
        batch_size=batch_size,
        shuffle=True,
    )

    positives = float(y[train_ids].sum())
    negatives = float(len(train_ids) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    minimum_epochs = int(cfg["training"]["minimum_epochs"])
    maximum_epochs = int(cfg["training"]["maximum_epochs"])
    patience = int(cfg["training"]["patience"])
    best_score = -1.0
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    history = []
    stale = 0

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logit"], yb, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        val_prob, _ = _predict(model, x[val_ids], y[val_ids], ids[val_ids], batch_size, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"validation_{k}": v for k, v in _metrics(y[val_ids], val_prob).items()}}
        history.append(row)
        score = row["validation_auprc"]
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= minimum_epochs and stale >= patience:
            break

    model.load_state_dict(best_state)
    train_prob, train_pred = _predict(model, x[train_ids], y[train_ids], ids[train_ids], batch_size, device)
    val_prob, val_pred = _predict(model, x[val_ids], y[val_ids], ids[val_ids], batch_size, device)
    metrics = {
        "best_epoch": best_epoch,
        **{f"train_{k}": v for k, v in _metrics(y[train_ids], train_prob).items()},
        **{f"validation_{k}": v for k, v in _metrics(y[val_ids], val_prob).items()},
    }

    run_dir = root / args.dataset / "transformer"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": model_cfg.__dict__, "metrics": metrics}, run_dir / "model.ckpt")
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    pd.DataFrame([metrics]).to_csv(run_dir / "model_metrics.csv", index=False)
    pd.concat([train_pred.assign(split="train"), val_pred.assign(split="validation")], ignore_index=True).to_parquet(
        run_dir / "predictions.parquet", index=False
    )
    manifest = {
        "dataset": args.dataset,
        "commit": git_commit(),
        "seed": seed,
        "device": device,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "files": {
            "checkpoint": "model.ckpt",
            "metrics": "model_metrics.csv",
            "history": "training_history.csv",
            "predictions": "predictions.parquet",
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(metrics)


if __name__ == "__main__":
    main()
