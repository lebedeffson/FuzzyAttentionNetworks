#!/usr/bin/env python3
"""Reviewer-requested ICP experiments: multi-seed baselines and faithfulness.

Expected data layouts:
  SWaT prepared arrays: X_train.npy, y_train.npy, X_test.npy, y_test.npy
  FD001 raw files: train_FD001.txt, test_FD001.txt, RUL_FD001.txt

Example:
  python experiments/icp/run_review_experiments.py \
    --dataset swat --data-dir /path/to/ICP --models fan cbm transformer \
    --seeds 42 43 44 --out-dir results/icp
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["swat", "fd001"], required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results/icp"))
    p.add_argument("--models", nargs="+", default=["fan", "cbm", "transformer", "cnn"],
                   choices=["fan", "cbm", "transformer", "cnn"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--limit-train", type=int, default=None,
                   help="Optional cap for quick/debug runs after train/val split.")
    p.add_argument("--limit-test", type=int, default=None,
                   help="Optional cap for quick/debug runs.")
    p.add_argument("--faithfulness-top-k", nargs="+", type=int, default=[1, 2])
    p.add_argument("--smoke-test", action="store_true",
                   help="Use synthetic data to validate the training path without real datasets.")
    return p


if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    build_arg_parser().parse_args()
    raise SystemExit(0)


try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:
    missing = exc.name
    raise SystemExit(
        f"Missing dependency: {missing}. Install experiment dependencies with "
        "`pip install -r requirements.txt` before running training."
    ) from exc


CONCEPTS = {
    "swat": [
        "tank_level_deviation",
        "flow_instability",
        "pressure_instability",
        "actuator_sensor_inconsistency",
    ],
    "fd001": [
        "operational_stress",
        "efficiency_loss",
        "thermal_degradation",
        "pressure_instability",
    ],
}


@dataclass
class Config:
    dataset: str
    data_dir: Path
    out_dir: Path
    model: str
    seed: int
    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    hidden_dim: int = 96
    latent_dim: int = 64
    lambda_concept: float = 0.35
    lambda_align: float = 0.05
    lambda_sparse: float = 0.01
    patience: int = 8
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    limit_train: Optional[int] = None
    limit_test: Optional[int] = None
    smoke_test: bool = False


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_fd001(path: Path) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    import pandas as pd

    cols = ["unit", "cycle"] + [f"setting_{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
    train = pd.read_csv(path / "train_FD001.txt", sep=r"\s+", header=None, names=cols)
    test = pd.read_csv(path / "test_FD001.txt", sep=r"\s+", header=None, names=cols)
    rul = pd.read_csv(path / "RUL_FD001.txt", sep=r"\s+", header=None, names=["rul"])

    max_cycle = train.groupby("unit")["cycle"].max()
    train["rul"] = train.apply(lambda r: max_cycle.loc[r["unit"]] - r["cycle"], axis=1)
    test_max = test.groupby("unit")["cycle"].max()
    final_rul = {unit: int(rul.iloc[unit - 1, 0]) for unit in test["unit"].unique()}
    test["rul"] = test.apply(lambda r: final_rul[int(r["unit"])] + test_max.loc[r["unit"]] - r["cycle"], axis=1)

    feature_cols = [c for c in cols if c not in {"unit", "cycle"}]
    return make_fd_windows(train, feature_cols), make_fd_windows(test, feature_cols)


def make_fd_windows(df: pd.DataFrame, feature_cols: List[str], window: int = 30, stride: int = 1):
    xs, ys = [], []
    for _, group in df.groupby("unit"):
        values = group[feature_cols].to_numpy(dtype=np.float32)
        labels = (group["rul"].to_numpy() <= 30).astype(np.float32)
        for start in range(0, len(group) - window + 1, stride):
            end = start + window
            xs.append(values[start:end].T)
            ys.append(labels[end - 1])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def synthetic_data(seed: int, n: int = 768, features: int = 12, steps: int = 30):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(n, features, steps)).astype(np.float32)
    score = (
        x[:, 0:2].mean(axis=(1, 2))
        + 0.8 * x[:, 3:5].std(axis=(1, 2))
        + 0.4 * rng.normal(size=n)
    )
    y = (score > np.quantile(score, 0.7)).astype(np.float32)
    split = int(n * 0.75)
    return x[:split], y[:split], x[split:], y[split:]


def cap_arrays(x: np.ndarray, y: np.ndarray, limit: Optional[int], seed: int):
    if limit is None or len(y) <= limit:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=limit, replace=False)
    return x[idx], y[idx]


def load_data(cfg: Config):
    if cfg.smoke_test:
        x_train, y_train, x_test, y_test = synthetic_data(cfg.seed)
    elif cfg.dataset == "swat":
        x_train = np.load(cfg.data_dir / "X_train.npy")
        y_train = np.load(cfg.data_dir / "y_train.npy")
        x_test = np.load(cfg.data_dir / "X_test.npy")
        y_test = np.load(cfg.data_dir / "y_test.npy")
    elif cfg.dataset == "fd001":
        prepared = [
            cfg.data_dir / "X_train_fd001.npy",
            cfg.data_dir / "y_train_fd001.npy",
            cfg.data_dir / "X_test_fd001.npy",
            cfg.data_dir / "y_test_fd001.npy",
        ]
        if all(path.exists() for path in prepared):
            x_train = np.load(prepared[0])
            y_train = np.load(prepared[1])
            x_test = np.load(prepared[2])
            y_test = np.load(prepared[3])
        else:
            (x_train, y_train), (x_test, y_test) = load_fd001(cfg.data_dir)
    else:
        raise ValueError(cfg.dataset)

    counts = np.bincount(y_train.astype(int))
    stratify = y_train if len(np.unique(y_train)) > 1 and counts.min() >= 2 else None
    if stratify is None:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=cfg.seed, stratify=None
        )
    else:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=cfg.seed, stratify=stratify
        )
    x_train, y_train = cap_arrays(x_train, y_train, cfg.limit_train, cfg.seed)
    x_val, y_val = cap_arrays(x_val, y_val, cfg.limit_train, cfg.seed + 1)
    x_test, y_test = cap_arrays(x_test, y_test, cfg.limit_test, cfg.seed + 2)
    scaler = StandardScaler()
    n_features = x_train.shape[1]
    x_train = scale_windows(scaler, x_train, fit=True)
    x_val = scale_windows(scaler, x_val)
    x_test = scale_windows(scaler, x_test)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), n_features


def scale_windows(scaler: StandardScaler, x: np.ndarray, fit: bool = False) -> np.ndarray:
    flat = x.transpose(0, 2, 1).reshape(-1, x.shape[1])
    flat = scaler.fit_transform(flat) if fit else scaler.transform(flat)
    return flat.reshape(x.shape[0], x.shape[2], x.shape[1]).transpose(0, 2, 1).astype(np.float32)


def concept_targets(x: torch.Tensor, dataset: str) -> torch.Tensor:
    level = x[:, 0:2].mean(dim=2).std(dim=1)
    flow = x[:, 2:6].diff(dim=2).abs().mean(dim=(1, 2)) if x.shape[1] >= 6 else x.diff(dim=2).abs().mean(dim=(1, 2))
    pressure = x[:, 6:10].std(dim=2).mean(dim=1) if x.shape[1] >= 10 else x.std(dim=2).mean(dim=1)
    actuator = (x[:, -4:] > 0).float().mean(dim=(1, 2)) if x.shape[1] >= 4 else x.mean(dim=(1, 2)).abs()

    if dataset == "fd001":
        level = x[:, :3].abs().mean(dim=(1, 2))
        flow = x[:, 3:10].mean(dim=(1, 2)).abs()
        pressure = x[:, 10:16].std(dim=2).mean(dim=1) if x.shape[1] >= 16 else pressure
        actuator = x[:, -5:].abs().mean(dim=(1, 2))

    c = torch.stack([level, flow, pressure, actuator], dim=1)
    lo, hi = c.min(dim=0, keepdim=True)[0], c.max(dim=0, keepdim=True)[0]
    return ((c - lo) / (hi - lo + 1e-6)).clamp(0, 1)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden: int, latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hidden, latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x).squeeze(-1))


class FANClassifier(nn.Module):
    def __init__(self, in_channels: int, cfg: Config):
        super().__init__()
        self.encoder = Encoder(in_channels, cfg.hidden_dim, cfg.latent_dim)
        self.concepts = nn.Sequential(nn.Linear(cfg.latent_dim, 4), nn.Sigmoid())
        self.centers = nn.Parameter(torch.full((4,), 0.5))
        self.widths = nn.Parameter(torch.full((4,), 0.25))
        self.attn = nn.Sequential(nn.Linear(8, cfg.hidden_dim), nn.ReLU(), nn.Linear(cfg.hidden_dim, 4))
        self.head = nn.Sequential(nn.Linear(4, cfg.hidden_dim), nn.ReLU(), nn.Linear(cfg.hidden_dim, 1))

    def membership(self, c: torch.Tensor) -> torch.Tensor:
        widths = self.widths.abs().clamp_min(1e-3)
        return torch.exp(-((c - self.centers) ** 2) / (2.0 * widths ** 2))

    def forward_from_concepts(self, c: torch.Tensor):
        mu = self.membership(c)
        alpha = torch.softmax(self.attn(torch.cat([c, mu], dim=1)), dim=1)
        logit = self.head(alpha * mu).squeeze(1)
        return logit, alpha, mu

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        c = self.concepts(z)
        logit, alpha, mu = self.forward_from_concepts(c)
        return {"logit": logit, "z": z, "concepts": c, "alpha": alpha, "membership": mu}


class CBMClassifier(nn.Module):
    def __init__(self, in_channels: int, cfg: Config):
        super().__init__()
        self.encoder = Encoder(in_channels, cfg.hidden_dim, cfg.latent_dim)
        self.concepts = nn.Sequential(nn.Linear(cfg.latent_dim, 4), nn.Sigmoid())
        self.head = nn.Linear(4, 1)

    def forward_from_concepts(self, c: torch.Tensor):
        return self.head(c).squeeze(1), torch.softmax(c.abs(), dim=1), c

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        c = self.concepts(z)
        logit, alpha, mu = self.forward_from_concepts(c)
        return {"logit": logit, "z": z, "concepts": c, "alpha": alpha, "membership": mu}


class CNNClassifier(nn.Module):
    def __init__(self, in_channels: int, cfg: Config):
        super().__init__()
        self.encoder = Encoder(in_channels, cfg.hidden_dim, cfg.latent_dim)
        self.head = nn.Linear(cfg.latent_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"logit": self.head(self.encoder(x)).squeeze(1)}


class TransformerClassifier(nn.Module):
    def __init__(self, in_channels: int, cfg: Config):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, cfg.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim, nhead=4, dim_feedforward=cfg.hidden_dim * 2,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.input_proj(x.transpose(1, 2))
        h = self.encoder(h).mean(dim=1)
        return {"logit": self.head(h).squeeze(1)}


def build_model(cfg: Config, in_channels: int):
    return {
        "fan": FANClassifier,
        "cbm": CBMClassifier,
        "cnn": CNNClassifier,
        "transformer": TransformerClassifier,
    }[cfg.model](in_channels, cfg)


def metrics_from_probs(targets: Iterable[float], probs: Iterable[float]) -> Dict[str, float]:
    probs = np.asarray(list(probs), dtype=float)
    targets = np.asarray(list(targets), dtype=int)
    pred = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(targets, pred),
        "precision": precision_score(targets, pred, zero_division=0),
        "recall": recall_score(targets, pred, zero_division=0),
        "f1": f1_score(targets, pred, zero_division=0),
        "roc_auc": roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else float("nan"),
    }


def evaluate(model, loader, cfg: Config) -> Dict[str, float]:
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(cfg.device)
            out = model(x)
            probs.extend(torch.sigmoid(out["logit"]).cpu().numpy().tolist())
            targets.extend(y.numpy().tolist())
    return metrics_from_probs(targets, probs)


def structural_alignment_loss(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    z_norm = F.normalize(z, dim=1)
    c_norm = F.normalize(c, dim=1)
    return F.mse_loss(z_norm @ z_norm.T, c_norm @ c_norm.T)


def train_one(cfg: Config) -> Tuple[Dict[str, float], nn.Module, DataLoader]:
    set_seed(cfg.seed)
    train, val, test, in_channels = load_data(cfg)
    train_loader = DataLoader(WindowDataset(*train), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(WindowDataset(*val), batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    test_loader = DataLoader(WindowDataset(*test), batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    model = build_model(cfg, in_channels).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    pos_weight = torch.tensor([(len(train[1]) - train[1].sum()) / max(train[1].sum(), 1)], device=cfg.device)

    best_state, best_f1, stale = None, -1.0, 0
    for _ in range(cfg.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            out = model(x)
            loss = F.binary_cross_entropy_with_logits(out["logit"], y, pos_weight=pos_weight)
            if cfg.model in {"fan", "cbm"}:
                target_c = concept_targets(x, cfg.dataset)
                loss = loss + cfg.lambda_concept * F.mse_loss(out["concepts"], target_c)
                loss = loss + cfg.lambda_align * structural_alignment_loss(out["z"], out["concepts"])
                if cfg.model == "fan":
                    entropy = -(out["alpha"] * (out["alpha"] + 1e-8).log()).sum(dim=1).mean()
                    loss = loss + cfg.lambda_sparse * entropy
            opt.zero_grad()
            loss.backward()
            opt.step()
        f1 = evaluate(model, val_loader, cfg)["f1"]
        if f1 > best_f1:
            best_f1, stale = f1, 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    metrics = evaluate(model, test_loader, cfg)
    metrics.update({"dataset": cfg.dataset, "model": cfg.model, "seed": cfg.seed})
    return metrics, model, test_loader


def faithfulness(model, loader, cfg: Config, ks: Iterable[int]) -> List[Dict[str, float]]:
    if cfg.model not in {"fan", "cbm"}:
        return []
    rows = []
    model.eval()
    for mode in ("remove", "insert"):
        for k in ks:
            probs, targets = [], []
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(cfg.device)
                    out = model(x)
                    c = out["concepts"].clone()
                    score = out["alpha"] * out["membership"].abs()
                    idx = score.argsort(dim=1, descending=True)
                    mask = torch.zeros_like(c)
                    mask.scatter_(1, idx[:, :k], 1.0)
                    edited = c * (1.0 - mask) if mode == "remove" else c * mask
                    logit, _, _ = model.forward_from_concepts(edited)
                    probs.extend(torch.sigmoid(logit).cpu().numpy().tolist())
                    targets.extend(y.numpy().tolist())
            metrics = metrics_from_probs(targets, probs)
            rows.append({
                "dataset": cfg.dataset,
                "model": cfg.model,
                "seed": cfg.seed,
                "mode": mode,
                "top_k": k,
                **metrics,
            })
    return rows


def add_faithfulness_deltas(rows: List[Dict[str, float]], baseline_rows: List[Dict[str, float]]) -> None:
    baseline = {(r["dataset"], r["model"], r["seed"]): r for r in baseline_rows}
    for row in rows:
        base = baseline.get((row["dataset"], row["model"], row["seed"]))
        if not base:
            continue
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            if metric in row and metric in base:
                row[f"{metric}_delta"] = float(base[metric] - row[metric])


def summarize(rows: List[Dict[str, float]], keys: List[str]) -> List[Dict[str, float]]:
    groups: Dict[Tuple, List[Dict[str, float]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out = []
    for key, items in groups.items():
        record = dict(zip(keys, key))
        for metric in (
            "accuracy", "precision", "recall", "f1", "roc_auc",
            "accuracy_delta", "precision_delta", "recall_delta", "f1_delta", "roc_auc_delta",
        ):
            vals = np.asarray([r[metric] for r in items if metric in r], dtype=float)
            if len(vals):
                record[f"{metric}_mean"] = float(np.nanmean(vals))
                record[f"{metric}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
                record[f"{metric}_ci95"] = float(1.96 * record[f"{metric}_std"] / math.sqrt(len(vals)))
        out.append(record)
    return out


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_cell(row: Dict[str, float], metric: str) -> str:
    mean = row.get(f"{metric}_mean", float("nan"))
    ci = row.get(f"{metric}_ci95", 0.0)
    if np.isnan(mean):
        return "--"
    return f"{mean:.4f} $\\pm$ {ci:.4f}"


def write_latex_summary(path: Path, rows: List[Dict[str, float]]) -> None:
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Model & Accuracy & Precision & Recall & F1-score & ROC-AUC \\\\",
        "\\midrule",
    ]
    for row in sorted(rows, key=lambda r: r["model"]):
        model = str(row["model"]).upper() if row["model"] != "fan" else "Proposed FAN"
        lines.append(
            f"{model} & {metric_cell(row, 'accuracy')} & {metric_cell(row, 'precision')} & "
            f"{metric_cell(row, 'recall')} & {metric_cell(row, 'f1')} & {metric_cell(row, 'roc_auc')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_faithfulness(path: Path, rows: List[Dict[str, float]]) -> None:
    summary = summarize(rows, ["dataset", "model", "mode", "top_k"])
    lines = [
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Model & Intervention & Top-K & F1 & $\\Delta$F1 \\\\",
        "\\midrule",
    ]
    for row in sorted(summary, key=lambda r: (r["model"], r["mode"], r["top_k"])):
        model = str(row["model"]).upper() if row["model"] != "fan" else "Proposed FAN"
        f1 = metric_cell(row, "f1")
        delta = metric_cell(row, "f1_delta") if "f1_delta_mean" in row else "--"
        lines.append(f"{model} & {row['mode']} & {row['top_k']} & {f1} & {delta} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()

    result_rows, faith_rows = [], []
    for model_name in args.models:
        for seed in args.seeds:
            cfg = Config(
                dataset=args.dataset,
                data_dir=args.data_dir,
                out_dir=args.out_dir,
                model=model_name,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                latent_dim=args.latent_dim,
                patience=args.patience,
                device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
                num_workers=args.num_workers,
                limit_train=args.limit_train,
                limit_test=args.limit_test,
                smoke_test=args.smoke_test,
            )
            metrics, model, test_loader = train_one(cfg)
            result_rows.append(metrics)
            faith_rows.extend(faithfulness(model, test_loader, cfg, args.faithfulness_top_k))
            print(json.dumps(metrics, sort_keys=True))

    add_faithfulness_deltas(faith_rows, result_rows)
    summary = summarize(result_rows, ["dataset", "model"])
    write_csv(args.out_dir / f"{args.dataset}_raw.csv", result_rows)
    write_csv(args.out_dir / f"{args.dataset}_summary.csv", summary)
    write_csv(args.out_dir / f"{args.dataset}_faithfulness.csv", faith_rows)
    write_latex_summary(args.out_dir / f"{args.dataset}_summary.tex", summary)
    write_latex_faithfulness(args.out_dir / f"{args.dataset}_faithfulness.tex", faith_rows)
    (args.out_dir / f"{args.dataset}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
