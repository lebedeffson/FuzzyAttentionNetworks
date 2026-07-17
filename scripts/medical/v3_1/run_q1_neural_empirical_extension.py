#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fan.concept.temporal import (
    MultiSetAdditiveTemporalConceptFANModel,
    SequenceTemporalEncoder,
    TemporalConceptAggregator,
    TokenConceptProjector,
)
from scripts.medical.v2.run_v2_program import make_episodes, split_frame


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONCEPT_NAMES = ["infection", "inflammation", "hemodynamics", "organ_dysfunction", "shock"]
ARMS = ["ConceptFAN-NoAlpha", "NoFuzzy", "TemporalCEM", "TemporalPCBM", "PlainTransformer"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    buf = io.BytesIO()
    cpu_state = {k: v.detach().cpu().contiguous() for k, v in sorted(state.items())}
    torch.save(cpu_state, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0 - 1e-8)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if mask.any():
            out += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(out)


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0 - 1e-8)
    return {
        "AUROC": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan"),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "ECE": ece_score(y, p),
        "NLL": float(log_loss(y, p, labels=[0, 1])),
    }


def best_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    grid = np.r_[np.linspace(0.50, 1.0, 11), np.linspace(1.1, 4.0, 30)]
    losses = [log_loss(y, sigmoid_np(logits / t), labels=[0, 1]) for t in grid]
    return float(grid[int(np.argmin(losses))])


def make_q1_splits(cfg: dict, n_samples: int, seed: int) -> dict[str, pd.DataFrame]:
    local = json.loads(json.dumps(cfg, default=str))
    local["dataset"]["n_samples"] = int(n_samples)
    return split_frame(make_episodes(seed, local, "clean"), seed)


def sequence_targets(df: pd.DataFrame, n_concepts: int = 5) -> np.ndarray:
    states = np.stack(df["states"].map(lambda v: np.asarray(v, dtype=np.float32)).to_numpy())
    return states[:, :36, :n_concepts].astype(np.float32)


def input_tensor(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["model_input"].map(lambda v: np.asarray(v, dtype=np.float32)).to_numpy()).astype(np.float32)


@dataclass
class Arrays:
    x_train: np.ndarray
    y_train: np.ndarray
    c_train: np.ndarray
    x_cal: np.ndarray
    y_cal: np.ndarray
    c_cal: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    c_val: np.ndarray
    x_holdout: np.ndarray
    y_holdout: np.ndarray
    c_holdout: np.ndarray


def build_arrays(splits: dict[str, pd.DataFrame]) -> Arrays:
    train_df = splits["train"].reset_index(drop=True)
    cut = int(0.80 * len(train_df))
    fit_df = train_df.iloc[:cut].reset_index(drop=True)
    cal_df = train_df.iloc[cut:].reset_index(drop=True)
    val_df = splits["validation"].reset_index(drop=True)
    holdout_df = splits["test"].reset_index(drop=True)
    return Arrays(
        x_train=input_tensor(fit_df),
        y_train=fit_df["target"].to_numpy(np.float32),
        c_train=sequence_targets(fit_df),
        x_cal=input_tensor(cal_df),
        y_cal=cal_df["target"].to_numpy(np.float32),
        c_cal=sequence_targets(cal_df),
        x_val=input_tensor(val_df),
        y_val=val_df["target"].to_numpy(np.float32),
        c_val=sequence_targets(val_df),
        x_holdout=input_tensor(holdout_df),
        y_holdout=holdout_df["target"].to_numpy(np.float32),
        c_holdout=sequence_targets(holdout_df),
    )


class ConceptEncoderBackbone(nn.Module):
    def __init__(self, input_dim: int, sequence_length: int, latent_dim: int, n_concepts: int, heads: int, ffn: int, layers: int):
        super().__init__()
        self.encoder = SequenceTemporalEncoder(input_dim, latent_dim, sequence_length, 0.05, num_layers=layers, nhead=heads, dim_feedforward=ffn)
        self.projector = TokenConceptProjector(latent_dim, n_concepts)
        self.temporal = TemporalConceptAggregator(n_concepts, sequence_length)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        trajectories = self.projector(h)
        summaries, temporal_weights = self.temporal(trajectories, "attention")
        return h, trajectories, summaries, temporal_weights


class NoFuzzyModel(nn.Module):
    def __init__(self, input_dim: int, sequence_length: int, latent_dim: int, n_concepts: int, heads: int, ffn: int, layers: int):
        super().__init__()
        self.backbone = ConceptEncoderBackbone(input_dim, sequence_length, latent_dim, n_concepts, heads, ffn, layers)
        self.head = nn.Linear(n_concepts, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, trajectories, summaries, temporal_weights = self.backbone(x)
        local = summaries * self.head.weight.view(1, -1)
        logit = self.head.bias.view(1) + local.sum(dim=-1)
        return {
            "logit": logit,
            "probability": torch.sigmoid(logit),
            "latent_sequence": h,
            "concept_trajectories": trajectories,
            "concept_summaries": summaries,
            "temporal_concept_weights": temporal_weights,
            "local_contributions": local,
        }


class TemporalCEMModel(nn.Module):
    def __init__(self, input_dim: int, sequence_length: int, latent_dim: int, n_concepts: int, heads: int, ffn: int, layers: int):
        super().__init__()
        self.backbone = ConceptEncoderBackbone(input_dim, sequence_length, latent_dim, n_concepts, heads, ffn, layers)
        self.local_nets = nn.ModuleList(
            [nn.Sequential(nn.Linear(1, 8), nn.GELU(), nn.Linear(8, 1)) for _ in range(n_concepts)]
        )
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, trajectories, summaries, temporal_weights = self.backbone(x)
        local = torch.cat([net(summaries[:, i : i + 1]) for i, net in enumerate(self.local_nets)], dim=1)
        logit = self.bias + local.sum(dim=-1)
        return {
            "logit": logit,
            "probability": torch.sigmoid(logit),
            "latent_sequence": h,
            "concept_trajectories": trajectories,
            "concept_summaries": summaries,
            "temporal_concept_weights": temporal_weights,
            "local_contributions": local,
        }


class TemporalPCBMModel(nn.Module):
    def __init__(self, input_dim: int, sequence_length: int, latent_dim: int, n_concepts: int, heads: int, ffn: int, layers: int):
        super().__init__()
        self.backbone = ConceptEncoderBackbone(input_dim, sequence_length, latent_dim, n_concepts, heads, ffn, layers)
        self.weight = nn.Parameter(torch.empty(n_concepts))
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.xavier_uniform_(self.weight.view(1, -1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, trajectories, summaries, temporal_weights = self.backbone(x)
        local = summaries * self.weight.view(1, -1)
        logit = self.bias + local.sum(dim=-1)
        return {
            "logit": logit,
            "probability": torch.sigmoid(logit),
            "latent_sequence": h,
            "concept_trajectories": trajectories,
            "concept_summaries": summaries,
            "temporal_concept_weights": temporal_weights,
            "local_contributions": local,
        }


class PlainTransformerModel(nn.Module):
    def __init__(self, input_dim: int, sequence_length: int, latent_dim: int, heads: int, ffn: int, layers: int):
        super().__init__()
        self.encoder = SequenceTemporalEncoder(input_dim, latent_dim, sequence_length, 0.05, num_layers=layers, nhead=heads, dim_feedforward=ffn)
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        pooled = h.mean(dim=1)
        logit = self.head(pooled).squeeze(-1)
        return {"logit": logit, "probability": torch.sigmoid(logit), "latent_sequence": h}


def model_config(cfg: dict) -> dict:
    return {
        "input_dim": int(cfg["model"]["input_dim"]),
        "sequence_length": int(cfg["dataset"]["observed_window"]),
        "latent_dim": 32,
        "n_concepts": 5,
        "heads": 2,
        "ffn": 64,
        "layers": 1,
        "n_memberships": 3,
        "membership": "mixed",
        "alpha_mode": "no_alpha",
    }


def make_model(arm: str, mcfg: dict) -> nn.Module:
    if arm == "ConceptFAN-NoAlpha":
        return MultiSetAdditiveTemporalConceptFANModel(
            input_dim=mcfg["input_dim"],
            sequence_length=mcfg["sequence_length"],
            latent_dim=mcfg["latent_dim"],
            n_concepts=mcfg["n_concepts"],
            n_memberships=mcfg["n_memberships"],
            membership=mcfg["membership"],
            oracle=False,
            temporal_mode="attention",
            dropout=0.05,
            encoder_layers=mcfg["layers"],
            encoder_heads=mcfg["heads"],
            encoder_ffn=mcfg["ffn"],
            alpha_mode="no_alpha",
        )
    if arm == "NoFuzzy":
        return NoFuzzyModel(mcfg["input_dim"], mcfg["sequence_length"], mcfg["latent_dim"], mcfg["n_concepts"], mcfg["heads"], mcfg["ffn"], mcfg["layers"])
    if arm == "TemporalCEM":
        return TemporalCEMModel(mcfg["input_dim"], mcfg["sequence_length"], mcfg["latent_dim"], mcfg["n_concepts"], mcfg["heads"], mcfg["ffn"], mcfg["layers"])
    if arm == "TemporalPCBM":
        return TemporalPCBMModel(mcfg["input_dim"], mcfg["sequence_length"], mcfg["latent_dim"], mcfg["n_concepts"], mcfg["heads"], mcfg["ffn"], mcfg["layers"])
    if arm == "PlainTransformer":
        return PlainTransformerModel(mcfg["input_dim"], mcfg["sequence_length"], mcfg["latent_dim"], mcfg["heads"], mcfg["ffn"], mcfg["layers"])
    raise ValueError(arm)


def model_output_dict(out) -> dict[str, torch.Tensor]:
    if isinstance(out, dict):
        return out
    return {
        "logit": out.logit,
        "probability": out.probability,
        "latent_sequence": out.latent_sequence,
        "concept_trajectories": out.concept_trajectories,
        "concept_summaries": out.concept_summaries,
        "temporal_concept_weights": out.temporal_concept_weights,
        "local_contributions": out.signed_decision_contributions,
        "memberships": out.memberships,
        "concept_weights": out.concept_weights,
    }


def run_grid(n_runs: int) -> list[dict[str, int]]:
    init_seeds = list(range(3101, 3111))
    order_seeds = list(range(4101, 4104))
    rows = []
    for init_seed in init_seeds:
        for order_seed in order_seeds:
            rows.append({"run_id": len(rows) + 1, "initialization_seed": init_seed, "data_order_seed": order_seed})
    return rows[: int(n_runs)]


def loader(x: np.ndarray, y: np.ndarray, c: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    gen = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)), torch.from_numpy(c)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen,
    )


def train_one(arm: str, arrays: Arrays, cfg: dict, run_meta: dict[str, int], out_dir: Path, *, epochs: int, batch_size: int) -> dict:
    set_all_seeds(run_meta["initialization_seed"])
    mcfg = model_config(cfg)
    model = make_model(arm, mcfg).to(DEVICE)
    if arm == "ConceptFAN-NoAlpha":
        with torch.no_grad():
            summaries = torch.from_numpy(arrays.c_train.mean(axis=1)).to(DEVICE)
            model.membership.initialize_from_quantiles(summaries)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    train_loader = loader(arrays.x_train, arrays.y_train, arrays.c_train, batch_size, True, run_meta["data_order_seed"])
    history = []
    for epoch in range(int(epochs)):
        losses = []
        model.train()
        for xb, yb, cb in train_loader:
            xb, yb, cb = xb.to(DEVICE), yb.to(DEVICE), cb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model_output_dict(model(xb))
            bce = F.binary_cross_entropy_with_logits(out["logit"], yb)
            concept_loss = xb.new_zeros(())
            if "concept_trajectories" in out:
                concept_loss = F.mse_loss(out["concept_trajectories"], cb)
            loss = bce + (0.75 * concept_loss if arm != "PlainTransformer" else 0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.item()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    cal_eval = evaluate_model(model, arrays.x_cal, arrays.y_cal, arrays.c_cal, batch_size)
    temperature = best_temperature(cal_eval["logit"], arrays.y_cal)
    val_eval = evaluate_model(model, arrays.x_val, arrays.y_val, arrays.c_val, batch_size)
    raw_prob = sigmoid_np(val_eval["logit"])
    cal_prob = sigmoid_np(val_eval["logit"] / temperature)
    metrics = {
        **run_meta,
        "model_arm": arm,
        "checkpoint": "",
        "parameter_sha256": "",
        "temperature": temperature,
        "train_epochs": int(epochs),
        "raw_AUPRC": binary_metrics(arrays.y_val, raw_prob)["AUPRC"],
        "calibrated_AUPRC": binary_metrics(arrays.y_val, cal_prob)["AUPRC"],
        "concept_MAE": float(np.mean(np.abs(val_eval["concept_trajectories"] - arrays.c_val))) if val_eval.get("concept_trajectories") is not None else float("nan"),
    }
    ckpt_dir = out_dir / "CHECKPOINTS" / arm / f"run_{run_meta['run_id']:02d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    param_sha = state_dict_sha256(model.state_dict())
    ckpt = {
        "arm": arm,
        "run": run_meta,
        "model_config": mcfg,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "parameter_sha256": param_sha,
        "temperature": float(temperature),
        "history": history,
        "metrics": metrics,
    }
    ckpt_path = ckpt_dir / "checkpoint.pt"
    torch.save(ckpt, ckpt_path)
    metrics["checkpoint"] = str(ckpt_path.relative_to(out_dir))
    metrics["parameter_sha256"] = param_sha
    (ckpt_dir / "manifest.json").write_text(json.dumps({k: v for k, v in ckpt.items() if k != "state_dict"}, indent=2, default=str), encoding="utf-8")
    return metrics


def evaluate_model(model: nn.Module, x: np.ndarray, y: np.ndarray, c: np.ndarray, batch_size: int) -> dict:
    model.eval()
    logits, probs = [], []
    concept_traj, local = [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(DEVICE)
            out = model_output_dict(model(xb))
            logits.append(out["logit"].detach().cpu().numpy())
            probs.append(out["probability"].detach().cpu().numpy())
            if "concept_trajectories" in out:
                concept_traj.append(out["concept_trajectories"].detach().cpu().numpy())
            if "local_contributions" in out:
                local.append(out["local_contributions"].detach().cpu().numpy())
    return {
        "target": y.astype(int),
        "logit": np.concatenate(logits),
        "probability": np.concatenate(probs),
        "concept_trajectories": np.concatenate(concept_traj, axis=0) if concept_traj else None,
        "local_contributions": np.concatenate(local, axis=0) if local else None,
    }


def load_checkpoint(path: Path) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = make_model(ckpt["arm"], ckpt["model_config"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def checkpoint_paths(output: Path) -> list[Path]:
    return sorted((output / "CHECKPOINTS").glob("*/*/checkpoint.pt"))


def collect_outputs(output: Path, arrays: Arrays, batch_size: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows, local_rows, pred_rows = [], [], []
    for ckpt_path in checkpoint_paths(output):
        model, ckpt = load_checkpoint(ckpt_path)
        ev = evaluate_model(model, arrays.x_val, arrays.y_val, arrays.c_val, batch_size)
        raw_prob = sigmoid_np(ev["logit"])
        cal_prob = sigmoid_np(ev["logit"] / float(ckpt["temperature"]))
        for calibrated, prob in [(False, raw_prob), (True, cal_prob)]:
            metric_rows.append(
                {
                    **ckpt["run"],
                    "model_arm": ckpt["arm"],
                    "checkpoint": str(ckpt_path.relative_to(output)),
                    "parameter_sha256": ckpt["parameter_sha256"],
                    "calibrated": calibrated,
                    "temperature": float(ckpt["temperature"] if calibrated else 1.0),
                    **binary_metrics(arrays.y_val, prob),
                }
            )
        if ev["local_contributions"] is not None:
            abs_mean = np.abs(ev["local_contributions"]).mean(axis=0)
            for concept_idx, concept in enumerate(CONCEPT_NAMES):
                local_rows.append(
                    {
                        **ckpt["run"],
                        "model_arm": ckpt["arm"],
                        "checkpoint": str(ckpt_path.relative_to(output)),
                        "concept": concept,
                        "mean_abs_local_contribution": float(abs_mean[concept_idx]),
                    }
                )
            for episode_id in range(min(200, len(arrays.y_val))):
                for concept_idx, concept in enumerate(CONCEPT_NAMES):
                    pred_rows.append(
                        {
                            **ckpt["run"],
                            "model_arm": ckpt["arm"],
                            "episode_id": episode_id,
                            "concept": concept,
                            "local_contribution": float(ev["local_contributions"][episode_id, concept_idx]),
                        }
                    )
    return pd.DataFrame(metric_rows), pd.DataFrame(local_rows), pd.DataFrame(pred_rows)


def pairwise_stability(local_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fan = local_summary[local_summary["model_arm"].eq("ConceptFAN-NoAlpha")]
    pivot = fan.pivot_table(index=["run_id", "checkpoint"], columns="concept", values="mean_abs_local_contribution").reset_index()
    concept_cols = CONCEPT_NAMES
    for i in range(len(pivot)):
        for j in range(i + 1, len(pivot)):
            a = pivot.loc[i, concept_cols].to_numpy(float)
            b = pivot.loc[j, concept_cols].to_numpy(float)
            top_a = set(np.argsort(-a)[:2].tolist())
            top_b = set(np.argsort(-b)[:2].tolist())
            rows.append(
                {
                    "run_a": int(pivot.loc[i, "run_id"]),
                    "run_b": int(pivot.loc[j, "run_id"]),
                    "checkpoint_a": pivot.loc[i, "checkpoint"],
                    "checkpoint_b": pivot.loc[j, "checkpoint"],
                    "spearman": float(stats.spearmanr(a, b).statistic),
                    "kendall": float(stats.kendalltau(a, b).statistic),
                    "jaccard_top2": float(len(top_a & top_b) / max(1, len(top_a | top_b))),
                }
            )
    return pd.DataFrame(rows)


def perturb_raw_x(x: np.ndarray, scenario: str, level, rng: np.random.Generator) -> np.ndarray:
    out = x.copy()
    if scenario == "noise":
        out[:, :, :8] = out[:, :, :8] + rng.normal(0.0, float(level), size=out[:, :, :8].shape).astype(np.float32)
    elif scenario == "mcar":
        mask = rng.random(out[:, :, :8].shape) < float(level)
        out[:, :, :8][mask] = 0.0
        out[:, :, 8:16][mask] = 0.0
    elif scenario == "block_missing":
        block = max(1, int(round(out.shape[1] * float(level))))
        start = max(0, (out.shape[1] - block) // 2)
        out[:, start : start + block, :8] = 0.0
        out[:, start : start + block, 8:16] = 0.0
    else:
        raise ValueError(scenario)
    return out.astype(np.float32)


def robustness(output: Path, arrays: Arrays, cfg: dict, batch_size: int) -> pd.DataFrame:
    rows = []
    shifted = make_q1_splits(cfg, n_samples=max(1200, min(3000, len(arrays.x_val) * 3)), seed=9090)
    shift_arrays = build_arrays(shifted)
    for ckpt_path in checkpoint_paths(output):
        model, ckpt = load_checkpoint(ckpt_path)
        rng = np.random.default_rng(int(ckpt["run"]["initialization_seed"]) + int(ckpt["run"]["data_order_seed"]))
        for scenario, levels in {"noise": [0.0, 0.02, 0.05, 0.10], "mcar": [0.0, 0.05, 0.10, 0.20], "block_missing": [0.0, 0.10, 0.20]}.items():
            for level in levels:
                xp = perturb_raw_x(arrays.x_val, scenario, level, rng)
                ev = evaluate_model(model, xp, arrays.y_val, arrays.c_val, batch_size)
                rows.append({**ckpt["run"], "model_arm": ckpt["arm"], "scenario": scenario, "level": level, **binary_metrics(arrays.y_val, sigmoid_np(ev["logit"] / ckpt["temperature"]))})
        ev = evaluate_model(model, shift_arrays.x_val, shift_arrays.y_val, shift_arrays.c_val, batch_size)
        rows.append({**ckpt["run"], "model_arm": ckpt["arm"], "scenario": "generator_shift", "level": "heldout_generator_seed_9090", **binary_metrics(shift_arrays.y_val, sigmoid_np(ev["logit"] / ckpt["temperature"]))})
    return pd.DataFrame(rows)


def frozen_interventions(output: Path, arrays: Arrays, batch_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    intervention_rows, control_rows = [], []
    subset_masks = {
        "shock_only": torch.tensor([[0, 0, 0, 0, 1]], dtype=torch.bool, device=DEVICE),
        "infection_shock": torch.tensor([[1, 0, 0, 0, 1]], dtype=torch.bool, device=DEVICE),
        "hemodynamics_organ_shock": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool, device=DEVICE),
    }
    for ckpt_path in sorted((output / "CHECKPOINTS" / "ConceptFAN-NoAlpha").glob("run_*/checkpoint.pt")):
        model, ckpt = load_checkpoint(ckpt_path)
        x = torch.from_numpy(arrays.x_val[:256]).to(DEVICE)
        y = arrays.y_val[:256].astype(int)
        with torch.no_grad():
            base = model_output_dict(model(x))
            base_prob = base["probability"].detach().cpu().numpy()
            local = base["local_contributions"].detach().cpu().numpy()
            rank = np.argsort(-np.abs(local), axis=1)
            for mode in ["top1_removed", "top2_removed"]:
                mask = torch.ones((x.shape[0], 5), dtype=torch.bool, device=DEVICE)
                k = 1 if mode == "top1_removed" else 2
                for i in range(x.shape[0]):
                    mask[i, rank[i, :k]] = False
                out = model_output_dict(model(x, concept_mask=mask))
                prob = out["probability"].detach().cpu().numpy()
                intervention_rows.append({**ckpt["run"], "checkpoint": str(ckpt_path.relative_to(output)), "intervention": mode, "mean_abs_delta": float(np.mean(np.abs(prob - base_prob))), "target_rate": float(y.mean())})
            for name, base_mask in subset_masks.items():
                mask = base_mask.expand(x.shape[0], -1)
                out = model_output_dict(model(x, concept_mask=mask))
                prob = out["probability"].detach().cpu().numpy()
                control_rows.append({**ckpt["run"], "checkpoint": str(ckpt_path.relative_to(output)), "control": f"frozen_subset_{name}", **binary_metrics(y, prob)})
            summaries = base["concept_summaries"]
            h = base["latent_sequence"]
            traj = base["concept_trajectories"]
            tw = base["temporal_concept_weights"]
            perm = torch.randperm(summaries.shape[0], device=DEVICE)
            shuffled = model.forward_from_summaries(h, traj, summaries[perm], tw)
            random_summaries = torch.rand_like(summaries)
            random_out = model.forward_from_summaries(h, traj, random_summaries, tw)
            for name, prob in [
                ("shuffled_predicted_concepts", shuffled.probability.detach().cpu().numpy()),
                ("random_predicted_concepts", random_out.probability.detach().cpu().numpy()),
            ]:
                control_rows.append({**ckpt["run"], "checkpoint": str(ckpt_path.relative_to(output)), "control": name, **binary_metrics(y, prob)})
    return pd.DataFrame(intervention_rows), pd.DataFrame(control_rows)


def train_torch_leakage(errors_train: np.ndarray, y_train: np.ndarray, errors_val: np.ndarray, y_val: np.ndarray, seed: int) -> dict:
    set_all_seeds(seed)
    model = nn.Linear(errors_train.shape[1], 1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-3)
    xtr = torch.from_numpy(errors_train.astype(np.float32)).to(DEVICE)
    ytr = torch.from_numpy(y_train.astype(np.float32)).to(DEVICE)
    for _ in range(120):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(xtr).squeeze(-1), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = model(torch.from_numpy(errors_val.astype(np.float32)).to(DEVICE)).squeeze(-1).cpu().numpy()
    return binary_metrics(y_val, sigmoid_np(logits))


def leakage_audit(output: Path, arrays: Arrays, batch_size: int) -> pd.DataFrame:
    rows = []
    for ckpt_path in sorted((output / "CHECKPOINTS" / "ConceptFAN-NoAlpha").glob("run_*/checkpoint.pt")):
        model, ckpt = load_checkpoint(ckpt_path)
        cal = evaluate_model(model, arrays.x_cal, arrays.y_cal, arrays.c_cal, batch_size)
        hold = evaluate_model(model, arrays.x_holdout, arrays.y_holdout, arrays.c_holdout, batch_size)
        cal_err = np.abs(cal["concept_trajectories"].mean(axis=1) - arrays.c_cal.mean(axis=1))
        hold_err = np.abs(hold["concept_trajectories"].mean(axis=1) - arrays.c_holdout.mean(axis=1))
        rows.append({**ckpt["run"], "checkpoint": str(ckpt_path.relative_to(output)), **train_torch_leakage(cal_err, arrays.y_cal, hold_err, arrays.y_holdout, ckpt["run"]["initialization_seed"])})
    return pd.DataFrame(rows)


def validate_neural_outputs(output: Path, arrays: Arrays, batch_size: int) -> dict:
    checks: list[dict] = []

    def add(check: str, passed: bool, detail: str) -> None:
        checks.append({"check": check, "passed": bool(passed), "detail": detail})

    paths = checkpoint_paths(output)
    add("150 checkpoints present", len(paths) == 30 * len(ARMS), f"count={len(paths)}")
    metrics = pd.read_csv(output / "TABLES" / "q1_neural_run_metrics.csv")
    add("five neural arms", set(metrics["model_arm"]) == set(ARMS), ",".join(sorted(metrics["model_arm"].unique())))
    add("30 runs per arm", metrics.groupby("model_arm")["run_id"].nunique().eq(30).all(), str(metrics.groupby("model_arm")["run_id"].nunique().to_dict()))
    add("raw and calibrated rows", set(metrics["calibrated"].astype(str)) == {"False", "True"}, ",".join(sorted(metrics["calibrated"].astype(str).unique())))
    unique_metric_counts = metrics[~metrics["calibrated"]].groupby("model_arm")["AUPRC"].nunique()
    add("more than three unique results per arm", unique_metric_counts.ge(4).all(), str(unique_metric_counts.to_dict()))
    sha_counts = metrics.groupby("model_arm")["parameter_sha256"].nunique()
    add("30 unique parameter SHA per arm", sha_counts.eq(30).all(), str(sha_counts.to_dict()))
    stability = pd.read_csv(output / "TABLES" / "q1_neural_stability_pairwise.csv")
    add("435 pairwise ConceptFAN checkpoint pairs", len(stability) == math.comb(30, 2), f"rows={len(stability)}")
    add("different checkpoints only", (stability["checkpoint_a"] != stability["checkpoint_b"]).all(), "checkpoint_a != checkpoint_b")
    for rel in [
        "TABLES/q1_neural_local_contributions.csv",
        "TABLES/q1_neural_frozen_interventions.csv",
        "TABLES/q1_neural_controls.csv",
        "TABLES/q1_neural_leakage_audit.csv",
        "TABLES/q1_neural_robustness.csv",
        "TABLES/q1_neural_model_summary.csv",
        "FIGURES/q1_neural_model_auprc.png",
    ]:
        path = output / rel
        add(f"artifact exists: {rel}", path.exists() and path.stat().st_size > 0, str(path))
    source = (ROOT / "scripts" / "medical" / "v3_1" / "run_q1_neural_empirical_extension.py").read_text(encoding="utf-8")
    banned_tokens = [
        "Logistic" + "Regression",
        "P" + "CA",
        "concept_feature" + "_matrix",
        "last" + "6",
        "slope" + "6",
    ]
    for forbidden in banned_tokens:
        add(f"forbidden surrogate token absent: {forbidden}", forbidden not in source, forbidden)
    for ckpt_path in paths[: len(ARMS)] + paths[-len(ARMS) :]:
        model, ckpt = load_checkpoint(ckpt_path)
        state_sha = state_dict_sha256(model.state_dict())
        add(f"checkpoint sha matches {ckpt_path.parent.parent.name}/{ckpt_path.parent.name}", state_sha == ckpt["parameter_sha256"], state_sha)
        xb = torch.from_numpy(arrays.x_val[:16]).to(DEVICE)
        with torch.no_grad():
            out = model_output_dict(model(xb))
            pert = arrays.x_val[:16].copy()
            pert[:, :, :8] += 0.05
            out2 = model_output_dict(model(torch.from_numpy(pert).to(DEVICE)))
        add(f"forward path accepts raw x {ckpt['arm']}", tuple(xb.shape) == (16, 36, 27) and "probability" in out, str(tuple(xb.shape)))
        add(f"raw observation perturbation changes output {ckpt['arm']}", float(torch.mean(torch.abs(out["probability"] - out2["probability"])).item()) > 1e-8, ckpt["arm"])
        if ckpt["arm"] != "PlainTransformer":
            add(f"local neural concepts present {ckpt['arm']}", "concept_trajectories" in out and "local_contributions" in out, ckpt["arm"])
    passed = all(row["passed"] for row in checks)
    return {
        "status": "Q1_NEURAL_VERIFIER_PASS" if passed else "Q1_NEURAL_VERIFIER_FAIL",
        "created_utc": now(),
        "passed": passed,
        "checks": checks,
        "failed_checks": [row for row in checks if not row["passed"]],
    }


def write_outputs(output: Path, arrays: Arrays, cfg: dict, batch_size: int) -> dict:
    tables = output / "TABLES"
    figs = output / "FIGURES"
    manifests = output / "MANIFESTS"
    for p in [tables, figs, manifests, output / "REPORTS"]:
        p.mkdir(parents=True, exist_ok=True)
    metrics, local_summary, local_episode = collect_outputs(output, arrays, batch_size)
    metrics.to_csv(tables / "q1_neural_run_metrics.csv", index=False)
    local_summary.to_csv(tables / "q1_neural_local_summary.csv", index=False)
    local_episode.to_csv(tables / "q1_neural_local_contributions.csv", index=False)
    stability = pairwise_stability(local_summary)
    stability.to_csv(tables / "q1_neural_stability_pairwise.csv", index=False)
    interventions, controls = frozen_interventions(output, arrays, batch_size)
    interventions.to_csv(tables / "q1_neural_frozen_interventions.csv", index=False)
    controls.to_csv(tables / "q1_neural_controls.csv", index=False)
    leak = leakage_audit(output, arrays, batch_size)
    leak.to_csv(tables / "q1_neural_leakage_audit.csv", index=False)
    rob = robustness(output, arrays, cfg, batch_size)
    rob.to_csv(tables / "q1_neural_robustness.csv", index=False)
    summary = (
        metrics.groupby(["model_arm", "calibrated"], as_index=False)
        .agg(runs=("run_id", "nunique"), AUPRC_mean=("AUPRC", "mean"), AUPRC_std=("AUPRC", "std"), AUROC_mean=("AUROC", "mean"), Brier_mean=("Brier", "mean"), ECE_mean=("ECE", "mean"), NLL_mean=("NLL", "mean"), temperature_mean=("temperature", "mean"))
        .sort_values(["calibrated", "AUPRC_mean"], ascending=[False, False])
    )
    summary.to_csv(tables / "q1_neural_model_summary.csv", index=False)
    plt.figure(figsize=(9, 4))
    metrics[~metrics["calibrated"]].boxplot(column="AUPRC", by="model_arm", rot=20)
    plt.suptitle("")
    plt.title("Q1 neural model AUPRC over checkpointed fits")
    plt.ylabel("AUPRC")
    plt.tight_layout()
    plt.savefig(figs / "q1_neural_model_auprc.png", dpi=180)
    plt.close()
    plt.figure(figsize=(7, 4))
    plt.hist(stability["spearman"], bins=12)
    plt.title("Q1 neural ConceptFAN local-contribution stability")
    plt.xlabel("Spearman")
    plt.ylabel("checkpoint pairs")
    plt.tight_layout()
    plt.savefig(figs / "q1_neural_stability_spearman.png", dpi=180)
    plt.close()
    verifier = validate_neural_outputs(output, arrays, batch_size)
    (manifests / "q1_neural_verification.json").write_text(json.dumps(verifier, indent=2), encoding="utf-8")
    manifest = {
        "status": "Q1_NEURAL_FAILED_PILOT",
        "created_utc": now(),
        "code_commit": git_text(["rev-parse", "HEAD"]),
        "branch": git_text(["branch", "--show-current"]),
        "git_status_clean_at_run": git_text(["status", "--porcelain"]) == "",
        "n_runs": 30,
        "arms": ARMS,
        "input_contract": "[B,36,27] raw model_input; no oracle state features are passed to model forward",
        "concept_labels_scope": "generator states are used only as supervised concept targets",
        "verifier": "MANIFESTS/q1_neural_verification.json",
    }
    (manifests / "q1_neural_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "REPORTS" / "Q1_NEURAL_STATUS.md").write_text(f"# Q1 Neural Status\n\nStatus: `{manifest['status']}`\n\nVerifier: `{verifier['status']}`\n", encoding="utf-8")
    return manifest


def package_output(output: Path, zip_output_dir: Path, *, final_name: str | None = None) -> Path:
    zip_output_dir.mkdir(parents=True, exist_ok=True)
    name = final_name or f"Med_CircuitBench_Q1_NEURAL_EMPIRICAL_EXTENSION_{git_text(['rev-parse', '--short', 'HEAD'])}.zip"
    zip_path = zip_output_dir / name
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zf.write(path, f"Q1_NEURAL_EMPIRICAL_EXTENSION/{path.relative_to(output).as_posix()}")
        for src, arc in [
            (ROOT / "scripts" / "medical" / "v3_1" / "run_q1_neural_empirical_extension.py", "SOURCE/scripts/medical/v3_1/run_q1_neural_empirical_extension.py"),
            (ROOT / "configs" / "medical" / "v3" / "full.yaml", "CONFIGS/medical/v3/full.yaml"),
            (ROOT / "AGENTS.md", "MANIFESTS/AGENTS.md"),
        ]:
            if src.exists():
                zf.write(src, arc)
    sidecar = zip_path.with_suffix(zip_path.suffix + ".sha256")
    sidecar.write_text(f"{sha256_file(zip_path)}  {zip_path.name}\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
    if bad is not None:
        raise RuntimeError(f"zip integrity failed at {bad}")
    return zip_path


def run_neural(cfg: dict, output: Path, *, n_runs: int, n_samples: int, epochs: int, batch_size: int, package: bool, zip_output_dir: Path) -> dict:
    if output.exists():
        shutil.rmtree(output)
    (output / "CHECKPOINTS").mkdir(parents=True, exist_ok=True)
    arrays = build_arrays(make_q1_splits(cfg, n_samples, seed=6262))
    for meta in run_grid(n_runs):
        for arm in ARMS:
            train_one(arm, arrays, cfg, meta, output, epochs=epochs, batch_size=batch_size)
    manifest = write_outputs(output, arrays, cfg, batch_size)
    zip_path = package_output(output, zip_output_dir) if package else None
    report = {
        "status": manifest["status"],
        "output": str(output),
        "zip": str(zip_path) if zip_path else None,
        "zip_sha256": sha256_file(zip_path) if zip_path else None,
        "verifier": str(output / "MANIFESTS" / "q1_neural_verification.json"),
    }
    (output / "MANIFESTS" / "q1_neural_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medical/v3/full.yaml")
    parser.add_argument("--output", default="artifacts/medical/q1_neural_empirical_extension")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--zip-output-dir", default="artifacts/medical")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    report = run_neural(
        cfg,
        ROOT / args.output,
        n_runs=args.runs,
        n_samples=args.n_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        package=args.package,
        zip_output_dir=ROOT / args.zip_output_dir,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "Q1_NEURAL_FAILED_PILOT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
