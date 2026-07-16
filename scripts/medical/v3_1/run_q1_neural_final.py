#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, r2_score, roc_auc_score
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
from med_circuitbench.models.transformer import ClinicalTransformer, TransformerConfig
from scripts.medical.v2.run_v2_program import make_episodes, split_frame, subset_cols
from scripts.medical.v3.run_fan_iteration import ConceptScaler
from scripts.medical.v3.run_research_program import prepare_arrays
from scripts.medical.v3.run_predicted_fan_strict import (
    initialize_memberships_from_predicted,
    make_predicted_model,
    train_concept_stage,
    train_fan_head,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARMS = ["ConceptFAN-NoAlpha", "NoFuzzy", "TemporalCEM", "TemporalPCBM", "PlainTransformer"]
CONCEPT_NAMES = ["I", "R", "V", "O", "S"]
GRID_INIT_SEEDS = list(range(3101, 3111))
GRID_ORDER_SEEDS = list(range(4101, 4104))
SANITY_SEEDS = [42, 43, 44]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu().contiguous() for k, v in sorted(state.items())}, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0 - 1e-8)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
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


def calibration_slope_intercept(y: np.ndarray, logits: np.ndarray) -> tuple[float, float]:
    x = np.asarray(logits, dtype=float).reshape(-1, 1)
    y = np.asarray(y).astype(int)
    if np.unique(y).size < 2 or float(np.std(x)) <= 1e-12:
        return float("nan"), float("nan")
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    clf.fit(x, y)
    return float(clf.coef_[0, 0]), float(clf.intercept_[0])


def best_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    grid = np.r_[np.linspace(0.50, 1.0, 11), np.linspace(1.1, 5.0, 40)]
    losses = [log_loss(y, sigmoid_np(logits / t), labels=[0, 1]) for t in grid]
    return float(grid[int(np.argmin(losses))])


def config_contract(cfg: dict) -> dict[str, bool]:
    return {
        "input_dim_27": int(cfg["model"]["input_dim"]) == 27,
        "observed_window_36": int(cfg["dataset"]["observed_window"]) == 36,
        "latent_dim_128": int(cfg["model"]["latent_dim"]) == 128,
        "layers_4": int(cfg["model"]["layers"]) == 4 and int(cfg["model"]["transformer_layers"]) == 4,
        "heads_4": int(cfg["model"]["heads"]) == 4 and int(cfg["model"]["transformer_heads"]) == 4,
        "ffn_512": int(cfg["model"]["d_ffn"]) == 512 and int(cfg["model"]["transformer_ffn"]) == 512,
        "dropout_0_1": abs(float(cfg["model"].get("dropout", 0.1)) - 0.1) < 1e-12,
        "max_epochs_50": int(cfg["training"]["max_epochs"]) == 50,
    }


def assert_config_contract(cfg: dict) -> None:
    checks = config_contract(cfg)
    if not all(checks.values()):
        raise ValueError("Canonical config contract failed: " + json.dumps(checks, sort_keys=True))


@dataclass
class Arrays:
    x_fit: np.ndarray
    y_fit: np.ndarray
    c_fit: np.ndarray
    x_cal: np.ndarray
    y_cal: np.ndarray
    c_cal: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    c_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    c_test: np.ndarray
    val_ids: np.ndarray
    test_ids: np.ndarray
    scaler: ConceptScaler


def build_arrays(cfg: dict, generator_seed: int) -> Arrays:
    frame = make_episodes(generator_seed, cfg, "clean")
    raw = prepare_arrays(split_frame(frame, generator_seed), subset_cols("full_input"), include_test=True)
    cut = int(0.80 * len(raw["x_train"]))
    scaler = ConceptScaler.fit(raw["c_train_seq"][:cut], "minmax_train")
    return Arrays(
        x_fit=raw["x_train"][:cut],
        y_fit=raw["y_train"][:cut],
        c_fit=scaler.transform(raw["c_train_seq"][:cut]),
        x_cal=raw["x_train"][cut:],
        y_cal=raw["y_train"][cut:],
        c_cal=scaler.transform(raw["c_train_seq"][cut:]),
        x_val=raw["x_val"],
        y_val=raw["y_val"],
        c_val=scaler.transform(raw["c_val_seq"]),
        x_test=raw["x_test"],
        y_test=raw["y_test"],
        c_test=scaler.transform(raw["c_test_seq"]),
        val_ids=raw["val_ids"],
        test_ids=raw["test_ids"],
        scaler=scaler,
    )


def make_loader(x: np.ndarray, y: np.ndarray, c: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)), torch.from_numpy(c.astype(np.float32))),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(int(seed)),
    )


def model_cfg(cfg: dict, n_memberships: int = 3) -> dict:
    return {
        "input_dim": int(cfg["model"]["input_dim"]),
        "sequence_length": int(cfg["dataset"]["observed_window"]),
        "latent_dim": int(cfg["model"]["latent_dim"]),
        "n_concepts": 5,
        "heads": int(cfg["model"]["heads"]),
        "ffn": int(cfg["model"]["d_ffn"]),
        "layers": int(cfg["model"]["layers"]),
        "dropout": float(cfg["model"].get("dropout", 0.1)),
        "n_memberships": int(n_memberships),
        "membership": "gaussian",
        "alpha_mode": "no_alpha",
    }


class NoFuzzyModel(nn.Module):
    def __init__(self, mcfg: dict):
        super().__init__()
        self.encoder = SequenceTemporalEncoder(mcfg["input_dim"], mcfg["latent_dim"], mcfg["sequence_length"], mcfg["dropout"], mcfg["layers"], mcfg["heads"], mcfg["ffn"])
        self.projector = TokenConceptProjector(mcfg["latent_dim"], mcfg["n_concepts"])
        self.temporal_aggregator = TemporalConceptAggregator(mcfg["n_concepts"], mcfg["sequence_length"])
        self.head = nn.Linear(mcfg["n_concepts"], 1)

    def freeze_concept_path(self) -> None:
        for module in [self.encoder, self.projector, self.temporal_aggregator]:
            for p in module.parameters():
                p.requires_grad = False

    def forward_from_summaries(self, h: torch.Tensor, trajectories: torch.Tensor, summaries: torch.Tensor, temporal_weights: torch.Tensor, concept_mask: torch.Tensor | None = None) -> dict:
        local = summaries * self.head.weight.view(1, -1)
        if concept_mask is not None:
            local = local * concept_mask.to(local.dtype)
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

    def forward(self, x: torch.Tensor, concept_mask: torch.Tensor | None = None) -> dict:
        h = self.encoder(x)
        trajectories = self.projector(h)
        summaries, temporal_weights = self.temporal_aggregator(trajectories, "attention")
        return self.forward_from_summaries(h, trajectories, summaries, temporal_weights, concept_mask)


class TemporalCEMModel(nn.Module):
    def __init__(self, mcfg: dict, embedding_dim: int = 16):
        super().__init__()
        self.encoder = SequenceTemporalEncoder(mcfg["input_dim"], mcfg["latent_dim"], mcfg["sequence_length"], mcfg["dropout"], mcfg["layers"], mcfg["heads"], mcfg["ffn"])
        self.projector = TokenConceptProjector(mcfg["latent_dim"], mcfg["n_concepts"])
        self.temporal_aggregator = TemporalConceptAggregator(mcfg["n_concepts"], mcfg["sequence_length"])
        self.positive_embeddings = nn.Parameter(torch.empty(mcfg["n_concepts"], embedding_dim))
        self.negative_embeddings = nn.Parameter(torch.empty(mcfg["n_concepts"], embedding_dim))
        self.gate = nn.Sequential(nn.Linear(1, 8), nn.GELU(), nn.Linear(8, 1), nn.Sigmoid())
        self.classifier_weight = nn.Parameter(torch.empty(mcfg["n_concepts"], embedding_dim))
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.xavier_uniform_(self.positive_embeddings)
        nn.init.xavier_uniform_(self.negative_embeddings)
        nn.init.xavier_uniform_(self.classifier_weight)

    def freeze_concept_path(self) -> None:
        for module in [self.encoder, self.projector, self.temporal_aggregator]:
            for p in module.parameters():
                p.requires_grad = False

    def forward_from_summaries(self, h: torch.Tensor, trajectories: torch.Tensor, summaries: torch.Tensor, temporal_weights: torch.Tensor, concept_mask: torch.Tensor | None = None) -> dict:
        p = summaries.clamp(0.0, 1.0).unsqueeze(-1)
        emb = p * self.positive_embeddings.unsqueeze(0) + (1.0 - p) * self.negative_embeddings.unsqueeze(0)
        gates = self.gate(summaries.unsqueeze(-1))
        local = (gates * emb * self.classifier_weight.unsqueeze(0)).sum(dim=-1)
        if concept_mask is not None:
            local = local * concept_mask.to(local.dtype)
        logit = self.bias + local.sum(dim=-1)
        return {
            "logit": logit,
            "probability": torch.sigmoid(logit),
            "latent_sequence": h,
            "concept_trajectories": trajectories,
            "concept_summaries": summaries,
            "temporal_concept_weights": temporal_weights,
            "concept_probabilities": summaries,
            "concept_gates": gates.squeeze(-1),
            "positive_embeddings": self.positive_embeddings,
            "negative_embeddings": self.negative_embeddings,
            "local_contributions": local,
        }

    def forward(self, x: torch.Tensor, concept_mask: torch.Tensor | None = None) -> dict:
        h = self.encoder(x)
        trajectories = self.projector(h)
        summaries, temporal_weights = self.temporal_aggregator(trajectories, "attention")
        return self.forward_from_summaries(h, trajectories, summaries, temporal_weights, concept_mask)


class TemporalPCBMModel(nn.Module):
    def __init__(self, mcfg: dict):
        super().__init__()
        self.backbone = SequenceTemporalEncoder(mcfg["input_dim"], mcfg["latent_dim"], mcfg["sequence_length"], mcfg["dropout"], mcfg["layers"], mcfg["heads"], mcfg["ffn"])
        self.task_head = nn.Linear(mcfg["latent_dim"], 1)
        self.concept_bank = nn.Linear(mcfg["latent_dim"], mcfg["n_concepts"])
        self.downstream = nn.Linear(mcfg["n_concepts"], 1)

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).mean(dim=1)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, concept_mask: torch.Tensor | None = None) -> dict:
        latent = self.pooled(x)
        concepts = torch.sigmoid(self.concept_bank(latent))
        local = concepts * self.downstream.weight.view(1, -1)
        if concept_mask is not None:
            local = local * concept_mask.to(local.dtype)
        logit = self.downstream.bias.view(1) + local.sum(dim=-1)
        return {
            "logit": logit,
            "probability": torch.sigmoid(logit),
            "latent": latent,
            "concept_summaries": concepts,
            "concept_trajectories": concepts.unsqueeze(1).expand(-1, 36, -1),
            "concept_bank": concepts,
            "local_contributions": local,
        }

    def task_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.task_head(self.pooled(x)).squeeze(-1)


class PlainTransformerModel(nn.Module):
    def __init__(self, mcfg: dict):
        super().__init__()
        self.model = ClinicalTransformer(
            TransformerConfig(
                input_dim=mcfg["input_dim"],
                layers=mcfg["layers"],
                d_model=mcfg["latent_dim"],
                heads=mcfg["heads"],
                d_ffn=mcfg["ffn"],
                dropout=mcfg["dropout"],
                sequence_length=mcfg["sequence_length"],
            )
        )

    def forward(self, x: torch.Tensor) -> dict:
        return self.model(x)


def make_concept_fan(cfg: dict, prevalence: float, n_memberships: int = 3) -> MultiSetAdditiveTemporalConceptFANModel:
    fan_cfg = cfg.copy()
    fan_cfg["fan"] = dict(cfg.get("fan", {}))
    fan_cfg["fan"]["alpha_mode"] = "no_alpha"
    model = make_predicted_model(fan_cfg, "no_alpha", prevalence)
    if int(n_memberships) != 3:
        mcfg = model_cfg(cfg, n_memberships)
        model = MultiSetAdditiveTemporalConceptFANModel(
            input_dim=mcfg["input_dim"],
            sequence_length=mcfg["sequence_length"],
            latent_dim=mcfg["latent_dim"],
            n_concepts=5,
            n_memberships=int(n_memberships),
            membership="gaussian",
            oracle=False,
            temporal_mode="attention",
            dropout=mcfg["dropout"],
            encoder_layers=mcfg["layers"],
            encoder_heads=mcfg["heads"],
            encoder_ffn=mcfg["ffn"],
            alpha_mode="no_alpha",
        ).to(DEVICE)
        model.decision_head.initialize_bias_from_prevalence(prevalence)
    return model.to(DEVICE)


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
        "fuzzy_values": out.fuzzy_values,
    }


def copy_concept_path(source_state: dict[str, torch.Tensor], model: nn.Module, *, include_temporal: bool = True) -> None:
    state = model.state_dict()
    prefixes = ("encoder.", "projector.", "temporal_aggregator.") if include_temporal else ("encoder.", "projector.")
    for key in list(state):
        if key.startswith(prefixes) and key in source_state:
            state[key] = source_state[key].detach().clone().to(state[key].device)
    model.load_state_dict(state)


def eval_model(model: nn.Module, x: np.ndarray, y: np.ndarray, c: np.ndarray, batch_size: int) -> dict:
    model.eval()
    logits, probs, local, traj, summaries = [], [], [], [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(DEVICE)
            out = model_output_dict(model(xb))
            logits.append(out["logit"].detach().cpu().numpy())
            probs.append(out["probability"].detach().cpu().numpy())
            if "local_contributions" in out:
                local.append(out["local_contributions"].detach().cpu().numpy())
            if "concept_trajectories" in out:
                traj.append(out["concept_trajectories"].detach().cpu().numpy())
            if "concept_summaries" in out:
                summaries.append(out["concept_summaries"].detach().cpu().numpy())
    return {
        "target": y.astype(int),
        "logit": np.concatenate(logits),
        "probability": np.concatenate(probs),
        "local_contributions": np.concatenate(local, axis=0) if local else None,
        "concept_trajectories": np.concatenate(traj, axis=0) if traj else None,
        "concept_summaries": np.concatenate(summaries, axis=0) if summaries else None,
    }


def train_linear_head_model(model: nn.Module, arrays: Arrays, cfg: dict, run_meta: dict, trainable_filter, batch_size: int) -> tuple[list[dict], dict]:
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if trainable_filter(name):
            p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    train_loader = make_loader(arrays.x_fit, arrays.y_fit, arrays.c_fit, batch_size, True, run_meta["data_order_seed"])
    history, best, best_state, best_opt = [], -1.0, None, None
    for epoch in range(int(cfg["training"]["max_epochs"])):
        model.train()
        losses = []
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model_output_dict(model(xb))
            loss = F.binary_cross_entropy_with_logits(out["logit"], yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(cfg["fan"].get("gradient_clip", 1.0)))
            opt.step()
            losses.append(float(loss.item()))
        val = eval_model(model, arrays.x_val, arrays.y_val, arrays.c_val, batch_size)
        auprc = binary_metrics(arrays.y_val, val["probability"])["AUPRC"]
        history.append({"epoch": epoch + 1, "task_loss": float(np.mean(losses)), "validation_AUPRC": auprc})
        if auprc > best:
            best = auprc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_opt = opt.state_dict()
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, {"optimizer_state_dict": best_opt or opt.state_dict()}


def train_plain(model: PlainTransformerModel, arrays: Arrays, cfg: dict, run_meta: dict, batch_size: int) -> tuple[list[dict], dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    train_loader = make_loader(arrays.x_fit, arrays.y_fit, arrays.c_fit, batch_size, True, run_meta["data_order_seed"])
    history, best, best_state, best_opt = [], -1.0, None, None
    for epoch in range(int(cfg["training"]["max_epochs"])):
        model.train()
        losses = []
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = F.binary_cross_entropy_with_logits(out["logit"], yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val = eval_model(model, arrays.x_val, arrays.y_val, arrays.c_val, batch_size)
        auprc = binary_metrics(arrays.y_val, val["probability"])["AUPRC"]
        history.append({"epoch": epoch + 1, "task_loss": float(np.mean(losses)), "validation_AUPRC": auprc})
        if auprc > best:
            best = auprc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_opt = opt.state_dict()
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, {"optimizer_state_dict": best_opt or opt.state_dict()}


def train_cem(model: TemporalCEMModel, arrays: Arrays, cfg: dict, run_meta: dict, batch_size: int) -> tuple[list[dict], dict]:
    train_loader = make_loader(arrays.x_fit, arrays.y_fit, arrays.c_fit, batch_size, True, run_meta["data_order_seed"])
    val_loader = make_loader(arrays.x_val, arrays.y_val, arrays.c_val, batch_size, False, run_meta["data_order_seed"])
    opt_c = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.projector.parameters()), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    concept_history = []
    for epoch in range(int(cfg["training"]["concept_epochs"])):
        model.train()
        losses = []
        for xb, _, cb in train_loader:
            xb, cb = xb.to(DEVICE), cb.to(DEVICE)
            opt_c.zero_grad(set_to_none=True)
            pred = model.projector(model.encoder(xb))
            loss = F.huber_loss(pred, cb) + 0.2 * F.huber_loss(pred[:, 1:] - pred[:, :-1], cb[:, 1:] - cb[:, :-1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.encoder.parameters()) + list(model.projector.parameters()), 1.0)
            opt_c.step()
            losses.append(float(loss.item()))
        concept_history.append({"epoch": epoch + 1, "concept_loss": float(np.mean(losses))})
    model.freeze_concept_path()
    task_history, task_state = train_linear_head_model(model, arrays, cfg, run_meta, lambda n: not (n.startswith("encoder.") or n.startswith("projector.") or n.startswith("temporal_aggregator.")), batch_size)
    return concept_history + task_history, {"concept_optimizer_state_dict": opt_c.state_dict(), **task_state}


def train_pcbm(model: TemporalPCBMModel, arrays: Arrays, cfg: dict, run_meta: dict, batch_size: int) -> tuple[list[dict], dict]:
    train_loader = make_loader(arrays.x_fit, arrays.y_fit, arrays.c_fit, batch_size, True, run_meta["data_order_seed"])
    opt_task = torch.optim.AdamW(list(model.backbone.parameters()) + list(model.task_head.parameters()), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    history, best, best_backbone, best_task = [], -1.0, None, None
    for epoch in range(int(cfg["training"]["max_epochs"])):
        model.train()
        losses = []
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt_task.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model.task_forward(xb), yb)
            loss.backward()
            opt_task.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            logits = []
            for start in range(0, len(arrays.x_val), batch_size):
                logits.append(model.task_forward(torch.from_numpy(arrays.x_val[start:start + batch_size]).to(DEVICE)).cpu().numpy())
        auprc = binary_metrics(arrays.y_val, sigmoid_np(np.concatenate(logits)))["AUPRC"]
        history.append({"stage": "task_backbone", "epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_AUPRC": auprc})
        if auprc > best:
            best = auprc
            best_backbone = {k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()}
            best_task = {k: v.detach().cpu().clone() for k, v in model.task_head.state_dict().items()}
    model.backbone.load_state_dict(best_backbone)
    model.task_head.load_state_dict(best_task)
    model.freeze_backbone()
    opt_probe = torch.optim.AdamW(model.concept_bank.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    for epoch in range(int(cfg["training"]["concept_epochs"])):
        losses = []
        for xb, _, cb in train_loader:
            xb, cb = xb.to(DEVICE), cb.to(DEVICE)
            opt_probe.zero_grad(set_to_none=True)
            pred = torch.sigmoid(model.concept_bank(model.pooled(xb)))
            target = cb.mean(dim=1)
            loss = F.huber_loss(pred, target)
            loss.backward()
            opt_probe.step()
            losses.append(float(loss.item()))
        history.append({"stage": "concept_probes", "epoch": epoch + 1, "loss": float(np.mean(losses))})
    for p in model.concept_bank.parameters():
        p.requires_grad = False
    opt_down = torch.optim.AdamW(model.downstream.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    best, best_down = -1.0, None
    for epoch in range(int(cfg["training"]["max_epochs"])):
        losses = []
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt_down.zero_grad(set_to_none=True)
            out = model(xb)
            loss = F.binary_cross_entropy_with_logits(out["logit"], yb)
            loss.backward()
            opt_down.step()
            losses.append(float(loss.item()))
        val = eval_model(model, arrays.x_val, arrays.y_val, arrays.c_val, batch_size)
        auprc = binary_metrics(arrays.y_val, val["probability"])["AUPRC"]
        history.append({"stage": "downstream_no_residual", "epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_AUPRC": auprc})
        if auprc > best:
            best = auprc
            best_down = {k: v.detach().cpu().clone() for k, v in model.downstream.state_dict().items()}
    model.downstream.load_state_dict(best_down)
    return history, {"task_optimizer_state_dict": opt_task.state_dict(), "probe_optimizer_state_dict": opt_probe.state_dict(), "downstream_optimizer_state_dict": opt_down.state_dict()}


def train_shared_concept_reference(arrays: Arrays, cfg: dict, run_meta: dict, batch_size: int) -> tuple[dict[str, torch.Tensor], list[dict], dict]:
    train_loader = make_loader(arrays.x_fit, arrays.y_fit, arrays.c_fit, batch_size, True, run_meta["data_order_seed"])
    val_loader = make_loader(arrays.x_val, arrays.y_val, arrays.c_val, batch_size, False, run_meta["data_order_seed"])
    model = make_concept_fan(cfg, float(arrays.y_fit.mean()), 3)
    concept_history, concept_met = train_concept_stage(model, cfg, train_loader, val_loader)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.startswith(("encoder.", "projector.", "temporal_aggregator."))}
    return state, concept_history, {k: v for k, v in concept_met.items() if k != "per_concept"}


def train_one_arm(arm: str, arrays: Arrays, cfg: dict, run_meta: dict, output: Path, shared_state: dict[str, torch.Tensor] | None, shared_history: list[dict] | None, shared_metrics: dict | None, batch_size: int) -> dict:
    set_all_seeds(run_meta["initialization_seed"])
    mcfg = model_cfg(cfg)
    optimizer_state = {}
    history = []
    concept_metrics = shared_metrics or {}
    if arm == "ConceptFAN-NoAlpha":
        model = make_concept_fan(cfg, float(arrays.y_fit.mean()), 3)
        copy_concept_path(shared_state, model, include_temporal=True)
        train_loader = make_loader(arrays.x_fit, arrays.y_fit, arrays.c_fit, batch_size, True, run_meta["data_order_seed"])
        val_loader = make_loader(arrays.x_val, arrays.y_val, arrays.c_val, batch_size, False, run_meta["data_order_seed"])
        initialize_memberships_from_predicted(model, train_loader)
        history = list(shared_history or []) + train_fan_head(model, cfg, train_loader, val_loader)
    elif arm == "NoFuzzy":
        model = NoFuzzyModel(mcfg).to(DEVICE)
        copy_concept_path(shared_state, model, include_temporal=True)
        history, optimizer_state = train_linear_head_model(model, arrays, cfg, run_meta, lambda n: n.startswith("head."), batch_size)
    elif arm == "TemporalCEM":
        model = TemporalCEMModel(mcfg).to(DEVICE)
        history, optimizer_state = train_cem(model, arrays, cfg, run_meta, batch_size)
    elif arm == "TemporalPCBM":
        model = TemporalPCBMModel(mcfg).to(DEVICE)
        history, optimizer_state = train_pcbm(model, arrays, cfg, run_meta, batch_size)
    elif arm == "PlainTransformer":
        model = PlainTransformerModel(mcfg).to(DEVICE)
        history, optimizer_state = train_plain(model, arrays, cfg, run_meta, batch_size)
    else:
        raise ValueError(arm)

    cal = eval_model(model, arrays.x_cal, arrays.y_cal, arrays.c_cal, batch_size)
    temperature = best_temperature(cal["logit"], arrays.y_cal)
    val = eval_model(model, arrays.x_val, arrays.y_val, arrays.c_val, batch_size)
    test = eval_model(model, arrays.x_test, arrays.y_test, arrays.c_test, batch_size)
    raw = binary_metrics(arrays.y_val, val["probability"])
    calibrated = binary_metrics(arrays.y_val, sigmoid_np(val["logit"] / temperature))
    confirm = binary_metrics(arrays.y_test, sigmoid_np(test["logit"] / temperature))
    concept_mae = float(np.mean(np.abs(val["concept_trajectories"] - arrays.c_val))) if val["concept_trajectories"] is not None and arm != "PlainTransformer" else float("nan")
    concept_r2 = float(r2_score(arrays.c_val.reshape(-1, 5), val["concept_trajectories"].reshape(-1, 5), multioutput="variance_weighted")) if val["concept_trajectories"] is not None and arm != "PlainTransformer" else float("nan")
    slope, intercept = calibration_slope_intercept(arrays.y_val, val["logit"] / temperature)
    ckpt_dir = output / "CHECKPOINTS" / arm / f"run_{run_meta['run_id']:02d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    param_sha = state_dict_sha256(model.state_dict())
    ckpt = {
        "arm": arm,
        "run": run_meta,
        "model_config": mcfg,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_training_state": optimizer_state,
        "epoch_history": history,
        "concept_metrics": concept_metrics,
        "temperature": temperature,
        "parameter_sha256": param_sha,
        "canonical_config": config_contract(cfg),
    }
    ckpt_path = ckpt_dir / "checkpoint.pt"
    torch.save(ckpt, ckpt_path)
    row = {
        **run_meta,
        "model_arm": arm,
        "checkpoint": str(ckpt_path.relative_to(output)),
        "parameter_sha256": param_sha,
        "train_epochs": int(cfg["training"]["max_epochs"]),
        "temperature": temperature,
        "raw_AUPRC": raw["AUPRC"],
        "raw_AUROC": raw["AUROC"],
        "calibrated_AUPRC": calibrated["AUPRC"],
        "calibrated_AUROC": calibrated["AUROC"],
        "confirmatory_AUPRC": confirm["AUPRC"],
        "confirmatory_AUROC": confirm["AUROC"],
        "Brier": calibrated["Brier"],
        "ECE": calibrated["ECE"],
        "NLL": calibrated["NLL"],
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "concept_MAE": concept_mae,
        "concept_R2": concept_r2,
        **{f"concept_{k}": float(v) for k, v in concept_metrics.items() if isinstance(v, (int, float, np.floating))},
    }
    (ckpt_dir / "manifest.json").write_text(json.dumps({k: v for k, v in ckpt.items() if k != "state_dict"}, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(history).to_csv(ckpt_dir / "epoch_history.csv", index=False)
    return row


def grid_runs(n_runs: int = 30) -> list[dict]:
    rows = []
    for init_seed in GRID_INIT_SEEDS:
        for order_seed in GRID_ORDER_SEEDS:
            rows.append({"run_id": len(rows) + 1, "initialization_seed": init_seed, "data_order_seed": order_seed})
    return rows[:n_runs]


def load_checkpoint(path: Path) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    mcfg = ckpt["model_config"]
    arm = ckpt["arm"]
    if arm == "ConceptFAN-NoAlpha":
        model = make_concept_fan({"model": {"input_dim": mcfg["input_dim"], "latent_dim": mcfg["latent_dim"], "layers": mcfg["layers"], "heads": mcfg["heads"], "d_ffn": mcfg["ffn"], "dropout": mcfg["dropout"]}, "dataset": {"observed_window": mcfg["sequence_length"]}, "fan": {"alpha_mode": "no_alpha"}}, 0.2, mcfg["n_memberships"])
    elif arm == "NoFuzzy":
        model = NoFuzzyModel(mcfg).to(DEVICE)
    elif arm == "TemporalCEM":
        model = TemporalCEMModel(mcfg).to(DEVICE)
    elif arm == "TemporalPCBM":
        model = TemporalPCBMModel(mcfg).to(DEVICE)
    elif arm == "PlainTransformer":
        model = PlainTransformerModel(mcfg).to(DEVICE)
    else:
        raise ValueError(arm)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def checkpoint_paths(output: Path) -> list[Path]:
    return sorted((output / "CHECKPOINTS").glob("*/*/checkpoint.pt"))


def local_contribution_tables(output: Path, arrays: Arrays, batch_size: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    cache: dict[str, np.ndarray] = {}
    for ckpt_path in checkpoint_paths(output):
        model, ckpt = load_checkpoint(ckpt_path)
        ev = eval_model(model, arrays.x_val, arrays.y_val, arrays.c_val, batch_size)
        if ev["local_contributions"] is None:
            continue
        rel = str(ckpt_path.relative_to(output))
        local = ev["local_contributions"]
        cache[rel] = local
        abs_rank = np.argsort(-np.abs(local), axis=1)
        rank_lookup = np.empty_like(abs_rank)
        for i in range(abs_rank.shape[0]):
            rank_lookup[i, abs_rank[i]] = np.arange(1, 6)
        for episode_pos, episode_id in enumerate(arrays.val_ids):
            for ci, concept in enumerate(CONCEPT_NAMES):
                signed = float(local[episode_pos, ci])
                rows.append(
                    {
                        **ckpt["run"],
                        "model_arm": ckpt["arm"],
                        "checkpoint": rel,
                        "episode_id": int(episode_id),
                        "episode_pos": int(episode_pos),
                        "concept": concept,
                        "concept_index": ci,
                        "signed_contribution": signed,
                        "absolute_contribution": abs(signed),
                        "rank": int(rank_lookup[episode_pos, ci]),
                        "sign": int(np.sign(signed)),
                        "original_logit": float(ev["logit"][episode_pos]),
                        "original_probability": float(ev["probability"][episode_pos]),
                    }
                )
    return pd.DataFrame(rows), cache


def pairwise_stability(local_df: pd.DataFrame) -> pd.DataFrame:
    fan = local_df[local_df["model_arm"].eq("ConceptFAN-NoAlpha")]
    rows = []
    checkpoints = sorted(fan["checkpoint"].unique())
    grouped = {ck: part.sort_values(["episode_pos", "concept_index"]) for ck, part in fan.groupby("checkpoint")}
    for i in range(len(checkpoints)):
        for j in range(i + 1, len(checkpoints)):
            a = grouped[checkpoints[i]]["signed_contribution"].to_numpy().reshape(-1, 5)
            b = grouped[checkpoints[j]]["signed_contribution"].to_numpy().reshape(-1, 5)
            for ep in range(a.shape[0]):
                aa, bb = a[ep], b[ep]
                top1 = int(np.argmax(np.abs(aa)) == np.argmax(np.abs(bb)))
                top3_a = set(np.argsort(-np.abs(aa))[:3])
                top3_b = set(np.argsort(-np.abs(bb))[:3])
                rows.append(
                    {
                        "checkpoint_a": checkpoints[i],
                        "checkpoint_b": checkpoints[j],
                        "episode_pos": ep,
                        "spearman": float(stats.spearmanr(aa, bb).statistic),
                        "kendall": float(stats.kendalltau(aa, bb).statistic),
                        "top1_agreement": top1,
                        "top3_jaccard": float(len(top3_a & top3_b) / len(top3_a | top3_b)),
                        "sign_agreement": float(np.mean(np.sign(aa) == np.sign(bb))),
                    }
                )
    return pd.DataFrame(rows)


def exhaustive_sufficient_subsets(output: Path, local_df: pd.DataFrame, arrays: Arrays) -> pd.DataFrame:
    concept_df = local_df[local_df["model_arm"].isin(["ConceptFAN-NoAlpha", "NoFuzzy", "TemporalCEM", "TemporalPCBM"])]
    rows = []
    thresholds = [(0.01, 0.10), (0.02, 0.20), (0.05, 0.50)]
    for (arm, ckpt), part in concept_df.groupby(["model_arm", "checkpoint"]):
        wide = part.pivot_table(index=["episode_pos", "episode_id", "original_logit", "original_probability"], columns="concept_index", values="signed_contribution").reset_index()
        contrib = wide[[0, 1, 2, 3, 4]].to_numpy(float)
        full_logit = wide["original_logit"].to_numpy(float)
        full_prob = wide["original_probability"].to_numpy(float)
        bias = full_logit - contrib.sum(axis=1)
        abs_order = np.argsort(-np.abs(contrib), axis=1)
        for mask_id in range(32):
            mask = np.asarray([(mask_id >> i) & 1 for i in range(5)], dtype=float)
            k = int(mask.sum())
            logit = bias + (contrib * mask.reshape(1, 5)).sum(axis=1)
            prob = sigmoid_np(logit)
            for prob_thr, logit_thr in thresholds:
                sufficient = (np.abs(prob - full_prob) <= prob_thr) | (np.abs(logit - full_logit) <= logit_thr)
                rows.append(
                    {
                        "model_arm": arm,
                        "checkpoint": ckpt,
                        "subset_mask_int": mask_id,
                        "M": k,
                        "control": "exhaustive",
                        "probability_threshold": prob_thr,
                        "logit_threshold": logit_thr,
                        "sufficient_fraction": float(np.mean(sufficient)),
                        "probability_mae": float(np.mean(np.abs(prob - full_prob))),
                        "logit_mae": float(np.mean(np.abs(logit - full_logit))),
                    }
                )
        rng = np.random.default_rng(20260716)
        for m in range(1, 6):
            top_masks, shuffled_masks, random_masks, mag_masks = [], [], [], []
            for ep in range(contrib.shape[0]):
                top = abs_order[ep, :m]
                shuffled = rng.permutation(abs_order[ep])[:m]
                random = rng.choice(5, size=m, replace=False)
                target_mag = float(np.abs(contrib[ep, top]).sum())
                best_combo = min(
                    [np.asarray(c) for c in __import__("itertools").combinations(range(5), m)],
                    key=lambda c: abs(float(np.abs(contrib[ep, list(c)]).sum()) - target_mag),
                )
                for target, store in [(top, top_masks), (shuffled, shuffled_masks), (random, random_masks), (best_combo, mag_masks)]:
                    mask = np.zeros(5, dtype=float)
                    mask[list(target)] = 1.0
                    store.append(mask)
            for name, masks in [("top_ranking", top_masks), ("shuffled_ranking", shuffled_masks), ("random_same_size", random_masks), ("magnitude_matched", mag_masks)]:
                mm = np.stack(masks)
                logit = bias + (contrib * mm).sum(axis=1)
                prob = sigmoid_np(logit)
                rows.append(
                    {
                        "model_arm": arm,
                        "checkpoint": ckpt,
                        "subset_mask_int": -1,
                        "M": m,
                        "control": name,
                        "probability_threshold": 0.02,
                        "logit_threshold": 0.20,
                        "sufficient_fraction": float(np.mean((np.abs(prob - full_prob) <= 0.02) | (np.abs(logit - full_logit) <= 0.20))),
                        "probability_mae": float(np.mean(np.abs(prob - full_prob))),
                        "logit_mae": float(np.mean(np.abs(logit - full_logit))),
                    }
                )
    return pd.DataFrame(rows)


def flatten_concepts(seq: np.ndarray) -> np.ndarray:
    last = seq[:, -1, :]
    mean = seq.mean(axis=1)
    maxv = seq.max(axis=1)
    return np.concatenate([last, mean, maxv], axis=1)


def leakage_audit(output: Path, arrays: Arrays, batch_size: int) -> pd.DataFrame:
    rows = []
    fan_paths = sorted((output / "CHECKPOINTS" / "ConceptFAN-NoAlpha").glob("run_*/checkpoint.pt"))
    for ckpt_path in fan_paths:
        model, ckpt = load_checkpoint(ckpt_path)
        cal = eval_model(model, arrays.x_cal, arrays.y_cal, arrays.c_cal, batch_size)
        hold = eval_model(model, arrays.x_test, arrays.y_test, arrays.c_test, batch_size)
        train_true = flatten_concepts(arrays.c_cal)
        hold_true = flatten_concepts(arrays.c_test)
        train_pred = flatten_concepts(cal["concept_trajectories"])
        hold_pred = flatten_concepts(hold["concept_trajectories"])
        train_res = train_pred - train_true
        hold_res = hold_pred - hold_true
        rng = np.random.default_rng(ckpt["run"]["initialization_seed"])
        perm = hold_res.copy()
        rng.shuffle(perm, axis=0)
        noise = rng.normal(0.0, np.std(train_res, axis=0, keepdims=True) + 1e-8, size=hold_res.shape)
        specs = {
            "true_concepts": (train_true, hold_true),
            "residuals": (train_res, hold_res),
            "true_concepts_plus_residuals": (np.concatenate([train_true, train_res], axis=1), np.concatenate([hold_true, hold_res], axis=1)),
            "permuted_residuals": (train_res, perm),
            "matched_random_noise": (train_res, noise),
        }
        for name, (xtr, xho) in specs.items():
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(xtr, arrays.y_cal.astype(int))
            prob = clf.predict_proba(xho)[:, 1]
            rows.append({**ckpt["run"], "checkpoint": str(ckpt_path.relative_to(output)), "feature_set": name, **binary_metrics(arrays.y_test, prob)})
    return pd.DataFrame(rows)


def recompute_delta_time(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.float32)
    last_seen = np.repeat(-10000, mask.shape[-1]).astype(int)
    for t in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[t, j] == 1:
                last_seen[j] = t
                out[t, j] = 0.0
            else:
                delta = 36 if last_seen[j] < 0 else min(t - last_seen[j], 36)
                out[t, j] = np.log1p(delta)
    return out


def perturb_raw(x_norm: np.ndarray, scenario: str, level: float, rng: np.random.Generator) -> np.ndarray:
    out = x_norm.copy()
    if scenario == "gaussian_noise":
        out[:, :, :8] += rng.normal(0.0, level, out[:, :, :8].shape).astype(np.float32)
    elif scenario == "mcar":
        miss = rng.random(out[:, :, :8].shape) < level
        out[:, :, :8][miss] = 0.0
        out[:, :, 8:16][miss] = 0.0
        for i in range(out.shape[0]):
            out[i, :, 16:24] = recompute_delta_time((out[i, :, 8:16] > 0).astype(np.float32))
    elif scenario == "block_missing":
        block = max(1, int(round(out.shape[1] * level)))
        start = max(0, (out.shape[1] - block) // 2)
        out[:, start : start + block, :8] = 0.0
        out[:, start : start + block, 8:16] = 0.0
        for i in range(out.shape[0]):
            out[i, :, 16:24] = recompute_delta_time((out[i, :, 8:16] > 0).astype(np.float32))
    elif scenario == "outliers_spikes":
        spike = rng.random(out[:, :, :8].shape) < level
        out[:, :, :8][spike] += rng.normal(0.0, 6.0, int(spike.sum())).astype(np.float32)
    elif scenario == "sensor_bias":
        out[:, :, :8] += level
    else:
        raise ValueError(scenario)
    return out.astype(np.float32)


def robustness(output: Path, arrays: Arrays, cfg: dict, batch_size: int) -> pd.DataFrame:
    rows = []
    shift_cfg = json.loads(json.dumps(cfg, default=str))
    shift_cfg["dataset"]["infection_impulse_strength"] = float(cfg["dataset"]["infection_impulse_strength"]) * 1.15
    shift_cfg["dataset"]["target_threshold"] = float(cfg["dataset"]["target_threshold"]) * 1.05
    shifted = build_arrays(shift_cfg, 9090)
    scenarios = {
        "gaussian_noise": [0.02, 0.05],
        "mcar": [0.05, 0.15],
        "block_missing": [0.10, 0.20],
        "outliers_spikes": [0.01, 0.03],
        "sensor_bias": [0.05, 0.10],
    }
    for ckpt_path in checkpoint_paths(output):
        model, ckpt = load_checkpoint(ckpt_path)
        rng = np.random.default_rng(ckpt["run"]["initialization_seed"] + ckpt["run"]["data_order_seed"])
        base = eval_model(model, arrays.x_val, arrays.y_val, arrays.c_val, batch_size)
        base_prob = sigmoid_np(base["logit"] / ckpt["temperature"])
        for scenario, levels in scenarios.items():
            for level in levels:
                xp = perturb_raw(arrays.x_val, scenario, level, rng)
                ev = eval_model(model, xp, arrays.y_val, arrays.c_val, batch_size)
                prob = sigmoid_np(ev["logit"] / ckpt["temperature"])
                met = binary_metrics(arrays.y_val, prob)
                rows.append(
                    {
                        **ckpt["run"],
                        "model_arm": ckpt["arm"],
                        "checkpoint": str(ckpt_path.relative_to(output)),
                        "scenario": scenario,
                        "level": level,
                        **met,
                        "probability_MAE_vs_clean": float(np.mean(np.abs(prob - base_prob))),
                        "concept_MAE": float(np.mean(np.abs(ev["concept_trajectories"] - arrays.c_val))) if ev["concept_trajectories"] is not None and ckpt["arm"] != "PlainTransformer" else float("nan"),
                        "explanation_degradation": float(np.mean(np.abs(ev["local_contributions"] - base["local_contributions"]))) if ev["local_contributions"] is not None and base["local_contributions"] is not None else float("nan"),
                    }
                )
        ev = eval_model(model, shifted.x_val, shifted.y_val, shifted.c_val, batch_size)
        rows.append({**ckpt["run"], "model_arm": ckpt["arm"], "checkpoint": str(ckpt_path.relative_to(output)), "scenario": "generator_parameter_shift", "level": "impulse_x1.15_threshold_x1.05", **binary_metrics(shifted.y_val, sigmoid_np(ev["logit"] / ckpt["temperature"]))})
    return pd.DataFrame(rows)


def m_sensitivity(arrays: Arrays, cfg: dict, output: Path, batch_size: int) -> pd.DataFrame:
    rows = []
    for seed in SANITY_SEEDS:
        run_meta = {"run_id": seed, "initialization_seed": seed, "data_order_seed": seed}
        shared_state, concept_history, concept_metrics = train_shared_concept_reference(arrays, cfg, run_meta, batch_size)
        train_loader = make_loader(arrays.x_fit, arrays.y_fit, arrays.c_fit, batch_size, True, seed)
        val_loader = make_loader(arrays.x_val, arrays.y_val, arrays.c_val, batch_size, False, seed)
        for m in [2, 3, 4, 5]:
            set_all_seeds(seed * 100 + m)
            model = make_concept_fan(cfg, float(arrays.y_fit.mean()), m)
            copy_concept_path(shared_state, model, include_temporal=True)
            initialize_memberships_from_predicted(model, train_loader)
            hist = train_fan_head(model, cfg, train_loader, val_loader)
            ev = eval_model(model, arrays.x_val, arrays.y_val, arrays.c_val, batch_size)
            rows.append({"seed": seed, "n_memberships": m, "epochs": len(hist), **binary_metrics(arrays.y_val, ev["probability"]), **{f"concept_{k}": float(v) for k, v in concept_metrics.items() if isinstance(v, (int, float, np.floating))}})
    return pd.DataFrame(rows)


def run_sanity_gate(cfg: dict, output: Path, batch_size: int) -> pd.DataFrame:
    rows = []
    sanity_dir = output / "SANITY"
    for seed in SANITY_SEEDS:
        arrays = build_arrays(cfg, seed)
        meta = {"run_id": seed, "initialization_seed": seed, "data_order_seed": seed}
        shared, shared_hist, shared_met = train_shared_concept_reference(arrays, cfg, meta, batch_size)
        for arm in ARMS:
            rows.append(train_one_arm(arm, arrays, cfg, meta, sanity_dir, shared, shared_hist, shared_met, batch_size))
    df = pd.DataFrame(rows)
    fan = df[df["model_arm"].eq("ConceptFAN-NoAlpha")]
    plain = df[df["model_arm"].eq("PlainTransformer")]
    passed = bool(fan["calibrated_AUPRC"].mean() >= 0.79 and fan["calibrated_AUPRC"].min() >= 0.79 and plain["calibrated_AUPRC"].mean() >= 0.79)
    df.attrs["status"] = "MODEL_ARM_SANITY_PASS" if passed else "MODEL_ARM_SANITY_FAIL"
    return df


def final_tables(output: Path, arrays: Arrays, cfg: dict, batch_size: int) -> dict:
    tables = output / "TABLES"
    figures = output / "FIGURES"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    reliability_rows = []
    for ckpt_path in checkpoint_paths(output):
        model, ckpt = load_checkpoint(ckpt_path)
        for split_name, x, y, c in [("validation", arrays.x_val, arrays.y_val, arrays.c_val), ("confirmatory_test", arrays.x_test, arrays.y_test, arrays.c_test)]:
            ev = eval_model(model, x, y, c, batch_size)
            for calibrated, prob in [(False, ev["probability"]), (True, sigmoid_np(ev["logit"] / ckpt["temperature"]))]:
                slope, intercept = calibration_slope_intercept(y, ev["logit"] / (ckpt["temperature"] if calibrated else 1.0))
                metric_rows.append({**ckpt["run"], "model_arm": ckpt["arm"], "checkpoint": str(ckpt_path.relative_to(output)), "split": split_name, "calibrated": calibrated, "temperature": ckpt["temperature"] if calibrated else 1.0, "parameter_sha256": ckpt["parameter_sha256"], **binary_metrics(y, prob), "calibration_slope": slope, "calibration_intercept": intercept})
                bins = np.linspace(0, 1, 11)
                for lo, hi in zip(bins[:-1], bins[1:]):
                    mask = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
                    if mask.any():
                        reliability_rows.append({**ckpt["run"], "model_arm": ckpt["arm"], "split": split_name, "calibrated": calibrated, "bin_low": lo, "bin_high": hi, "count": int(mask.sum()), "mean_probability": float(prob[mask].mean()), "empirical_rate": float(y[mask].mean())})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(tables / "q1_neural_predictive_calibration_metrics.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(tables / "q1_neural_reliability_curves.csv", index=False)
    local_df, _ = local_contribution_tables(output, arrays, batch_size)
    local_df.to_parquet(tables / "q1_neural_episode_local_contributions.parquet", index=False)
    stability = pairwise_stability(local_df)
    stability.to_parquet(tables / "q1_neural_episode_pairwise_stability.parquet", index=False)
    suff = exhaustive_sufficient_subsets(output, local_df, arrays)
    suff.to_parquet(tables / "q1_neural_exhaustive_32_mask_sufficiency.parquet", index=False)
    leak = leakage_audit(output, arrays, batch_size)
    leak.to_csv(tables / "q1_neural_heldout_leakage_audit.csv", index=False)
    rob = robustness(output, arrays, cfg, batch_size)
    rob.to_csv(tables / "q1_neural_raw_input_robustness.csv", index=False)
    ms = m_sensitivity(arrays, cfg, output, batch_size)
    ms.to_csv(tables / "q1_neural_membership_sensitivity.csv", index=False)
    summary = metrics[(metrics["split"] == "validation") & (metrics["calibrated"] == True)].groupby("model_arm", as_index=False).agg(runs=("run_id", "nunique"), AUPRC_mean=("AUPRC", "mean"), AUPRC_std=("AUPRC", "std"), AUROC_mean=("AUROC", "mean"), Brier_mean=("Brier", "mean"), ECE_mean=("ECE", "mean"), NLL_mean=("NLL", "mean"))
    summary.to_csv(tables / "q1_neural_article_ready_model_summary.csv", index=False)
    plt.figure(figsize=(9, 4))
    metrics[(metrics["split"] == "validation") & (metrics["calibrated"] == True)].boxplot(column="AUPRC", by="model_arm", rot=25)
    plt.suptitle("")
    plt.title("Q1 Neural Validation AUPRC")
    plt.tight_layout()
    plt.savefig(figures / "q1_neural_validation_auprc.png", dpi=180)
    plt.close()
    plt.figure(figsize=(8, 4))
    stability.groupby(["checkpoint_a", "checkpoint_b"])["spearman"].mean().hist(bins=20)
    plt.title("ConceptFAN Episode-Level Contribution Stability")
    plt.xlabel("mean episode-level Spearman")
    plt.tight_layout()
    plt.savefig(figures / "q1_neural_episode_stability.png", dpi=180)
    plt.close()
    return {
        "metrics_rows": int(len(metrics)),
        "local_rows": int(len(local_df)),
        "stability_rows": int(len(stability)),
        "sufficiency_rows": int(len(suff)),
        "leakage_rows": int(len(leak)),
        "robustness_rows": int(len(rob)),
        "m_sensitivity_rows": int(len(ms)),
    }


def write_model_cards(output: Path) -> None:
    cards = output / "MODEL_CARDS"
    cards.mkdir(parents=True, exist_ok=True)
    cards.joinpath("ConceptFAN-NoAlpha.md").write_text("ConceptFAN-NoAlpha: canonical temporal encoder, supervised concept projector, Gaussian fuzzy memberships, no-alpha additive decision contributions. Oracle states are labels only, never model inputs.\n", encoding="utf-8")
    cards.joinpath("NoFuzzy.md").write_text("NoFuzzy: uses the exact shared trained ConceptFAN concept path for each run and replaces only the fuzzy decision head with an additive linear concept head.\n", encoding="utf-8")
    cards.joinpath("TemporalCEM.md").write_text("TemporalCEM adaptation: temporal concept probabilities feed concept gates plus positive/negative concept embeddings; no oracle state inputs and no residual feature bypass.\n", encoding="utf-8")
    cards.joinpath("TemporalPCBM.md").write_text("Temporal PCBM-style adaptation: task backbone is trained first, frozen, post-hoc concept probes form a concept bank, and the downstream classifier uses only concept-bank outputs with no residual bypass.\n", encoding="utf-8")
    cards.joinpath("PlainTransformer.md").write_text("PlainTransformer: canonical ClinicalTransformer over raw [B,36,27] inputs, no concept labels or oracle states in the forward path.\n", encoding="utf-8")


def package_dir(source: Path, zip_path: Path, root_name: str) -> Path:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                zf.write(path, f"{root_name}/{path.relative_to(source).as_posix()}")
    zip_path.with_suffix(zip_path.suffix + ".sha256").write_text(f"{sha256_file(zip_path)}  {zip_path.name}\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
    if bad:
        raise RuntimeError(f"zip integrity failed at {bad}")
    return zip_path


def run_final(cfg: dict, output: Path, *, batch_size: int, package: bool, zip_output_dir: Path, skip_sanity: bool = False, grid_run_count: int = 30, dev_skip_analyses: bool = False, resume_existing_checkpoints: bool = False) -> dict:
    assert_config_contract(cfg)
    if output.exists() and not resume_existing_checkpoints:
        shutil.rmtree(output)
    for rel in ["CHECKPOINTS", "TABLES", "FIGURES", "MANIFESTS", "REPORTS", "MODEL_CARDS", "ARTICLE_READY"]:
        (output / rel).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    timing_probe = json.loads((ROOT / "artifacts" / "medical" / "canonical_3_seed_parity" / "MANIFESTS" / "canonical_3_seed_parity_manifest.json").read_text(encoding="utf-8"))
    existing_model_checkpoints = len(checkpoint_paths(output))
    fit_metrics_path = output / "TABLES" / "q1_neural_checkpoint_fit_metrics.csv"
    resume_training_complete = resume_existing_checkpoints and existing_model_checkpoints == 150 and fit_metrics_path.exists()
    sanity_df = pd.read_csv(output / "TABLES" / "q1_neural_model_arm_sanity_gate.csv") if resume_training_complete and (output / "TABLES" / "q1_neural_model_arm_sanity_gate.csv").exists() else (run_sanity_gate(cfg, output, batch_size) if not skip_sanity else pd.DataFrame())
    if not skip_sanity and not resume_training_complete:
        sanity_df.to_csv(output / "TABLES" / "q1_neural_model_arm_sanity_gate.csv", index=False)
        sanity_status = sanity_df.attrs.get("status", "MODEL_ARM_SANITY_FAIL")
        if sanity_status != "MODEL_ARM_SANITY_PASS":
            manifest = {"status": "Q1_NEURAL_FINAL_BLOCKED_MODEL_ARM_SANITY", "sanity_status": sanity_status, "created_utc": now()}
            (output / "MANIFESTS" / "q1_neural_final_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            return manifest
    arrays = build_arrays(cfg, 6262)
    metric_rows = []
    if not resume_training_complete:
        for meta in grid_runs(grid_run_count):
            shared_state, shared_history, shared_metrics = train_shared_concept_reference(arrays, cfg, meta, batch_size)
            shared_dir = output / "CHECKPOINTS" / "shared_concept_path" / f"run_{meta['run_id']:02d}"
            shared_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"run": meta, "state_dict": shared_state, "parameter_sha256": state_dict_sha256(shared_state), "epoch_history": shared_history, "concept_metrics": shared_metrics}, shared_dir / "shared_concept_path.pt")
            for arm in ARMS:
                metric_rows.append(train_one_arm(arm, arrays, cfg, meta, output, shared_state, shared_history, shared_metrics, batch_size))
        pd.DataFrame(metric_rows).to_csv(fit_metrics_path, index=False)
    write_model_cards(output)
    if dev_skip_analyses:
        table_counts = {"dev_skip_analyses": True}
        manifest = {
            "status": "Q1_NEURAL_FINAL_DEV_INCOMPLETE",
            "created_utc": now(),
            "code_commit": git_text(["rev-parse", "HEAD"]),
            "n_model_checkpoints": len(checkpoint_paths(output)),
            "grid_run_count": int(grid_run_count),
            "table_counts": table_counts,
        }
        (output / "MANIFESTS" / "q1_neural_final_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    table_counts = final_tables(output, arrays, cfg, batch_size)
    elapsed = time.perf_counter() - t0
    final_eligible = (not skip_sanity) and int(grid_run_count) == 30
    manifest = {
        "status": "Q1_EMPIRICAL_EXTENSION_NEURAL_VALIDATED" if final_eligible else "Q1_NEURAL_FINAL_DEV_INCOMPLETE",
        "created_utc": now(),
        "code_commit": git_text(["rev-parse", "HEAD"]),
        "branch": git_text(["branch", "--show-current"]),
        "git_status_clean_at_run": git_text(["status", "--porcelain"]) == "",
        "canonical_parity_reference": timing_probe.get("status"),
        "config_contract": config_contract(cfg),
        "n_model_checkpoints": len(checkpoint_paths(output)),
        "arms": ARMS,
        "grid": "10 initialization seeds x 3 data-order seeds",
        "grid_run_count": int(grid_run_count),
        "resume_existing_checkpoints": bool(resume_training_complete),
        "elapsed_seconds": elapsed,
        "table_counts": table_counts,
        "no_oracle_states_as_primary_input": True,
        "no_worktree_release": "_worktree" not in str(output),
    }
    (output / "MANIFESTS" / "q1_neural_final_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "REPORTS" / "Q1_NEURAL_FINAL_STATUS.md").write_text(f"# Q1 Neural Final\n\nStatus: `{manifest['status']}`\n\nCheckpoints: {manifest['n_model_checkpoints']}\n", encoding="utf-8")
    article = output / "ARTICLE_READY"
    for rel in ["q1_neural_article_ready_model_summary.csv", "q1_neural_membership_sensitivity.csv", "q1_neural_heldout_leakage_audit.csv", "q1_neural_raw_input_robustness.csv"]:
        shutil.copy2(output / "TABLES" / rel, article / rel)
    for path in (output / "FIGURES").glob("*.png"):
        shutil.copy2(path, article / path.name)
    zip_paths = {}
    if package:
        short = git_text(["rev-parse", "--short", "HEAD"])
        final_zip = package_dir(output, zip_output_dir / f"Med_CircuitBench_Q1_NEURAL_FINAL_{short}.zip", "Q1_NEURAL_FINAL")
        article_zip = package_dir(article, zip_output_dir / f"ARTICLE_READY_TABLES_AND_FIGURES_{short}.zip", "ARTICLE_READY_TABLES_AND_FIGURES")
        zip_paths = {"final_zip": str(final_zip), "final_zip_sha256": sha256_file(final_zip), "article_zip": str(article_zip), "article_zip_sha256": sha256_file(article_zip)}
    report = {**manifest, **zip_paths, "readonly_verifier": str(output / "MANIFESTS" / "q1_neural_readonly_validation.json")}
    (output / "MANIFESTS" / "q1_neural_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medical/v3/full.yaml")
    parser.add_argument("--output", default="artifacts/medical/q1_neural_final")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--zip-output-dir", default="artifacts/medical")
    parser.add_argument("--skip-sanity", action="store_true")
    parser.add_argument("--grid-run-count", type=int, default=30)
    parser.add_argument("--dev-skip-analyses", action="store_true")
    parser.add_argument("--resume-existing-checkpoints", action="store_true")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    report = run_final(cfg, ROOT / args.output, batch_size=args.batch_size, package=args.package, zip_output_dir=ROOT / args.zip_output_dir, skip_sanity=args.skip_sanity, grid_run_count=args.grid_run_count, dev_skip_analyses=args.dev_skip_analyses, resume_existing_checkpoints=args.resume_existing_checkpoints)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("status") == "Q1_EMPIRICAL_EXTENSION_NEURAL_VALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
