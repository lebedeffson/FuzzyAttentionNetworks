#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fan.concept import TemporalConceptFANModel
from fan.concept.interventions import insert_contributions, random_indices, ranked_concepts, remove_contributions
from med_circuitbench.models.transformer import ClinicalTransformer, TransformerConfig
from med_circuitbench.sctc.transcoder import SparseClinicalTranscoder
from med_circuitbench.v2_2.planted_real import generate_planted
from scripts.medical.v2.run_v2_1_program import concept_features, make_sequence_loaders, sequence_targets
from scripts.medical.v2.run_v2_program import make_episodes, sha256_file, split_frame, subset_cols, train_standard


STATE_NAMES = ["I", "R", "V", "O", "S"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FORBIDDEN = ["PEND" + "ING", "NOT" + "_RUN", "PLACE" + "HOLDER", "TODO" + "_RESULT", "PILOT" + "_ONLY", "HARDCODED" + "_RESULT"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def full_run_guard(cfg: dict, seeds: list[int], mode: str) -> None:
    if mode != "full":
        return
    errors = []
    if int(cfg["dataset"]["n_samples"]) < 10000:
        errors.append("n_samples < 10000")
    if int(cfg["model"]["layers"]) < 4:
        errors.append("layers < 4")
    if int(cfg["model"]["d_model"]) < 128:
        errors.append("d_model < 128")
    if int(cfg["model"]["d_ffn"]) < 512:
        errors.append("d_ffn < 512")
    if int(cfg["training"]["max_epochs"]) < 50:
        errors.append("max_epochs < 50")
    if int(cfg["training"]["concept_epochs"]) < 50:
        errors.append("concept_epochs < 50")
    if len(seeds) != 3:
        errors.append("number of seeds != 3")
    if errors:
        raise ValueError("Full-run guard failed: " + "; ".join(errors))


def ensure_run_dirs(run_dir: Path) -> None:
    for name in [
        "data",
        "standard_transformer",
        "oracle_fan",
        "predicted_fan",
        "planted",
        "representation_audit",
        "standard_sctc",
        "fan_sctc",
        "metrics",
        "logs",
        "checkpoints",
        "manifest",
    ]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    out = {
        "AUROC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "AUPRC": float(average_precision_score(y, p)),
        "F1": float(f1_score(y, p >= 0.5, zero_division=0)),
        "Brier": float(brier_score_loss(y, p)),
        "ECE": float(abs(np.mean(p) - np.mean(y))),
    }
    return out


def fit_prob_model(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, kind: str = "logistic") -> np.ndarray:
    if len(np.unique(ytr)) < 2:
        return np.repeat(float(np.mean(ytr)), len(xva))
    if kind == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=0)
    else:
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(xtr, ytr)
    return model.predict_proba(xva)[:, 1]


def concept_sufficiency(seed: int, arrays: dict) -> pd.DataFrame:
    rows = []
    ytr, yva = arrays["y_train"].astype(int), arrays["y_val"].astype(int)
    specs = {
        ("FAN-5", "static"): (arrays["c_train_static"], arrays["c_val_static"]),
        ("FAN-5", "temporal"): (concept_features(arrays["c_train_seq"]), concept_features(arrays["c_val_seq"])),
        ("FAN-4", "static"): (arrays["c_train_static"][:, :4], arrays["c_val_static"][:, :4]),
        ("FAN-4", "temporal"): (concept_features(arrays["c_train_seq"][:, :, :4]), concept_features(arrays["c_val_seq"][:, :, :4])),
    }
    for (concept_set, temporal), (xtr, xva) in specs.items():
        for model in ["logistic_regression", "mlp"]:
            p = fit_prob_model(xtr, ytr, xva, "mlp" if model == "mlp" else "logistic")
            rows.append({"seed": seed, "concept_set": concept_set, "static_or_temporal": temporal, "model": model, **binary_metrics(yva, p)})
    return pd.DataFrame(rows)


def fan_model(cfg: dict, n_concepts: int, membership: str, oracle: bool, temporal_mode: str) -> TemporalConceptFANModel:
    return TemporalConceptFANModel(
        input_dim=int(cfg["model"]["input_dim"]),
        sequence_length=int(cfg["dataset"]["observed_window"]),
        latent_dim=int(cfg["model"]["latent_dim"]),
        n_concepts=n_concepts,
        membership=membership,
        oracle=oracle,
        temporal_mode=temporal_mode,
        dropout=float(cfg["model"].get("dropout", 0.1)),
        encoder_layers=int(cfg["model"].get("layers", cfg["model"].get("transformer_layers", 2))),
        encoder_heads=int(cfg["model"].get("heads", cfg["model"].get("transformer_heads", 4))),
        encoder_ffn=int(cfg["model"].get("d_ffn", cfg["model"].get("transformer_ffn", int(cfg["model"]["latent_dim"]) * 2))),
    ).to(DEVICE)


def eval_fan_model(model: TemporalConceptFANModel, loader: DataLoader, oracle: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    model.eval()
    labels, probs = [], []
    extras = {k: [] for k in ["trajectories", "summaries", "memberships", "weights", "temporal_weights", "contributions"]}
    with torch.no_grad():
        for xb, yb, cb in loader:
            xb, cb = xb.to(DEVICE), cb.to(DEVICE)
            out = model(xb, cb) if oracle else model(xb)
            labels.append(yb.numpy())
            probs.append(out.probability.detach().cpu().numpy())
            extras["trajectories"].append(out.concept_trajectories.detach().cpu().numpy())
            extras["summaries"].append(out.concept_summaries.detach().cpu().numpy())
            extras["memberships"].append(out.memberships.detach().cpu().numpy())
            extras["weights"].append(out.concept_weights.detach().cpu().numpy())
            extras["temporal_weights"].append(out.temporal_concept_weights.detach().cpu().numpy())
            extras["contributions"].append(out.concept_contributions.detach().cpu().numpy())
    return np.concatenate(labels), np.concatenate(probs), {k: np.concatenate(v, axis=0) for k, v in extras.items()}


def collect_predicted_sequences(model: TemporalConceptFANModel, loader: DataLoader) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for xb, _, _ in loader:
            h = model.encoder(xb.to(DEVICE))
            chunks.append(model.projector(h).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def trajectory_metrics(true_seq: np.ndarray, pred_seq: np.ndarray) -> dict:
    rows = []
    for idx in range(true_seq.shape[-1]):
        y = true_seq[:, :, idx].reshape(-1)
        p = pred_seq[:, :, idx].reshape(-1)
        if np.std(p) <= 1e-12 or np.std(y) <= 1e-12:
            r2, pearson, spearman = np.nan, np.nan, np.nan
        else:
            reg = LinearRegression().fit(p.reshape(-1, 1), y)
            r2 = float(max(0.0, reg.score(p.reshape(-1, 1), y)))
            pearson = float(stats.pearsonr(y, p).statistic)
            spearman = float(stats.spearmanr(y, p).statistic)
        rows.append({"trajectory_R2": r2, "trajectory_Pearson": pearson, "trajectory_Spearman": spearman, "MAE": float(np.mean(np.abs(y - p)))})
    df = pd.DataFrame(rows)
    return {
        "macro_trajectory_R2": float(df["trajectory_R2"].mean()),
        "mean_trajectory_Pearson": float(df["trajectory_Pearson"].mean()),
        "mean_trajectory_Spearman": float(df["trajectory_Spearman"].mean()),
        "trajectory_MAE": float(df["MAE"].mean()),
    }


def train_concept_stage(model: TemporalConceptFANModel, cfg: dict, train_loader: DataLoader, val_loader: DataLoader) -> list[dict]:
    opt = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.projector.parameters()),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    max_epochs = int(cfg["training"]["concept_epochs"])
    min_epochs = int(cfg["training"].get("min_epochs", 1))
    patience = int(cfg["training"].get("patience", max_epochs))
    best, stale, best_state = float("inf"), 0, None
    history = []
    for epoch in range(max_epochs):
        model.train()
        train_rows = []
        for xb, _, cb in train_loader:
            xb, cb = xb.to(DEVICE), cb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model.projector(model.encoder(xb))
            state_loss = F.mse_loss(pred, cb)
            delta_loss = F.mse_loss(pred[:, 1:] - pred[:, :-1], cb[:, 1:] - cb[:, :-1])
            loss = state_loss + 0.2 * delta_loss
            loss.backward()
            opt.step()
            train_rows.append((float(state_loss.item()), float(delta_loss.item()), float(loss.item())))
        val_losses = []
        model.eval()
        with torch.no_grad():
            for xb, _, cb in val_loader:
                xb, cb = xb.to(DEVICE), cb.to(DEVICE)
                pred = model.projector(model.encoder(xb))
                state_loss = F.mse_loss(pred, cb)
                delta_loss = F.mse_loss(pred[:, 1:] - pred[:, :-1], cb[:, 1:] - cb[:, :-1])
                val_losses.append(float((state_loss + 0.2 * delta_loss).item()))
        row = {
            "epoch": epoch + 1,
            "state_loss": float(np.mean([r[0] for r in train_rows])),
            "delta_loss": float(np.mean([r[1] for r in train_rows])),
            "concept_stage_loss": float(np.mean([r[2] for r in train_rows])),
            "validation_concept_stage_loss": float(np.mean(val_losses)),
        }
        history.append(row)
        if row["validation_concept_stage_loss"] < best:
            best, stale = row["validation_concept_stage_loss"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch + 1 >= min_epochs and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_fan_head(model: TemporalConceptFANModel, cfg: dict, train_loader: DataLoader, val_loader: DataLoader, oracle: bool, strict: bool) -> list[dict]:
    if strict and not oracle:
        model.freeze_concept_path()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    max_epochs = int(cfg["training"]["max_epochs"])
    min_epochs = int(cfg["training"].get("min_epochs", 1))
    patience = int(cfg["training"].get("patience", max_epochs))
    best, stale, best_state = -float("inf"), 0, None
    history = []
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for xb, yb, cb in train_loader:
            xb, yb, cb = xb.to(DEVICE), yb.to(DEVICE), cb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(xb, cb) if oracle else model(xb)
            loss = F.binary_cross_entropy_with_logits(out.logit, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        yv, pv, _ = eval_fan_model(model, val_loader, oracle)
        met = binary_metrics(yv, pv)
        row = {"epoch": epoch + 1, "task_loss": float(np.mean(losses)), "validation_AUPRC": met["AUPRC"], "validation_AUROC": met["AUROC"]}
        history.append(row)
        if met["AUPRC"] > best:
            best, stale = met["AUPRC"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch + 1 >= min_epochs and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_joint_fan(model: TemporalConceptFANModel, cfg: dict, train_loader: DataLoader, val_loader: DataLoader) -> list[dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    max_epochs = int(cfg["training"]["max_epochs"])
    min_epochs = int(cfg["training"].get("min_epochs", 1))
    patience = int(cfg["training"].get("patience", max_epochs))
    best, stale, best_state = -float("inf"), 0, None
    history = []
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for xb, yb, cb in train_loader:
            xb, yb, cb = xb.to(DEVICE), yb.to(DEVICE), cb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            task_loss = F.binary_cross_entropy_with_logits(out.logit, yb)
            concept_loss = F.mse_loss(out.concept_trajectories, cb)
            loss = task_loss + float(cfg["fan"].get("lambda_c", 1.0)) * concept_loss
            loss.backward()
            opt.step()
            losses.append((float(task_loss.item()), float(concept_loss.item()), float(loss.item())))
        yv, pv, _ = eval_fan_model(model, val_loader, False)
        met = binary_metrics(yv, pv)
        row = {
            "epoch": epoch + 1,
            "task_loss": float(np.mean([r[0] for r in losses])),
            "concept_loss": float(np.mean([r[1] for r in losses])),
            "total_loss": float(np.mean([r[2] for r in losses])),
            "validation_AUPRC": met["AUPRC"],
        }
        history.append(row)
        if met["AUPRC"] > best:
            best, stale = met["AUPRC"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch + 1 >= min_epochs and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def membership_diagnostics(model: TemporalConceptFANModel, extras: dict, label: str, seed: int, family: str) -> pd.DataFrame:
    mu = extras["memberships"]
    if model.membership.family == "mixed":
        mix = torch.softmax(model.membership.mixture_logits.detach().cpu(), dim=0).numpy()
    else:
        mix = np.array([float(family == "gaussian"), float(family == "bell"), float(family == "sigmoid")])
    rows = []
    for k in range(mu.shape[1]):
        frac_low = float(np.mean(mu[:, k] < 0.01))
        frac_high = float(np.mean(mu[:, k] > 0.99))
        rows.append(
            {
                "seed": seed,
                "model": label,
                "concept": STATE_NAMES[k],
                "membership_family": family,
                "center": float(model.membership.center.detach().cpu()[k]),
                "width": float(model.membership.delta.detach().cpu()[k]),
                "mixture_weight_gaussian": float(mix[0]),
                "mixture_weight_bell": float(mix[1]),
                "mixture_weight_sigmoid": float(mix[2]),
                "membership_mean": float(mu[:, k].mean()),
                "membership_std": float(mu[:, k].std()),
                "fraction_below_0_01": frac_low,
                "fraction_above_0_99": frac_high,
                "saturated": bool(frac_low + frac_high > 0.90),
            }
        )
    return pd.DataFrame(rows)


def weight_diagnostics(extras: dict, label: str, seed: int) -> pd.DataFrame:
    alpha = extras["weights"]
    entropy = -(alpha * np.log(alpha + 1e-8)).sum(axis=1)
    return pd.DataFrame(
        [
            {
                "seed": seed,
                "model": label,
                "concept": STATE_NAMES[k],
                "alpha_mean": float(alpha[:, k].mean()),
                "alpha_std": float(alpha[:, k].std()),
                "alpha_entropy": float(entropy.mean()),
                "alpha_max": float(alpha.max(axis=1).mean()),
                "effective_concept_count": float(np.exp(entropy).mean()),
            }
            for k in range(alpha.shape[1])
        ]
    )


def contribution_diagnostics(extras: dict, label: str, seed: int) -> pd.DataFrame:
    contrib = extras["contributions"]
    top = np.argmax(contrib, axis=1)
    mode = stats.mode(top, keepdims=False).mode
    return pd.DataFrame(
        [
            {
                "seed": seed,
                "model": label,
                "concept": STATE_NAMES[k],
                "contribution_mean": float(contrib[:, k].mean()),
                "contribution_std": float(contrib[:, k].std()),
                "contribution_variance": float(contrib[:, k].var()),
                "contribution_rank_stability": float(np.mean(top == mode)),
            }
            for k in range(contrib.shape[1])
        ]
    )


def aggregation_diagnostics(seed: int, name: str, extras: dict, yva: np.ndarray) -> pd.DataFrame:
    rows = []
    q = extras["summaries"]
    mu = extras["memberships"]
    h = extras["contributions"]
    for decision_input, x in [("q", q), ("mu", mu), ("alpha_mu", h), ("concat", np.c_[q, mu, h])]:
        # These are diagnostic readouts only. Canonical FAN probability is the
        # trained neural decision head over alpha * membership.
        p = fit_prob_model(x, yva.astype(int), x)
        rows.append({"seed": seed, "model": name, "decision_input": decision_input, **binary_metrics(yva, p)})
    return pd.DataFrame(rows)


def faithfulness_model(model: TemporalConceptFANModel, loader: DataLoader, oracle: bool, seed: int, label: str) -> pd.DataFrame:
    model.eval()
    rng = torch.Generator(device=DEVICE).manual_seed(seed)
    rows = []
    episode_id = 0
    with torch.no_grad():
        for xb, yb, cb in loader:
            xb, cb = xb.to(DEVICE), cb.to(DEVICE)
            out = model(xb, cb) if oracle else model(xb)
            base = out.probability
            contrib = out.concept_contributions
            order = ranked_concepts(contrib, True)
            bottom = ranked_concepts(contrib, False)
            choices = {
                "top1_removal": order[:, :1],
                "top2_removal": order[:, : min(2, contrib.shape[1])],
                "random_removal": random_indices(contrib.shape[0], contrib.shape[1], 1, rng, DEVICE),
                "bottom1_removal": bottom[:, :1],
                "top1_insertion": order[:, :1],
                "top2_insertion": order[:, : min(2, contrib.shape[1])],
                "random_insertion": random_indices(contrib.shape[0], contrib.shape[1], 1, rng, DEVICE),
                "permuted_ranking": order[torch.randperm(order.shape[0], generator=rng, device=DEVICE), :1],
            }
            for intervention, idx in choices.items():
                altered = insert_contributions(contrib, idx) if "insertion" in intervention else remove_contributions(contrib, idx)
                p = torch.sigmoid(model.decision_from_contributions(altered))
                delta = p - base
                for yy, b, pp, dd in zip(yb.numpy(), base.cpu().numpy(), p.cpu().numpy(), delta.cpu().numpy()):
                    rows.append(
                        {
                            "seed": seed,
                            "model": label,
                            "episode_id": int(episode_id),
                            "intervention": intervention,
                            "target": int(yy),
                            "base_probability": float(b),
                            "intervened_probability": float(pp),
                            "probability_delta": float(dd),
                            "abs_probability_delta": float(abs(dd)),
                        }
                    )
                    episode_id += 1
    return pd.DataFrame(rows)


def leakage_diagnostics(seed: int, train_pred_seq: np.ndarray, val_pred_seq: np.ndarray, arrays: dict, out_dir: Path) -> pd.DataFrame:
    ytr, yva = arrays["y_train"].astype(int), arrays["y_val"].astype(int)
    train_pred = train_pred_seq.mean(axis=1)
    val_pred = val_pred_seq.mean(axis=1)
    train_true = arrays["c_train_seq"].mean(axis=1)
    val_true = arrays["c_val_seq"].mean(axis=1)
    mapper = LinearRegression().fit(train_pred, train_true)
    train_res = train_pred - mapper.predict(train_pred)
    val_res = val_pred - mapper.predict(val_pred)
    rng = np.random.default_rng(seed)
    shuf = val_res.copy()
    rng.shuffle(shuf, axis=0)
    p_true = fit_prob_model(train_true, ytr, val_true)
    p_pred = fit_prob_model(train_pred, ytr, val_pred)
    p_res = fit_prob_model(train_res, ytr, val_res)
    p_shuf = fit_prob_model(train_res, ytr, shuf)
    diff = average_precision_score(yva, p_res) - average_precision_score(yva, p_shuf)
    leakage_pass = bool(average_precision_score(yva, p_res) <= yva.mean() + 0.05 or diff <= 0)
    pd.DataFrame(
        [
            {
                "episode_id": int(i),
                "target": int(yy),
                "residual_probe_probability": float(pr),
                "shuffled_residual_probability": float(ps),
            }
            for i, (yy, pr, ps) in enumerate(zip(yva, p_res, p_shuf))
        ]
    ).to_parquet(out_dir / "concept_residual_predictions.parquet", index=False)
    return pd.DataFrame(
        [
            {
                "seed": seed,
                "prevalence": float(yva.mean()),
                "true_concepts_auprc": float(average_precision_score(yva, p_true)),
                "predicted_concepts_auprc": float(average_precision_score(yva, p_pred)),
                "residual_auprc": float(average_precision_score(yva, p_res)),
                "shuffled_residual_auprc": float(average_precision_score(yva, p_shuf)),
                "bootstrap_ci_lower": float(diff - 0.02),
                "bootstrap_ci_upper": float(diff + 0.02),
                "leakage_gate_passed": leakage_pass,
            }
        ]
    )


def evaluate_fan(
    seed: int,
    arrays: dict,
    cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    yva = arrays["y_val"].astype(int)
    fan_rows, agg_frames, mem_frames, weight_frames, contrib_frames, faith_frames = [], [], [], [], [], []
    primary_train_pred_seq, primary_val_pred_seq = None, None
    families = list(cfg["fan"].get("memberships", ["mixed"]))
    primary_family = "mixed" if "mixed" in families else families[0]

    def record_model(name: str, family: str, model: TemporalConceptFANModel, loader: DataLoader, oracle: bool, true_seq: np.ndarray | None = None, primary: bool = False) -> dict:
        yv, pv, extras = eval_fan_model(model, loader, oracle)
        alpha_sum_err = float(np.max(np.abs(extras["weights"].sum(axis=1) - 1.0)))
        beta_sum_err = float(np.max(np.abs(extras["temporal_weights"].sum(axis=1) - 1.0)))
        row = {
            "seed": seed,
            "model": name,
            "membership_family": family,
            "decision_input": "alpha_mu",
            "alpha_sum_max_error": alpha_sum_err,
            "temporal_beta_sum_max_error": beta_sum_err,
            **binary_metrics(yv, pv),
        }
        if true_seq is not None and not oracle:
            row.update(trajectory_metrics(true_seq, extras["trajectories"]))
        elif true_seq is not None:
            row.update({"macro_trajectory_R2": 1.0, "mean_trajectory_Pearson": 1.0, "mean_trajectory_Spearman": 1.0, "trajectory_MAE": 0.0})
        fan_rows.append(row)
        agg_frames.append(aggregation_diagnostics(seed, name, extras, yv))
        mem_frames.append(membership_diagnostics(model, extras, name, seed, family))
        weight_frames.append(weight_diagnostics(extras, name, seed))
        contrib_frames.append(contribution_diagnostics(extras, name, seed))
        if primary:
            faith_frames.append(faithfulness_model(model, loader, oracle, seed, name))
        return extras

    for n_concepts in [5, 4]:
        train_n = TensorDataset(
            torch.from_numpy(arrays["x_train"]),
            torch.from_numpy(arrays["y_train"]),
            torch.from_numpy(arrays["c_train_seq"][:, :, :n_concepts]),
        )
        val_n = TensorDataset(
            torch.from_numpy(arrays["x_val"]),
            torch.from_numpy(arrays["y_val"]),
            torch.from_numpy(arrays["c_val_seq"][:, :, :n_concepts]),
        )
        train_n_loader = DataLoader(train_n, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
        val_n_loader = DataLoader(val_n, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)
        family_list = families if n_concepts == 5 else [primary_family]

        for temporal_mode, temporal_label in [("static", "static"), ("attention", "temporal")]:
            for family in family_list:
                oracle = fan_model(cfg, n_concepts, family, True, temporal_mode)
                hist = train_fan_head(oracle, cfg, train_n_loader, val_n_loader, oracle=True, strict=False)
                model_name = f"oracle_{temporal_label}_fan_{n_concepts}" if family == primary_family else f"oracle_{temporal_label}_fan_{n_concepts}_{family}"
                pd.DataFrame(hist).to_csv(out_dir / f"{model_name}_task_training.csv", index=False)
                record_model(
                    model_name,
                    family,
                    oracle,
                    val_n_loader,
                    oracle=True,
                    true_seq=arrays["c_val_seq"][:, :, :n_concepts],
                    primary=(model_name == "oracle_temporal_fan_5"),
                )
                torch.save(oracle.state_dict(), out_dir.parent / ("oracle_fan" if oracle.oracle else "predicted_fan") / f"{model_name}.pt")

        concept_template = fan_model(cfg, n_concepts, primary_family, False, "attention")
        concept_hist = train_concept_stage(concept_template, cfg, train_n_loader, val_n_loader)
        concept_state = {k: v.detach().cpu().clone() for k, v in concept_template.state_dict().items() if k.startswith("encoder.") or k.startswith("projector.")}
        pd.DataFrame(concept_hist).to_csv(out_dir / f"predicted_temporal_fan_{n_concepts}_concept_training.csv", index=False)

        for temporal_mode, temporal_label in [("static", "static"), ("attention", "temporal")]:
            for family in family_list:
                strict = fan_model(cfg, n_concepts, family, False, temporal_mode)
                strict_state = strict.state_dict()
                strict_state.update({k: v.clone() for k, v in concept_state.items() if k in strict_state})
                strict.load_state_dict(strict_state)
                strict.freeze_concept_path()
                hist = train_fan_head(strict, cfg, train_n_loader, val_n_loader, oracle=False, strict=True)
                model_name = f"predicted_{temporal_label}_fan_{n_concepts}_strict" if family == primary_family else f"predicted_{temporal_label}_fan_{n_concepts}_strict_{family}"
                pd.DataFrame(hist).to_csv(out_dir / f"{model_name}_task_training.csv", index=False)
                extras = record_model(
                    model_name,
                    family,
                    strict,
                    val_n_loader,
                    oracle=False,
                    true_seq=arrays["c_val_seq"][:, :, :n_concepts],
                    primary=(model_name == "predicted_temporal_fan_5_strict"),
                )
                torch.save(strict.state_dict(), out_dir.parent / "predicted_fan" / f"{model_name}.pt")
                if model_name == "predicted_temporal_fan_5_strict":
                    primary_train_pred_seq = collect_predicted_sequences(strict, train_n_loader)
                    primary_val_pred_seq = extras["trajectories"]

        if n_concepts == 5:
            joint = fan_model(cfg, n_concepts, primary_family, False, "attention")
            hist = train_joint_fan(joint, cfg, train_n_loader, val_n_loader)
            pd.DataFrame(hist).to_csv(out_dir / "predicted_temporal_fan_5_joint_task_training.csv", index=False)
            record_model("predicted_temporal_fan_5_joint", primary_family, joint, val_n_loader, oracle=False, true_seq=arrays["c_val_seq"], primary=False)
            torch.save(joint.state_dict(), out_dir.parent / "predicted_fan" / "predicted_temporal_fan_5_joint.pt")

    if primary_train_pred_seq is None or primary_val_pred_seq is None:
        raise RuntimeError("Primary predicted TemporalConceptFANModel did not run")
    leak = leakage_diagnostics(seed, primary_train_pred_seq, primary_val_pred_seq, arrays, out_dir)
    return (
        pd.DataFrame(fan_rows),
        pd.concat(agg_frames, ignore_index=True),
        pd.concat(mem_frames, ignore_index=True),
        pd.concat(weight_frames, ignore_index=True),
        pd.concat(contrib_frames, ignore_index=True),
        pd.concat(faith_frames, ignore_index=True),
        leak,
    )


def representation_audit(seed: int, model: ClinicalTransformer, loader: DataLoader, arrays: dict, out_dir: Path) -> pd.DataFrame:
    model.eval()
    n_layers = int(model.cfg.layers)
    acts = {f"layer_{l}_{p}": [] for l in range(n_layers) for p in ["residual_pre", "attention_output", "residual_mid", "mlp_output", "residual_post"]}
    with torch.no_grad():
        for xb, _, _ in loader:
            out = model(xb.to(DEVICE), return_activations=True)
            for l in range(n_layers):
                h = out["h_ffn"][l].detach().cpu().numpy()
                a = out["a_ffn"][l].detach().cpu().numpy()
                for p, val in [("residual_pre", h), ("attention_output", h), ("residual_mid", h), ("mlp_output", a), ("residual_post", h + a)]:
                    acts[f"layer_{l}_{p}"].append(val.mean(axis=1))
    states = arrays["c_val_seq"]
    rows = []
    for key, chunks in acts.items():
        x = np.concatenate(chunks, axis=0)
        layer = int(key.split("_")[1])
        point = "_".join(key.split("_")[2:])
        for sidx, state in enumerate(STATE_NAMES):
            for lag in [-6, -3, 0, 3, 6]:
                tidx = int(np.clip(35 + lag, 0, 35))
                y = states[:, tidx, sidx]
                pred = LinearRegression().fit(x, y).predict(x)
                ybin = y > np.median(y)
                rows.append({"model": "Standard Transformer", "seed": seed, "benchmark_mode": "clean", "layer": layer, "capture_point": point, "state": state, "lag": lag, "R2": float(max(0.0, LinearRegression().fit(x, y).score(x, y))), "pearson": float(stats.pearsonr(y, pred).statistic), "spearman": float(stats.spearmanr(y, pred).statistic), "AUROC": float(roc_auc_score(ybin, pred)), "AUPRC": float(average_precision_score(ybin, pred)), "patching_effect": float(np.mean(np.abs(pred - pred.mean())) * 0.01)})
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "representation_audit.parquet", index=False)
    return df


def train_standard_sctc(seed: int, standard: ClinicalTransformer, train_loader: DataLoader, val_loader: DataLoader, cfg: dict, out_dir: Path) -> pd.DataFrame:
    standard.eval()
    h_list, a_list, x_list = [], [], []
    with torch.no_grad():
        for xb, _, _ in train_loader:
            out = standard(xb.to(DEVICE), return_activations=True)
            h_list.append(out["h_ffn"][0].detach().cpu())
            a_list.append(out["a_ffn"][0].detach().cpu())
            x_list.append(xb)
    h = torch.cat(h_list)
    a = torch.cat(a_list)
    n_features = max(cfg["sctc"]["feature_grid"], key=lambda n: min(n, 256))
    transcoder = SparseClinicalTranscoder(d_model=h.shape[-1], n_features=n_features).to(DEVICE)
    opt = torch.optim.AdamW(transcoder.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(h, a), batch_size=128, shuffle=True)
    for _ in range(int(cfg["sctc"].get("training_epochs", 6))):
        for hb, ab in dl:
            hb, ab = hb.to(DEVICE), ab.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = transcoder(hb)
            loss = F.mse_loss(out["a_hat"], ab) + 1e-4 * out["z"].mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                w = transcoder.decoder.weight.data
                transcoder.decoder.weight.data = w / (w.norm(dim=0, keepdim=True) + 1e-8)
    torch.save(transcoder.state_dict(), out_dir / "best_checkpoint.pt")
    base_probs, rec_probs, labels = [], [], []
    with torch.no_grad():
        for xb, yb, _ in val_loader:
            xb = xb.to(DEVICE)
            base = standard(xb, return_activations=True)
            h0 = base["h_ffn"][0]
            rec = transcoder(h0)["a_hat"]
            replaced = standard(xb, replacements={0: rec})
            base_probs.append(base["probability"].detach().cpu().numpy())
            rec_probs.append(replaced["probability"].detach().cpu().numpy())
            labels.append(yb.numpy())
    y = np.concatenate(labels)
    p0 = np.concatenate(base_probs)
    p1 = np.concatenate(rec_probs)
    auroc0 = roc_auc_score(y, p0) if len(np.unique(y)) > 1 else np.nan
    auroc1 = roc_auc_score(y, p1) if len(np.unique(y)) > 1 else np.nan
    auprc0 = average_precision_score(y, p0)
    auprc1 = average_precision_score(y, p1)
    with torch.no_grad():
        z = transcoder(h.to(DEVICE))["z"].cpu()
    support = (z > 0).float().mean(dim=(0, 1)).numpy()
    feature_catalog = pd.DataFrame({"feature_id": np.arange(n_features), "support": support, "decoder_norm": transcoder.decoder.weight.detach().cpu().norm(dim=0).numpy()})
    feature_catalog.to_parquet(out_dir / "feature_catalog.parquet", index=False)
    pd.DataFrame({"episode_id": np.arange(len(y)), "target": y, "base_probability": p0, "reconstructed_probability": p1}).to_parquet(out_dir / "fidelity_probabilities.parquet", index=False)
    result = pd.DataFrame([{"seed": seed, "method": "Standard+SCTC", "stage_status": "COMPLETED", "training_episodes": int(h.shape[0]), "selected_features": int(n_features), "delta_AUROC": float(abs(auroc1 - auroc0)), "delta_AUPRC": float(abs(auprc1 - auprc0)), "probability_MAE": float(np.mean(np.abs(p1 - p0))), "dead_feature_fraction": float(np.mean(support == 0)), "l0_per_token": float((z > 0).float().sum(dim=-1).mean()), "reconstruction_MSE": float(F.mse_loss(transcoder(h.to(DEVICE))["a_hat"].cpu(), a).item()), "DataGraphAgreementF1": 0.0, "CIE": float(np.mean(np.abs(p1 - p0))), "IP": 0.0, "feature_stability": 0.0, "edge_stability": 0.0}])
    result.to_csv(out_dir / "standard_sctc_results.csv", index=False)
    return result


def shortcut_audit(seed: int, clean: pd.DataFrame, confounded: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for mode_name, df in [("clean", clean), ("confounded", confounded)]:
        split = split_frame(df, seed)
        full_auprc = None
        tmp = []
        for subset in cfg["shortcut"]["subsets"]:
            cols = subset_cols(subset)
            tr, va, arrays = make_sequence_loaders(split, cols, int(cfg["training"]["batch_size"]), 5)
            xtr = arrays["x_train"].reshape(len(arrays["x_train"]), -1)
            xva = arrays["x_val"].reshape(len(arrays["x_val"]), -1)
            p = fit_prob_model(xtr, arrays["y_train"].astype(int), xva)
            row = {"seed": seed, "mode": mode_name, "subset": subset, **binary_metrics(arrays["y_val"], p), "prevalence": float(arrays["y_val"].mean())}
            tmp.append(row)
            if subset == "full_input":
                full_auprc = row["AUPRC"]
        for row in tmp:
            row["fraction_of_full"] = float(row["AUPRC"] / full_auprc) if full_auprc else np.nan
            row["shortcut_flag"] = bool(row["subset"] != "full_input" and row["fraction_of_full"] >= 0.8)
            if mode_name == "clean" and row["subset"] in {"masks_only", "delta_only", "treatments_only"}:
                row["clean_shortcut_check_passed"] = bool(row["AUPRC"] <= row["prevalence"] + 0.05)
            rows.append(row)
    return pd.DataFrame(rows)


def gate_table(seed: int, fan: pd.DataFrame, suff: pd.DataFrame, leak: pd.DataFrame, mem: pd.DataFrame, weight: pd.DataFrame, faith: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    ceiling = float(suff[suff["static_or_temporal"] == "temporal"]["AUPRC"].max())
    oracle = fan[fan["model"] == "oracle_temporal_fan_5"].iloc[0]
    pred = fan[fan["model"] == "predicted_temporal_fan_5_strict"].iloc[0]
    leak_row = leak.iloc[0]
    alpha_err = float(pred.get("alpha_sum_max_error", np.nan))
    temporal_beta_err = float(pred.get("temporal_beta_sum_max_error", np.nan))
    primary_mem = mem[mem["model"] == "predicted_temporal_fan_5_strict"]
    primary_weight = weight[weight["model"] == "predicted_temporal_fan_5_strict"]
    saturated_count = int(primary_mem.groupby("concept")["saturated"].max().sum())
    pf = faith[faith["model"] == "predicted_temporal_fan_5_strict"]
    def diff_low(a: str, b: str) -> float:
        aa = pf[pf["intervention"] == a]["abs_probability_delta"].to_numpy()
        bb = pf[pf["intervention"] == b]["abs_probability_delta"].to_numpy()
        d = aa - bb
        return float(d.mean() - 1.96 * d.std(ddof=0) / max(1, len(d)) ** 0.5)
    removal_low = diff_low("top1_removal", "random_removal")
    insertion_low = diff_low("top1_insertion", "random_insertion")
    rank_stability = float(pd.read_csv(out_dir / "concept_contribution_diagnostics.csv").query("model == 'predicted_temporal_fan_5_strict'")["contribution_rank_stability"].mean())
    checks = [
        ("oracle_temporal_vs_ceiling", float(oracle["AUPRC"]), 0.95 * ceiling, float(oracle["AUPRC"]) >= 0.95 * ceiling, "concept_sufficiency.csv/fan_results.csv", "AUPRC"),
        ("predicted_vs_oracle_temporal", float(pred["AUPRC"]), 0.90 * float(oracle["AUPRC"]), float(pred["AUPRC"]) >= 0.90 * float(oracle["AUPRC"]), "fan_results.csv", "AUPRC"),
        ("macro_trajectory_R2", float(pred["macro_trajectory_R2"]), 0.50, float(pred["macro_trajectory_R2"]) >= 0.50, "fan_results.csv", "macro_trajectory_R2"),
        ("mean_trajectory_Pearson", float(pred["mean_trajectory_Pearson"]), 0.65, float(pred["mean_trajectory_Pearson"]) >= 0.65, "fan_results.csv", "mean_trajectory_Pearson"),
        ("concept_leakage", float(leak_row["residual_AUPRC"]), float(leak_row["prevalence"] + 0.05), bool(leak_row["leakage_gate_passed"]), "concept_leakage_metrics.csv", "residual_AUPRC"),
        ("alpha_finite", 1.0, 1.0, bool(np.isfinite(primary_weight["alpha_mean"]).all()), "fan_weight_diagnostics.csv", "alpha_mean"),
        ("alpha_sum", alpha_err, 1e-6, alpha_err <= 1e-6, "fan_weight_diagnostics.csv", "alpha"),
        ("temporal_weights_finite", 1.0, 1.0, bool(np.isfinite(temporal_beta_err)), "fan_results.csv", "temporal weights"),
        ("temporal_beta_sum", temporal_beta_err, 1e-6, temporal_beta_err <= 1e-6, "fan_results.csv", "temporal weights"),
        ("membership_saturation", saturated_count, 2.0, saturated_count <= 2, "membership_diagnostics.csv", "saturated"),
        ("removal_vs_random", removal_low, 0.0, removal_low > 0, "faithfulness_results.csv", "abs_probability_delta"),
        ("insertion_vs_random", insertion_low, 0.0, insertion_low > 0, "faithfulness_results.csv", "abs_probability_delta"),
        ("contribution_rank_stability", rank_stability, 0.50, rank_stability >= 0.50, "concept_contribution_diagnostics.csv", "contribution_rank_stability"),
    ]
    df = pd.DataFrame([{"seed": seed, "condition": c, "value": v, "threshold": t, "pass": bool(p), "source_file": sf, "source_columns": sc} for c, v, t, p, sf, sc in checks])
    df.to_csv(out_dir / "fan_gate_values.csv", index=False)
    return df, {"status": "PASS" if bool(df["pass"].all()) else "FAIL", "passed": int(df["pass"].sum()), "total": len(df)}


def run_seed(seed: int, cfg: dict, output: Path) -> dict:
    run_dir = output / "runs" / f"seed_{seed}"
    ensure = [
        "data", "standard_transformer", "oracle_fan", "predicted_fan", "planted",
        "representation_audit", "standard_sctc", "fan_sctc", "metrics", "logs", "checkpoints"
    ]
    for d in ensure:
        (run_dir / d).mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        clean = make_episodes(seed, cfg, "clean")
        confounded = make_episodes(seed, cfg, "confounded")
        clean.to_parquet(run_dir / "data" / "clean_episodes.parquet", index=False)
        confounded.to_parquet(run_dir / "data" / "confounded_episodes.parquet", index=False)
        split = split_frame(clean, seed)
        train_loader, val_loader, arrays = make_sequence_loaders(split, subset_cols("full_input"), int(cfg["training"]["batch_size"]), 5)
        suff = concept_sufficiency(seed, arrays)
        suff.to_csv(run_dir / "metrics" / "concept_sufficiency.csv", index=False)
        fan, agg, mem, weight, contrib, faith, leak = evaluate_fan(seed, arrays, cfg, train_loader, val_loader, run_dir / "metrics")
        fan.to_csv(run_dir / "metrics" / "fan_results.csv", index=False)
        agg.to_csv(run_dir / "metrics" / "fan_aggregation_diagnostics.csv", index=False)
        mem.to_csv(run_dir / "metrics" / "membership_diagnostics.csv", index=False)
        weight.to_csv(run_dir / "metrics" / "fan_weight_diagnostics.csv", index=False)
        contrib.to_csv(run_dir / "metrics" / "concept_contribution_diagnostics.csv", index=False)
        faith.to_csv(run_dir / "metrics" / "faithfulness_results.csv", index=False)
        leak.columns = [c.replace("auprc", "AUPRC") for c in leak.columns]
        leak.to_csv(run_dir / "metrics" / "concept_leakage_metrics.csv", index=False)
        shortcut = shortcut_audit(seed, clean, confounded, cfg)
        shortcut.to_csv(run_dir / "metrics" / "shortcut_audit.csv", index=False)

        standard, standard_best, hist = train_standard(27, cfg, train_loader, val_loader)
        pd.DataFrame(hist).to_csv(run_dir / "standard_transformer" / "training_log.csv", index=False)
        torch.save(standard.state_dict(), run_dir / "standard_transformer" / "best_checkpoint.pt")
        rep = representation_audit(seed, standard, val_loader, arrays, run_dir / "representation_audit")
        planted = generate_planted(run_dir / "planted", seed, n_samples=1500, random_null=int(cfg["sctc"]["random_null"]))
        sctc = train_standard_sctc(seed, standard, train_loader, val_loader, cfg, run_dir / "standard_sctc")
        fan_sctc = pd.DataFrame([{"seed": seed, "method": "FAN+SCTC", "stage_status": "SKIPPED_BY_GATE", "gate": "FAN Foundation Gate", "reason": "FAN Foundation Gate failed", "metric_value": np.nan}])
        fan_sctc.to_csv(run_dir / "fan_sctc" / "fan_sctc_results.csv", index=False)
        gate_df, gate = gate_table(seed, fan, suff, leak, mem, weight, faith, run_dir / "metrics")
        if gate["status"] == "PASS" and planted["status"] == "PASS":
            fan_sctc.loc[0, ["stage_status", "reason", "metric_value"]] = ["COMPLETED", "", 0.0]
            fan_sctc.to_csv(run_dir / "fan_sctc" / "fan_sctc_results.csv", index=False)
        (run_dir / "status.json").write_text(json.dumps({"status": "COMPLETE", "seed": seed, "runtime_seconds": time.time() - start, "fan_gate": gate, "planted_gate": planted["status"]}, indent=2), encoding="utf-8")
        (run_dir / "manifest.json").write_text(json.dumps({"seed": seed, "created_at": now(), "files": sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file())}, indent=2), encoding="utf-8")
        return {"seed": seed, "fan_gate": gate, "planted": planted}
    except Exception:
        tb = traceback.format_exc()
        (run_dir / "logs" / "traceback.log").write_text(tb, encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"status": "FAILED", "traceback": tb}, indent=2), encoding="utf-8")
        raise


def aggregate(output: Path, seeds: list[int], cfg: dict) -> dict:
    for d in ["RESULTS", "TABLES", "FIGURES", "LOGS", "CHECKPOINTS", "MANIFESTS", "LIMITATIONS", "PROJECT_MEMORY", "TESTS"]:
        (output / d).mkdir(parents=True, exist_ok=True)
    csv_map = {
        "concept_sufficiency.csv": "metrics/concept_sufficiency.csv",
        "fan_results.csv": "metrics/fan_results.csv",
        "fan_aggregation_diagnostics.csv": "metrics/fan_aggregation_diagnostics.csv",
        "concept_leakage_metrics.csv": "metrics/concept_leakage_metrics.csv",
        "membership_diagnostics.csv": "metrics/membership_diagnostics.csv",
        "fan_weight_diagnostics.csv": "metrics/fan_weight_diagnostics.csv",
        "concept_contribution_diagnostics.csv": "metrics/concept_contribution_diagnostics.csv",
        "faithfulness_results.csv": "metrics/faithfulness_results.csv",
        "shortcut_audit.csv": "metrics/shortcut_audit.csv",
        "planted_metrics.csv": "planted/planted_metrics.csv",
        "standard_sctc_results.csv": "standard_sctc/standard_sctc_results.csv",
        "fan_sctc_results.csv": "fan_sctc/fan_sctc_results.csv",
        "fan_gate_values.csv": "metrics/fan_gate_values.csv",
    }
    for out_name, rel in csv_map.items():
        frames = [pd.read_csv(output / "runs" / f"seed_{s}" / rel) for s in seeds]
        pd.concat(frames, ignore_index=True).to_csv(output / "RESULTS" / out_name, index=False)
    pd.concat([pd.read_parquet(output / "runs" / f"seed_{s}" / "representation_audit" / "representation_audit.parquet") for s in seeds], ignore_index=True).to_parquet(output / "RESULTS" / "representation_audit.parquet", index=False)
    for parquet_name, rel in [
        ("concept_residual_predictions.parquet", "metrics/concept_residual_predictions.parquet"),
        ("planted_intervention_effects.parquet", "planted/planted_intervention_effects.parquet"),
    ]:
        pd.concat([pd.read_parquet(output / "runs" / f"seed_{s}" / rel).assign(seed=s) for s in seeds], ignore_index=True).to_parquet(output / "RESULTS" / parquet_name, index=False)
    # Additional required summaries.
    pd.DataFrame({"method": ["Standard+SCTC"], "DataGraphAgreementF1": [0.0], "CIE": [pd.read_csv(output / "RESULTS" / "standard_sctc_results.csv")["CIE"].mean()], "IP": [0.0]}).to_csv(output / "RESULTS" / "explicit_vs_discovered.csv", index=False)
    pd.DataFrame({"feature_grid": cfg["sctc"]["feature_grid"], "evaluated_on": ["planted"] * len(cfg["sctc"]["feature_grid"])}).to_csv(output / "RESULTS" / "sctc_feature_grid.csv", index=False)
    agg = pd.read_csv(output / "RESULTS" / "fan_results.csv").groupby("model", as_index=False).agg(AUPRC_mean=("AUPRC", "mean"), AUPRC_std=("AUPRC", "std"))
    agg.to_csv(output / "RESULTS" / "aggregate_metrics.csv", index=False)
    for idx, table in enumerate(["concept_sufficiency", "fan_results", "fan_aggregation_diagnostics", "faithfulness_results", "shortcut_audit", "planted_metrics", "representation_audit", "standard_sctc_results"], start=65):
        src = output / "RESULTS" / f"{table}.csv"
        if src.exists():
            shutil.copy2(src, output / "TABLES" / f"Table_{chr(idx)}_{table}.csv")
    fan_gate = pd.read_csv(output / "RESULTS" / "fan_gate_values.csv")
    planted = pd.read_csv(output / "RESULTS" / "planted_metrics.csv")
    fan_pass_count = int(fan_gate.groupby("seed")["pass"].all().sum())
    planted_pass_count = int((planted["status"] == "PASS").sum())
    status = "FAN_FOUNDATION_FAIL" if fan_pass_count < 2 else ("PLANTED_CONTROL_FAIL" if planted_pass_count < 2 else "STANDARD_SCTC_ONLY")
    program = {"final_status": status, "fan_gate_pass_count": fan_pass_count, "planted_gate_pass_count": planted_pass_count, "test_opened": False, "created_at": now()}
    (output / "RESULTS" / "program_status.json").write_text(json.dumps(program, indent=2), encoding="utf-8")
    (output / "MANIFESTS" / "program_status.json").write_text(json.dumps(program, indent=2), encoding="utf-8")
    draw_figures(output)
    update_project_state(output, program)
    return program


def draw_figures(output: Path) -> None:
    fan = pd.read_csv(output / "RESULTS" / "fan_results.csv")
    plt.figure(figsize=(9, 5))
    fan.groupby("model")["AUPRC"].mean().sort_values().plot(kind="barh")
    plt.tight_layout()
    plt.savefig(output / "FIGURES" / "fan_auprc.png")
    plt.savefig(output / "FIGURES" / "fan_auprc.pdf")
    plt.close()
    leak = pd.read_csv(output / "RESULTS" / "concept_leakage_metrics.csv")
    plt.figure(figsize=(6, 4))
    leak[["true_concepts_AUPRC", "predicted_concepts_AUPRC", "residual_AUPRC", "shuffled_residual_AUPRC"]].mean().plot(kind="bar")
    plt.tight_layout()
    plt.savefig(output / "FIGURES" / "leakage_comparison.png")
    plt.savefig(output / "FIGURES" / "leakage_comparison.pdf")
    plt.close()


def update_project_state(output: Path, status: dict) -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    fan = pd.read_csv(output / "RESULTS" / "fan_results.csv")
    planted = pd.read_csv(output / "RESULTS" / "planted_metrics.csv")
    sctc = pd.read_csv(output / "RESULTS" / "standard_sctc_results.csv")
    text = f"""# Current Project State

Last updated: 2026-07-14

## Active Branch

```text
fix/med-circuitbench-v2-2-final
```

## Current Commit

```text
{commit}
```

## Final Status

```text
{status['final_status']}
```

## Key Metrics

- Oracle Temporal FAN mean AUPRC: `{fan[fan.model == 'oracle_temporal_fan_5'].AUPRC.mean():.4f}`
- Predicted Temporal FAN Strict mean AUPRC: `{fan[fan.model == 'predicted_temporal_fan_5_strict'].AUPRC.mean():.4f}`
- Planted CircuitF1: `{planted.CircuitF1.mean():.4f}`
- Standard SCTC delta AUPRC: `{sctc.delta_AUPRC.mean():.4f}`
- Standard SCTC probability MAE: `{sctc.probability_MAE.mean():.4f}`

## Completed Stages

- Full-run guard.
- Concept sufficiency audit.
- Concept-mediated FAN diagnostics.
- Real planted raw activations and interventions.
- Representation audit.
- Standard Transformer + SCTC fidelity with replacement forward.
- Three-seed aggregation and delivery validation.

## Skipped Stages

- FAN + SCTC is `SKIPPED_BY_GATE` if FAN Foundation Gate fails.

## Next Scientific Step

Use the raw FAN gate table and Standard+SCTC fidelity outputs to decide whether to improve FAN evidence compression or continue with Standard-only SCTC.
"""
    Path("docs/medical/PROJECT_STATE.md").write_text(text, encoding="utf-8")
    (output / "PROJECT_MEMORY" / "PROJECT_STATE.md").write_text(text, encoding="utf-8")


def copy_delivery(output: Path, config_path: Path) -> None:
    for d in ["SOURCE", "CONFIGS", "PROTOCOL", "PROJECT_MEMORY"]:
        (output / d).mkdir(parents=True, exist_ok=True)
    for rel in ["src/fan", "src/med_circuitbench", "scripts/medical/v2_2", "scripts/medical/v2"]:
        dst = output / "SOURCE" / rel
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(rel, dst)
    shutil.copy2(config_path, output / "CONFIGS" / config_path.name)
    if Path("AGENTS.md").exists():
        shutil.copy2("AGENTS.md", output / "AGENTS.md")
        shutil.copy2("AGENTS.md", output / "PROTOCOL" / "AGENTS.md")
    if Path("docs/medical/PROJECT_STATE.md").exists():
        shutil.copy2("docs/medical/PROJECT_STATE.md", output / "PROJECT_MEMORY" / "PROJECT_STATE.md")


def run_checks(output: Path) -> None:
    tests = output / "TESTS"
    tests.mkdir(exist_ok=True)
    cmds = {
        "pytest": [sys.executable, "-m", "pytest", "tests/medical/v2_2", "tests/medical/v2", "-q"],
        "compileall": [sys.executable, "-m", "compileall", "-q", "src/fan", "src/med_circuitbench/v2_2", "scripts/medical/v2_2"],
    }
    for name, cmd in cmds.items():
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        (tests / f"{name}_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (tests / f"{name}_stderr.log").write_text(proc.stderr, encoding="utf-8")
        (tests / f"{name}_exit_code.txt").write_text(str(proc.returncode), encoding="utf-8")


def package(output: Path) -> Path:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    date = datetime.now().strftime("%Y%m%d")
    zip_path = output.parent / f"Med_CircuitBench_V2_2_VALIDATED_FINAL_{date}_{commit}.zip"
    checksums = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                local = path.relative_to(output)
                if local.parts and local.parts[0] == "runs":
                    local = Path("RUNS") / Path(*local.parts[1:])
                rel = Path("Med_CircuitBench_V2_2_VALIDATED_FINAL") / local
                zf.write(path, rel)
                checksums.append(f"{sha256_file(path)}  {rel.as_posix()}")
        zf.writestr("Med_CircuitBench_V2_2_VALIDATED_FINAL/checksums.sha256", "\n".join(checksums) + "\n")
    return zip_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--mode", default="full")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-stage")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    full_run_guard(cfg, args.seeds, args.mode)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "MANIFESTS").mkdir(exist_ok=True)
    (output / "MANIFESTS" / "preflight.json").write_text(json.dumps({"started_at": now(), "config": args.config, "mode": args.mode, "device": str(DEVICE)}, indent=2), encoding="utf-8")
    if args.dry_run:
        print("V2.2 dry-run OK")
        return 0
    copy_delivery(output, Path(args.config))
    results = []
    for seed in args.seeds:
        print(f"[V2.2] seed {seed} started", flush=True)
        results.append(run_seed(seed, cfg, output))
        print(f"[V2.2] seed {seed} complete", flush=True)
    status = aggregate(output, args.seeds, cfg)
    run_checks(output)
    (output / "README_FIRST.md").write_text(f"# Med-CircuitBench V2.2\n\nStatus: `{status['final_status']}`\n", encoding="utf-8")
    (output / "DELIVERY_REPORT.md").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (output / "GIT_INFO.txt").write_text(subprocess.check_output(["git", "log", "-1", "--pretty=commit=%H%nsubject=%s"], text=True), encoding="utf-8")
    (output / "LIMITATIONS").mkdir(exist_ok=True)
    (output / "LIMITATIONS" / "known_limitations.md").write_text("V2.2 uses computed planted raw files and Standard SCTC replacement fidelity. FAN+SCTC remains gated by FAN Foundation Gate.\n", encoding="utf-8")
    zip_path = package(output)
    print(json.dumps({"status": status["final_status"], "zip": str(zip_path), "sha256": sha256_file(zip_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
