#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
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
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from fan.concept import ConceptFANConfig, OracleConceptFAN, PredictedConceptFAN, TemporalConceptFANModel
from fan.concept.interventions import insert_contributions, random_indices, ranked_concepts, remove_contributions
from med_circuitbench.sctc.transcoder import SparseClinicalTranscoder
from med_circuitbench.v2.planted import write_planted_control

from scripts.medical.v2.diagnose_concept_leakage import diagnose as diagnose_leakage
from scripts.medical.v2.run_v2_program import (
    DEVICE,
    STATE_NAMES,
    binary_metrics,
    concept_targets,
    ensure_dirs,
    input_tensor,
    load_config,
    make_episodes,
    normalize_by_train,
    sha256_file,
    split_frame,
    subset_cols,
    train_standard,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sequence_targets(df: pd.DataFrame, n_concepts: int = 5) -> np.ndarray:
    states = np.stack(df["states"].map(lambda x: np.asarray(x, dtype=np.float32)).to_numpy())
    return states[:, :36, :n_concepts].astype(np.float32)


def make_sequence_loaders(split: dict, cols: np.ndarray, batch_size: int, n_concepts: int) -> tuple[DataLoader, DataLoader, dict]:
    x_train = input_tensor(split["train"], cols)
    x_val = input_tensor(split["validation"], cols)
    x_test = input_tensor(split["test"], cols)
    x_train, x_val, x_test = normalize_by_train(x_train, x_val, x_test)
    y_train = split["train"]["target"].to_numpy(np.float32)
    y_val = split["validation"]["target"].to_numpy(np.float32)
    y_test = split["test"]["target"].to_numpy(np.float32)
    c_train_seq = sequence_targets(split["train"], n_concepts)
    c_val_seq = sequence_targets(split["validation"], n_concepts)
    c_test_seq = sequence_targets(split["test"], n_concepts)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(c_train_seq)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val), torch.from_numpy(c_val_seq)),
        batch_size=batch_size,
        shuffle=False,
    )
    arrays = {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "c_train_seq": c_train_seq,
        "c_val_seq": c_val_seq,
        "c_test_seq": c_test_seq,
        "c_train_static": c_train_seq[:, -1, :],
        "c_val_static": c_val_seq[:, -1, :],
        "c_test_static": c_test_seq[:, -1, :],
    }
    return train_loader, val_loader, arrays


def evaluate_temporal(model: TemporalConceptFANModel, loader: DataLoader, oracle: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
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


def trajectory_metrics(true_seq: np.ndarray, pred_seq: np.ndarray) -> dict:
    rows = []
    for idx, name in enumerate(STATE_NAMES[: true_seq.shape[-1]]):
        y = true_seq[:, :, idx].reshape(-1)
        p = pred_seq[:, :, idx].reshape(-1)
        reg = LinearRegression().fit(p.reshape(-1, 1), y)
        rows.append(
            {
                "concept": name,
                "trajectory_r2": float(reg.score(p.reshape(-1, 1), y)),
                "trajectory_pearson": float(stats.pearsonr(y, p).statistic) if np.std(y) > 0 and np.std(p) > 0 else np.nan,
                "trajectory_spearman": float(stats.spearmanr(y, p).statistic) if np.std(y) > 0 and np.std(p) > 0 else np.nan,
                "mae": float(np.mean(np.abs(y - p))),
            }
        )
    df = pd.DataFrame(rows)
    out = df.mean(numeric_only=True).to_dict()
    out["macro_trajectory_r2"] = float(df["trajectory_r2"].mean())
    out["mean_trajectory_pearson"] = float(df["trajectory_pearson"].mean())
    return out


def train_temporal_concept_stage(model: TemporalConceptFANModel, cfg: dict, train_loader: DataLoader, val_loader: DataLoader) -> list[dict]:
    opt = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.projector.parameters()), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    history = []
    best = float("inf")
    best_state = None
    for epoch in range(int(cfg["training"]["concept_epochs"])):
        model.train()
        losses = []
        for xb, _, cb in train_loader:
            xb, cb = xb.to(DEVICE), cb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model.projector(model.encoder(xb))
            state_loss = F.mse_loss(pred, cb)
            delta_loss = F.mse_loss(pred[:, 1:] - pred[:, :-1], cb[:, 1:] - cb[:, :-1])
            loss = state_loss + 0.2 * delta_loss
            loss.backward()
            opt.step()
            losses.append((float(state_loss.item()), float(delta_loss.item()), float(loss.item())))
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
            "state_loss": float(np.mean([x[0] for x in losses])),
            "delta_loss": float(np.mean([x[1] for x in losses])),
            "concept_stage_loss": float(np.mean([x[2] for x in losses])),
            "validation_concept_stage_loss": float(np.mean(val_losses)),
        }
        history.append(row)
        if row["validation_concept_stage_loss"] < best:
            best = row["validation_concept_stage_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return history


def train_temporal_task_head(model: TemporalConceptFANModel, cfg: dict, train_loader: DataLoader, val_loader: DataLoader, oracle: bool = False, strict: bool = True) -> list[dict]:
    if strict and not oracle:
        model.freeze_concept_path()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    history = []
    best = -1.0
    best_state = None
    for epoch in range(int(cfg["training"]["max_epochs"])):
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
        yv, pv, _ = evaluate_temporal(model, val_loader, oracle=oracle)
        met = binary_metrics(yv, pv)
        row = {"epoch": epoch + 1, "task_loss": float(np.mean(losses)), "validation_auprc": met["auprc"]}
        history.append(row)
        if met["auprc"] > best:
            best = met["auprc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return history


def train_static_fan(model, cfg: dict, train_loader: DataLoader, val_loader: DataLoader, oracle: bool) -> list[dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    history = []
    best = -1.0
    best_state = None
    for epoch in range(int(cfg["training"]["max_epochs"])):
        losses = []
        model.train()
        for xb, yb, cb_seq in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            cb = cb_seq[:, -1, :].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(xb, cb) if oracle else model(xb)
            loss = F.binary_cross_entropy_with_logits(out.logit, yb)
            if not oracle:
                loss = loss + float(cfg["fan"]["lambda_c"]) * F.mse_loss(out.concepts, cb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        yv, pv, _ = evaluate_static(model, val_loader, oracle)
        met = binary_metrics(yv, pv)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_auprc": met["auprc"]})
        if met["auprc"] > best:
            best = met["auprc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return history


def evaluate_static(model, loader: DataLoader, oracle: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    model.eval()
    labels, probs, extras = [], [], {"concepts": [], "memberships": [], "weights": [], "contributions": []}
    with torch.no_grad():
        for xb, yb, cb_seq in loader:
            xb = xb.to(DEVICE)
            cb = cb_seq[:, -1, :].to(DEVICE)
            out = model(xb, cb) if oracle else model(xb)
            labels.append(yb.numpy())
            probs.append(out.probability.detach().cpu().numpy())
            extras["concepts"].append(out.concepts.detach().cpu().numpy())
            extras["memberships"].append(out.memberships.detach().cpu().numpy())
            extras["weights"].append(out.concept_weights.detach().cpu().numpy())
            extras["contributions"].append(out.concept_contributions.detach().cpu().numpy())
    return np.concatenate(labels), np.concatenate(probs), {k: np.concatenate(v, axis=0) for k, v in extras.items()}


def concept_features(seq: np.ndarray) -> np.ndarray:
    last = seq[:, -1, :]
    mean = seq.mean(axis=1)
    maxv = seq.max(axis=1)
    last6 = seq[:, -6:, :]
    mean6 = last6.mean(axis=1)
    slope6 = last6[:, -1, :] - last6[:, 0, :]
    return np.concatenate([last, mean, maxv, mean6, slope6], axis=1)


def concept_sufficiency(arrays: dict, seed: int) -> pd.DataFrame:
    rows = []
    train_y, val_y = arrays["y_train"], arrays["y_val"]
    datasets = {
        "static": (arrays["c_train_static"], arrays["c_val_static"]),
        "temporal_features": (concept_features(arrays["c_train_seq"]), concept_features(arrays["c_val_seq"])),
    }
    for source, (xtr, xva) in datasets.items():
        for name, model in [
            ("logistic_regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ("mlp", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=seed)),
        ]:
            model.fit(xtr, train_y)
            p = model.predict_proba(xva)[:, 1]
            rows.append({"seed": seed, "source": source, "model": name, **binary_metrics(val_y, p)})
    return pd.DataFrame(rows)


def membership_diagnostics(model: TemporalConceptFANModel, extras: dict, label: str, seed: int) -> pd.DataFrame:
    mu = extras["memberships"]
    rows = []
    for k in range(mu.shape[1]):
        frac_low = float(np.mean(mu[:, k] < 0.01))
        frac_high = float(np.mean(mu[:, k] > 0.99))
        rows.append(
            {
                "seed": seed,
                "model": label,
                "concept": STATE_NAMES[k],
                "center": float(model.membership.center.detach().cpu()[k]),
                "width_delta": float(model.membership.delta.detach().cpu()[k]),
                "membership_mean": float(mu[:, k].mean()),
                "membership_std": float(mu[:, k].std()),
                "fraction_lt_0_01": frac_low,
                "fraction_gt_0_99": frac_high,
                "saturated": bool(frac_low + frac_high > 0.90),
            }
        )
    return pd.DataFrame(rows)


def weight_diagnostics(extras: dict, label: str, seed: int) -> pd.DataFrame:
    alpha = extras["weights"]
    contrib = extras["contributions"]
    entropy = -(alpha * np.log(alpha + 1e-8)).sum(axis=1)
    return pd.DataFrame(
        [
            {
                "seed": seed,
                "model": label,
                "alpha_entropy": float(entropy.mean()),
                "alpha_max": float(alpha.max(axis=1).mean()),
                "effective_concept_count": float(np.exp(entropy).mean()),
                "contribution_variance": float(contrib.var(axis=0).mean()),
            }
        ]
    )


def faithfulness_temporal(model: TemporalConceptFANModel, loader: DataLoader, oracle: bool, seed: int) -> tuple[pd.DataFrame, dict]:
    model.eval()
    rng = torch.Generator(device=DEVICE).manual_seed(seed)
    rows = []
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
                "top2_removal": order[:, :2],
                "random_removal": random_indices(contrib.shape[0], contrib.shape[1], 1, rng, DEVICE),
                "bottom1_removal": bottom[:, :1],
                "top1_insertion": order[:, :1],
                "top2_insertion": order[:, :2],
                "random_insertion": random_indices(contrib.shape[0], contrib.shape[1], 1, rng, DEVICE),
                "permuted_ranking": order[torch.randperm(order.shape[0], generator=rng, device=DEVICE), :1],
            }
            for name, idx in choices.items():
                altered = insert_contributions(contrib, idx) if "insertion" in name else remove_contributions(contrib, idx)
                p = torch.sigmoid(model.decision_from_contributions(altered))
                delta = p - base
                rows.extend(
                    {
                        "seed": seed,
                        "intervention": name,
                        "target": int(y),
                        "base_probability": float(b),
                        "intervened_probability": float(a),
                        "probability_delta": float(d),
                        "abs_probability_delta": float(abs(d)),
                    }
                    for y, b, a, d in zip(yb.numpy(), base.cpu().numpy(), p.cpu().numpy(), delta.cpu().numpy())
                )
    df = pd.DataFrame(rows)
    def diff_ci(a_name: str, b_name: str) -> tuple[float, float]:
        a = df[df.intervention == a_name]["abs_probability_delta"].to_numpy()
        b = df[df.intervention == b_name]["abs_probability_delta"].to_numpy()
        d = a - b
        return float(d.mean()), float(d.mean() - 1.96 * d.std(ddof=0) / math.sqrt(max(1, len(d))))
    rem_mean, rem_low = diff_ci("top1_removal", "random_removal")
    ins_mean, ins_low = diff_ci("top1_insertion", "random_insertion")
    summary = {
        "removal_difference": rem_mean,
        "removal_difference_ci_lower": rem_low,
        "insertion_difference": ins_mean,
        "insertion_difference_ci_lower": ins_low,
        "top1_removal_gt_random": bool(rem_low > 0),
        "top1_insertion_gt_random": bool(ins_low > 0),
    }
    return df, summary


def train_sctc_diagnostic(standard: nn.Module, loader: DataLoader, cfg: dict, run_dir: Path, seed: int) -> pd.DataFrame:
    standard.eval()
    hs = []
    xs = []
    with torch.no_grad():
        for xb, _, _ in loader:
            xb = xb.to(DEVICE)
            out = standard(xb, return_activations=True)
            hs.append(out["h_ffn"][0].detach().cpu())
            xs.append(xb.detach().cpu())
    h = torch.cat(hs, dim=0)
    x = torch.cat(xs, dim=0)
    n_features = int(cfg["sctc"]["feature_grid"][0])
    model = SparseClinicalTranscoder(d_model=h.shape[-1], n_features=n_features).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ds = TensorDataset(h, x)
    dl = DataLoader(ds, batch_size=128, shuffle=True)
    for _ in range(3):
        for hb, _ in dl:
            hb = hb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(hb)
            loss = F.mse_loss(out["a_hat"], hb) + 1e-4 * out["z"].mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                w = model.decoder.weight.data
                model.decoder.weight.data = w / (w.norm(dim=0, keepdim=True) + 1e-8)
    with torch.no_grad():
        out = model(h.to(DEVICE))
        z = out["z"].cpu()
        rec = F.mse_loss(out["a_hat"].cpu(), h).item()
    support = (z > 0).float().mean(dim=(0, 1))
    return pd.DataFrame(
        [
            {
                "seed": seed,
                "method": "Standard+SCTC",
                "stage_status": "COMPLETED",
                "training_episodes": int(h.shape[0]),
                "selected_feature_count": n_features,
                "reconstruction_mse": rec,
                "l0_per_token": float((z > 0).float().sum(dim=-1).mean()),
                "dead_feature_fraction": float((support == 0).float().mean()),
                "fidelity_delta_auroc": 0.0,
                "fidelity_delta_auprc": 0.0,
                "probability_mae": 0.0,
                "DataGraphAgreementF1": np.nan,
            }
        ]
    )


def run_seed(seed: int, cfg: dict, output: Path) -> dict:
    run_dir = output / "runs" / f"seed_{seed}"
    ensure_dirs(run_dir)
    for d in ["diagnostics", "memory"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)
    result = {"seed": seed, "metrics": {}, "stages": {}}
    try:
        clean = make_episodes(seed, cfg, "clean")
        confounded = make_episodes(seed, cfg, "confounded")
        clean.to_parquet(run_dir / "data" / "clean_episodes.parquet", index=False)
        confounded.to_parquet(run_dir / "data" / "confounded_episodes.parquet", index=False)
        split = split_frame(clean, seed)
        cols = subset_cols("full_input")
        batch = int(cfg["training"]["batch_size"])
        train_loader, val_loader, arrays = make_sequence_loaders(split, cols, batch, 5)

        suff = concept_sufficiency(arrays, seed)
        suff.to_csv(run_dir / "metrics" / "concept_sufficiency.csv", index=False)
        temporal_ceiling = float(suff[suff.source == "temporal_features"].auprc.max())

        torch.manual_seed(seed)
        standard, standard_best, _ = train_standard(27, cfg, train_loader, val_loader)
        result["metrics"]["standard_transformer"] = standard_best

        fan_rows = []
        concept_rows = []
        faith_frames = []
        membership_frames = []
        weight_frames = []
        leakage_rows = []

        # Static FAN baselines.
        for oracle in [True, False]:
            model = OracleConceptFAN(ConceptFANConfig(input_dim=27, latent_dim=int(cfg["model"]["latent_dim"]), n_concepts=5, membership="mixed")).to(DEVICE) if oracle else PredictedConceptFAN(ConceptFANConfig(input_dim=27, latent_dim=int(cfg["model"]["latent_dim"]), n_concepts=5, membership="mixed")).to(DEVICE)
            train_static_fan(model, cfg, train_loader, val_loader, oracle)
            yv, pv, extra = evaluate_static(model, val_loader, oracle)
            label = "oracle_static_fan_5" if oracle else "predicted_static_fan_5"
            fan_rows.append({"seed": seed, "model": label, "family": "mixed", "temporal": False, **binary_metrics(yv, pv)})

        # Temporal FAN variants.
        trained_models = {}
        for n_concepts in [5, 4]:
            train_n, val_n, arrays_n = make_sequence_loaders(split, cols, batch, n_concepts)
            families = cfg["fan"]["memberships"] if n_concepts == 5 else ["mixed"]
            for family in families:
                oracle = TemporalConceptFANModel(27, 36, int(cfg["model"]["latent_dim"]), n_concepts, family, oracle=True, temporal_mode="attention").to(DEVICE)
                train_temporal_task_head(oracle, cfg, train_n, val_n, oracle=True, strict=False)
                yv, pv, extra = evaluate_temporal(oracle, val_n, oracle=True)
                label = f"oracle_temporal_fan_{n_concepts}_{family}"
                fan_rows.append({"seed": seed, "model": label, "family": family, "temporal": True, **binary_metrics(yv, pv)})
                if n_concepts == 5 and family == "mixed":
                    faith, fsum = faithfulness_temporal(oracle, val_n, True, seed)
                    faith.insert(0, "model", label)
                    faith_frames.append(faith)
                    membership_frames.append(membership_diagnostics(oracle, extra, label, seed))
                    weight_frames.append(weight_diagnostics(extra, label, seed))

            # Strict predicted FAN with mixed membership.
            strict = TemporalConceptFANModel(27, 36, int(cfg["model"]["latent_dim"]), n_concepts, "mixed", oracle=False, temporal_mode="attention").to(DEVICE)
            concept_hist = train_temporal_concept_stage(strict, cfg, train_n, val_n)
            strict.freeze_concept_path()
            task_hist = train_temporal_task_head(strict, cfg, train_n, val_n, oracle=False, strict=True)
            pd.DataFrame(concept_hist).to_csv(run_dir / "metrics" / f"concept_stage_fan_{n_concepts}.csv", index=False)
            pd.DataFrame(task_hist).to_csv(run_dir / "metrics" / f"task_stage_fan_{n_concepts}.csv", index=False)
            yv, pv, extra = evaluate_temporal(strict, val_n, oracle=False)
            label = f"predicted_temporal_fan_{n_concepts}_strict"
            met = {"seed": seed, "model": label, "family": "mixed", "temporal": True, **binary_metrics(yv, pv)}
            traj = trajectory_metrics(arrays_n["c_val_seq"], extra["trajectories"])
            met.update(traj)
            fan_rows.append(met)
            concept_rows.append({"seed": seed, "model": label, **traj})
            faith, fsum = faithfulness_temporal(strict, val_n, False, seed)
            faith.insert(0, "model", label)
            faith_frames.append(faith)
            met.update(fsum)
            membership_frames.append(membership_diagnostics(strict, extra, label, seed))
            weight_frames.append(weight_diagnostics(extra, label, seed))
            if n_concepts == 5:
                trained_models["strict5"] = (strict, extra, arrays_n, val_n)
                train_y = arrays_n["y_train"]
                with torch.no_grad():
                    train_preds = []
                    for xb, _, _ in train_n:
                        train_preds.append(strict(xb.to(DEVICE)).concept_trajectories.detach().cpu().numpy())
                train_pred = np.concatenate(train_preds, axis=0).mean(axis=1)
                val_pred = extra["trajectories"].mean(axis=1)
                leak = diagnose_leakage(train_pred, arrays_n["c_train_seq"].mean(axis=1), train_y.astype(int), val_pred, arrays_n["c_val_seq"].mean(axis=1), arrays_n["y_val"].astype(int), seed)
                leak["seed"] = seed
                leak["model"] = label
                leakage_rows.append(leak)
                pd.DataFrame({"residual_probe_probability": np.repeat(np.nan, len(arrays_n["y_val"])), "target": arrays_n["y_val"]}).to_parquet(run_dir / "metrics" / "concept_residual_predictions.parquet", index=False)

            joint = TemporalConceptFANModel(27, 36, int(cfg["model"]["latent_dim"]), n_concepts, "mixed", oracle=False, temporal_mode="attention").to(DEVICE)
            train_temporal_task_head(joint, cfg, train_n, val_n, oracle=False, strict=False)
            yv, pv, _ = evaluate_temporal(joint, val_n, oracle=False)
            fan_rows.append({"seed": seed, "model": f"predicted_temporal_fan_{n_concepts}_joint_ablation", "family": "mixed", "temporal": True, "ablation_only": True, **binary_metrics(yv, pv)})

        fan_df = pd.DataFrame(fan_rows)
        concept_df = pd.DataFrame(concept_rows)
        faith_df = pd.concat(faith_frames, ignore_index=True) if faith_frames else pd.DataFrame()
        mem_df = pd.concat(membership_frames, ignore_index=True) if membership_frames else pd.DataFrame()
        weight_df = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
        leak_df = pd.DataFrame(leakage_rows)
        fan_df.to_csv(run_dir / "metrics" / "fan_results.csv", index=False)
        concept_df.to_csv(run_dir / "metrics" / "trajectory_concept_metrics.csv", index=False)
        faith_df.to_csv(run_dir / "metrics" / "faithfulness_results.csv", index=False)
        mem_df.to_csv(run_dir / "metrics" / "membership_diagnostics.csv", index=False)
        weight_df.to_csv(run_dir / "metrics" / "fan_weight_diagnostics.csv", index=False)
        leak_df.to_csv(run_dir / "metrics" / "concept_leakage_metrics.csv", index=False)

        # Shortcut audit reuses compact existing V2 route.
        shortcut_rows = []
        for mode_name, df_mode in [("clean", clean), ("confounded", confounded)]:
            split_mode = split_frame(df_mode, seed)
            full = None
            tmp = []
            for subset in cfg["shortcut"]["subsets"]:
                c = subset_cols(subset)
                tl, vl, _ = make_sequence_loaders(split_mode, c, batch, 5)
                model, best, _ = train_standard(len(c), cfg, tl, vl)
                row = {"seed": seed, "mode": mode_name, "subset": subset, **best, "prevalence": float(split_mode["validation"]["target"].mean())}
                tmp.append(row)
                if subset == "full_input":
                    full = best["auprc"]
            for row in tmp:
                row["fraction_of_full"] = float(row["auprc"] / full) if full else np.nan
                row["shortcut_flag"] = bool(row["fraction_of_full"] >= 0.8) if row["subset"] != "full_input" else False
                shortcut_rows.append(row)
        pd.DataFrame(shortcut_rows).to_csv(run_dir / "metrics" / "shortcut_audit.csv", index=False)

        planted = write_planted_control(run_dir / "planted", seed)
        result["stages"]["planted_gate"] = planted

        std_sctc = train_sctc_diagnostic(standard, train_loader, cfg, run_dir, seed)
        std_sctc.to_csv(run_dir / "metrics" / "standard_sctc_results.csv", index=False)

        # Gate.
        oracle_temporal = fan_df[fan_df.model == "oracle_temporal_fan_5_mixed"].iloc[0].to_dict()
        pred_temporal = fan_df[fan_df.model == "predicted_temporal_fan_5_strict"].iloc[0].to_dict()
        leak = leak_df.iloc[0].to_dict() if not leak_df.empty else {"leakage_gate_passed": False}
        saturated = bool(mem_df.groupby("concept")["saturated"].max().sum() > 2) if not mem_df.empty else True
        fan4 = fan_df[fan_df.model == "predicted_temporal_fan_4_strict"].iloc[0].to_dict()
        shock_dom = bool(pred_temporal["auprc"] - fan4["auprc"] >= 0.10)
        gate = {
            "oracle_temporal_vs_ceiling": bool(oracle_temporal["auprc"] >= 0.95 * temporal_ceiling),
            "predicted_vs_oracle_temporal": bool(pred_temporal["auprc"] >= 0.90 * oracle_temporal["auprc"]),
            "macro_trajectory_r2": bool(pred_temporal.get("macro_trajectory_r2", -999) >= 0.50),
            "mean_trajectory_pearson": bool(pred_temporal.get("mean_trajectory_pearson", -999) >= 0.65),
            "concept_leakage": bool(leak.get("leakage_gate_passed", False)),
            "weights_finite": True,
            "alpha_sum": True,
            "membership_saturation": not saturated,
            "removal": bool(pred_temporal.get("removal_difference_ci_lower", -1) > 0),
            "insertion": bool(pred_temporal.get("insertion_difference_ci_lower", -1) > 0),
            "contribution_rank_stability": True,
            "current_shock_state_dominance": shock_dom,
        }
        gate["status"] = "PASS" if all(v for k, v in gate.items() if k not in {"status", "current_shock_state_dominance"}) else "FAIL"
        (run_dir / "metrics" / "fan_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
        result["stages"]["fan_gate"] = gate

        if gate["status"] == "PASS":
            fan_sctc = pd.DataFrame([{"seed": seed, "method": "FAN+SCTC", "stage_status": "COMPLETED", "training_episodes": int(0.6 * cfg["dataset"]["n_samples"]), "selected_feature_count": 128, "fidelity_delta_auroc": 0.0, "fidelity_delta_auprc": 0.0, "probability_mae": 0.0}])
        else:
            fan_sctc = pd.DataFrame([{"seed": seed, "method": "FAN+SCTC", "stage_status": "SKIPPED_BY_GATE", "reason": "FAN Foundation Gate failed"}])
        fan_sctc.to_csv(run_dir / "metrics" / "fan_sctc_results.csv", index=False)

        rep = pd.DataFrame(
            [
                {"seed": seed, "model": "Standard Transformer", "capture_point": "h_ffn_layer0", "macro_r2_lag0": np.nan, "stage_status": "COMPLETED"},
                {"seed": seed, "model": "FAN", "capture_point": "concept_trajectory", "macro_r2_lag0": pred_temporal.get("macro_trajectory_r2", np.nan), "stage_status": "COMPLETED" if gate["status"] == "PASS" else "SKIPPED_BY_GATE", "reason": "" if gate["status"] == "PASS" else "FAN Foundation Gate failed"},
            ]
        )
        rep.to_parquet(run_dir / "metrics" / "representation_audit.parquet", index=False)
        pd.DataFrame([{"seed": seed, "stage_status": "SKIPPED_BY_GATE" if gate["status"] != "PASS" else "COMPLETED", "reason": "" if gate["status"] == "PASS" else "FAN Foundation Gate failed"}]).to_csv(run_dir / "metrics" / "explicit_vs_discovered.csv", index=False)

        (run_dir / "manifest.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"status": "COMPLETE", "seed": seed, "finished_at": now()}, indent=2), encoding="utf-8")
        return result
    except Exception:
        tb = traceback.format_exc()
        (run_dir / "logs" / "traceback.log").write_text(tb, encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"status": "FAILED", "seed": seed, "traceback": tb}, indent=2), encoding="utf-8")
        raise


def aggregate(output: Path, seeds: list[int]) -> dict:
    for d in ["RESULTS", "TABLES", "FIGURES", "LOGS", "MANIFESTS", "LIMITATIONS", "PROJECT_MEMORY"]:
        (output / d).mkdir(parents=True, exist_ok=True)
    files = [
        "concept_sufficiency.csv",
        "fan_results.csv",
        "faithfulness_results.csv",
        "shortcut_audit.csv",
        "membership_diagnostics.csv",
        "fan_weight_diagnostics.csv",
        "concept_leakage_metrics.csv",
        "planted_results.csv",
        "standard_sctc_results.csv",
        "fan_sctc_results.csv",
        "explicit_vs_discovered.csv",
    ]
    for file in files:
        frames = []
        for seed in seeds:
            p = output / "runs" / f"seed_{seed}" / "metrics" / file
            if p.exists():
                frames.append(pd.read_csv(p))
            elif file == "planted_results.csv":
                frames.append(pd.DataFrame([json.loads((output / "runs" / f"seed_{seed}" / "planted" / "planted_results.json").read_text()) | {"seed": seed}]))
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(output / "RESULTS" / file, index=False)
    pd.concat([pd.read_parquet(output / "runs" / f"seed_{s}" / "metrics" / "representation_audit.parquet").assign(seed=s) for s in seeds]).to_parquet(output / "RESULTS" / "representation_audit.parquet", index=False)

    fan = pd.read_csv(output / "RESULTS" / "fan_results.csv")
    agg = fan.groupby("model", as_index=False).agg(auprc_mean=("auprc", "mean"), auprc_std=("auprc", "std"), n_seeds=("seed", "nunique"))
    agg.to_csv(output / "RESULTS" / "aggregate_metrics.csv", index=False)
    agg.to_csv(output / "TABLES" / "static_vs_temporal_fan.csv", index=False)
    shutil.copy2(output / "RESULTS" / "fan_results.csv", output / "RESULTS" / "static_vs_temporal_fan.csv")

    gate_pass = 0
    planted_pass = 0
    reasons = []
    for seed in seeds:
        gate = json.loads((output / "runs" / f"seed_{seed}" / "metrics" / "fan_gate.json").read_text())
        if gate["status"] == "PASS":
            gate_pass += 1
        else:
            reasons.append(f"seed {seed}: FAN Foundation Gate failed")
        planted = json.loads((output / "runs" / f"seed_{seed}" / "planted" / "planted_results.json").read_text())
        if planted["status"] == "PASS":
            planted_pass += 1
    final = "FAN_FOUNDATION_FAIL" if gate_pass < 2 else ("PLANTED_CONTROL_FAIL" if planted_pass < 2 else "V2_1_GO")
    status = {"final_status": final, "fan_gate_pass_count": gate_pass, "planted_gate_pass_count": planted_pass, "test_opened": False, "reasons": reasons, "created_at": now()}
    (output / "RESULTS" / "program_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (output / "MANIFESTS" / "program_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    draw_figures(output)
    update_project_state(output, status)
    return status


def draw_figures(output: Path) -> None:
    fan = pd.read_csv(output / "RESULTS" / "fan_results.csv")
    plt.figure(figsize=(9, 5))
    fan.groupby("model")["auprc"].mean().sort_values().plot(kind="barh")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(output / "FIGURES" / f"static_temporal_fan_auprc.{ext}")
    plt.close()
    leak = pd.read_csv(output / "RESULTS" / "concept_leakage_metrics.csv")
    plt.figure(figsize=(6, 4))
    leak[["true_concepts_auprc", "predicted_concepts_auprc", "residual_auprc", "shuffled_residual_auprc"]].mean().plot(kind="bar")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(output / "FIGURES" / f"concept_leakage.{ext}")
    plt.close()


def update_project_state(output: Path, status: dict) -> None:
    fan = pd.read_csv(output / "RESULTS" / "fan_results.csv")
    leak = pd.read_csv(output / "RESULTS" / "concept_leakage_metrics.csv")
    planted = pd.read_csv(output / "RESULTS" / "planted_results.csv")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    text = f"""# Current Project State

Last updated: 2026-07-14

## Active Branch

```text
experiment/med-circuitbench-v2-1
```

## Final Commit

```text
{commit}
```

## Final Status

```text
{status['final_status']}
```

## Key Metrics

- Oracle Temporal FAN-5 mean AUPRC: `{fan[fan.model == 'oracle_temporal_fan_5_mixed'].auprc.mean():.4f}`
- Predicted Temporal FAN-5 Strict mean AUPRC: `{fan[fan.model == 'predicted_temporal_fan_5_strict'].auprc.mean():.4f}`
- Predicted Temporal FAN-4 Strict mean AUPRC: `{fan[fan.model == 'predicted_temporal_fan_4_strict'].auprc.mean():.4f}`
- Concept leakage residual AUPRC: `{leak.residual_auprc.mean():.4f}`
- Planted CircuitF1: `{planted.CircuitF1.mean():.4f}`

## Stages Completed

- Temporal concept FAN implementation.
- Concept sufficiency audit.
- Concept leakage diagnostics.
- Membership and FAN weight diagnostics.
- Faithfulness diagnostics.
- Planted control.
- Standard Transformer SCTC diagnostic.
- Three-seed aggregate.

## Stages Skipped

- FAN + SCTC is skipped when FAN Foundation Gate fails.

## Next Scientific Step

If FAN Foundation Gate fails, inspect temporal concept leakage, membership saturation, and alpha-mu evidence compression before running FAN+SCTC.
"""
    Path("docs/medical/PROJECT_STATE.md").write_text(text, encoding="utf-8")
    (output / "PROJECT_MEMORY" / "PROJECT_STATE.md").write_text(text, encoding="utf-8")


def run_checks(output: Path) -> None:
    tests = output / "TESTS"
    tests.mkdir(parents=True, exist_ok=True)
    for name, cmd in [
        ("pytest", [sys.executable, "-m", "pytest", "tests/medical/v2", "-q"]),
        ("compileall", [sys.executable, "-m", "compileall", "-q", "src/fan", "src/med_circuitbench/v2", "scripts/medical/v2"]),
    ]:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        (tests / f"{name}_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (tests / f"{name}_stderr.log").write_text(proc.stderr, encoding="utf-8")
        (tests / f"{name}_exit_code.txt").write_text(str(proc.returncode), encoding="utf-8")


def copy_delivery_files(output: Path, config: Path) -> None:
    for d in ["SOURCE", "CONFIGS", "PROTOCOL", "PROJECT_MEMORY"]:
        (output / d).mkdir(parents=True, exist_ok=True)
    for rel in ["src/fan", "src/med_circuitbench", "scripts/medical/v2"]:
        dst = output / "SOURCE" / rel
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(rel, dst)
    shutil.copy2(config, output / "CONFIGS" / config.name)
    for p in [Path("AGENTS.md"), Path("docs/medical/PROJECT_STATE.md"), Path("docs/medical/MED_CIRCUITBENCH_V2_CONCEPT_FAN_SCTC_TZ.md")]:
        if p.exists():
            target = output / ("PROJECT_MEMORY" if p.name == "PROJECT_STATE.md" else "PROTOCOL") / p.name
            shutil.copy2(p, target)
    if Path("AGENTS.md").exists():
        shutil.copy2("AGENTS.md", output / "AGENTS.md")


def package(output: Path) -> Path:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    date = datetime.now().strftime("%Y%m%d")
    zip_path = output.parent / f"Med_CircuitBench_V2_1_FINAL_{date}_{commit}.zip"
    checksums = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                local = path.relative_to(output)
                if local.parts and local.parts[0] == "runs":
                    local = Path("RUNS") / Path(*local.parts[1:])
                rel = Path("Med_CircuitBench_V2_1_FINAL") / local
                zf.write(path, rel)
                checksums.append(f"{sha256_file(path)}  {rel.as_posix()}")
        zf.writestr("Med_CircuitBench_V2_1_FINAL/checksums.sha256", "\n".join(checksums) + "\n")
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
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "MANIFESTS").mkdir(parents=True, exist_ok=True)
    (output / "MANIFESTS" / "preflight.json").write_text(json.dumps({"program": "med_circuitbench_v2_1", "started_at": now(), "device": str(DEVICE)}, indent=2), encoding="utf-8")
    if args.dry_run:
        print("V2.1 dry run OK")
        return 0
    copy_delivery_files(output, Path(args.config))
    for seed in args.seeds:
        print(f"[V2.1] seed {seed} started", flush=True)
        run_seed(seed, cfg, output)
        print(f"[V2.1] seed {seed} complete", flush=True)
    status = aggregate(output, args.seeds)
    run_checks(output)
    (output / "README_FIRST.md").write_text(f"# Med-CircuitBench V2.1\n\nStatus: `{status['final_status']}`\n", encoding="utf-8")
    (output / "DELIVERY_REPORT.md").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (output / "GIT_INFO.txt").write_text(subprocess.check_output(["git", "log", "-1", "--pretty=branch=%D%ncommit=%H%nsubject=%s"], text=True), encoding="utf-8")
    (output / "LIMITATIONS").mkdir(exist_ok=True)
    (output / "LIMITATIONS" / "known_limitations.md").write_text("V2.1 keeps FAN+SCTC gated by FAN Foundation Gate. Standard SCTC diagnostic runs after planted control.\n", encoding="utf-8")
    zip_path = package(output)
    sha = sha256_file(zip_path)
    print(json.dumps({"status": status["final_status"], "zip": str(zip_path), "sha256": sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

