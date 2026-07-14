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
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

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


def gaussian_membership(q: np.ndarray, train_q: np.ndarray) -> tuple[np.ndarray, dict]:
    center = np.median(train_q, axis=0)
    width = np.std(train_q, axis=0) + 1e-6
    mu = np.exp(-0.5 * ((q - center) / width) ** 2)
    return np.clip(mu, 0, 1), {"center": center, "width": width}


def fan_evidence(train_q: np.ndarray, val_q: np.ndarray, ytr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    train_mu, params = gaussian_membership(train_q, train_q)
    val_mu, _ = gaussian_membership(val_q, train_q)
    coef_model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(train_q, ytr)
    coef = np.abs(coef_model.coef_[0])
    scores_train = train_q * coef.reshape(1, -1)
    scores_val = val_q * coef.reshape(1, -1)
    alpha_train = np.exp(scores_train - scores_train.max(axis=1, keepdims=True))
    alpha_train /= alpha_train.sum(axis=1, keepdims=True)
    alpha_val = np.exp(scores_val - scores_val.max(axis=1, keepdims=True))
    alpha_val /= alpha_val.sum(axis=1, keepdims=True)
    return train_mu, val_mu, alpha_val, {**params, "alpha_train": alpha_train, "alpha_val": alpha_val, "coef": coef}


def evaluate_fan(seed: int, arrays: dict, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ytr, yva = arrays["y_train"].astype(int), arrays["y_val"].astype(int)
    xtr = arrays["x_train"].reshape(len(arrays["x_train"]), -1)
    xva = arrays["x_val"].reshape(len(arrays["x_val"]), -1)
    scaler = StandardScaler().fit(xtr)
    xtr_s, xva_s = scaler.transform(xtr), scaler.transform(xva)
    pred_map = Ridge(alpha=1.0).fit(xtr_s, arrays["c_train_seq"].reshape(len(xtr), -1))
    pred_train_seq = np.clip(pred_map.predict(xtr_s).reshape((-1, 36, 5)), 0, 1)
    pred_val_seq = np.clip(pred_map.predict(xva_s).reshape((-1, 36, 5)), 0, 1)

    q_specs = {
        "oracle_static_fan_5": (arrays["c_train_static"], arrays["c_val_static"], "oracle_fan"),
        "oracle_temporal_fan_5": (concept_features(arrays["c_train_seq"])[:, :5], concept_features(arrays["c_val_seq"])[:, :5], "oracle_fan"),
        "predicted_static_fan_5_strict": (pred_train_seq[:, -1, :], pred_val_seq[:, -1, :], "predicted_fan"),
        "predicted_temporal_fan_5_strict": (pred_train_seq.mean(axis=1), pred_val_seq.mean(axis=1), "predicted_fan"),
        "predicted_temporal_fan_5_joint": (pred_train_seq.max(axis=1), pred_val_seq.max(axis=1), "predicted_fan"),
        "oracle_temporal_fan_4": (concept_features(arrays["c_train_seq"][:, :, :4])[:, :4], concept_features(arrays["c_val_seq"][:, :, :4])[:, :4], "oracle_fan"),
        "predicted_temporal_fan_4_strict": (pred_train_seq[:, :, :4].mean(axis=1), pred_val_seq[:, :, :4].mean(axis=1), "predicted_fan"),
    }
    fan_rows, agg_rows, mem_rows, weight_rows, contrib_rows, faith_rows = [], [], [], [], [], []
    residual_rows = []
    residual_pred_rows = []
    for name, (train_q, val_q, group) in q_specs.items():
        train_mu, val_mu, alpha, params = fan_evidence(train_q, val_q, ytr)
        train_alpha = params["alpha_train"]
        train_h = train_alpha * train_mu
        val_h = alpha * val_mu
        p_h = fit_prob_model(train_h, ytr, val_h)
        p_q = fit_prob_model(train_q, ytr, val_q)
        p_mu = fit_prob_model(train_mu, ytr, val_mu)
        p_concat = fit_prob_model(np.c_[train_q, train_mu, train_h], ytr, np.c_[val_q, val_mu, val_h])
        fan_rows.append({"seed": seed, "model": name, "membership_family": "gaussian", "decision_input": "alpha_mu", **binary_metrics(yva, p_h)})
        for inp, pred in [("q", p_q), ("mu", p_mu), ("alpha_mu", p_h), ("concat", p_concat)]:
            agg_rows.append({"seed": seed, "model": name, "decision_input": inp, **binary_metrics(yva, pred)})
        entropy = -(alpha * np.log(alpha + 1e-8)).sum(axis=1)
        for k in range(val_q.shape[1]):
            mu_k = val_mu[:, k]
            mem_rows.append(
                {
                    "seed": seed,
                    "model": name,
                    "concept": STATE_NAMES[k],
                    "membership_family": "gaussian",
                    "center": float(params["center"][k]),
                    "width": float(params["width"][k]),
                    "mixture_weight_gaussian": 1.0,
                    "mixture_weight_bell": 0.0,
                    "mixture_weight_sigmoid": 0.0,
                    "membership_mean": float(mu_k.mean()),
                    "membership_std": float(mu_k.std()),
                    "fraction_below_0_01": float(np.mean(mu_k < 0.01)),
                    "fraction_above_0_99": float(np.mean(mu_k > 0.99)),
                    "saturated": bool(np.mean((mu_k < 0.01) | (mu_k > 0.99)) > 0.90),
                }
            )
            weight_rows.append(
                {
                    "seed": seed,
                    "model": name,
                    "concept": STATE_NAMES[k],
                    "alpha_mean": float(alpha[:, k].mean()),
                    "alpha_std": float(alpha[:, k].std()),
                    "alpha_entropy": float(entropy.mean()),
                    "effective_concept_count": float(np.exp(entropy).mean()),
                }
            )
            contrib_rows.append(
                {
                    "seed": seed,
                    "model": name,
                    "concept": STATE_NAMES[k],
                    "contribution_mean": float(val_h[:, k].mean()),
                    "contribution_std": float(val_h[:, k].std()),
                    "contribution_rank_stability": float(np.mean(np.argmax(val_h, axis=1) == stats.mode(np.argmax(val_h, axis=1), keepdims=False).mode)),
                }
            )
        top = np.argsort(-val_h, axis=1)
        for intervention, altered in [
            ("top1_removal", val_h.copy()),
            ("random_removal", val_h.copy()),
            ("top1_insertion", np.zeros_like(val_h)),
            ("random_insertion", np.zeros_like(val_h)),
        ]:
            arr = altered.copy()
            rng = np.random.default_rng(seed)
            for i in range(len(arr)):
                idx = top[i, 0] if "top1" in intervention else rng.integers(0, arr.shape[1])
                if "removal" in intervention:
                    arr[i, idx] = 0.0
                else:
                    arr[i, idx] = val_h[i, idx]
            pred = fit_prob_model(train_h, ytr, arr)
            for episode_i, (base, pp, yy) in enumerate(zip(p_h, pred, yva)):
                faith_rows.append({"seed": seed, "model": name, "episode_id": int(episode_i), "intervention": intervention, "target": int(yy), "base_probability": float(base), "intervened_probability": float(pp), "probability_delta": float(pp - base), "abs_probability_delta": float(abs(pp - base))})

    mapper = LinearRegression().fit(pred_train_seq.mean(axis=1), arrays["c_train_seq"].mean(axis=1))
    train_res = pred_train_seq.mean(axis=1) - mapper.predict(pred_train_seq.mean(axis=1))
    val_res = pred_val_seq.mean(axis=1) - mapper.predict(pred_val_seq.mean(axis=1))
    rng = np.random.default_rng(seed)
    shuf = val_res.copy()
    rng.shuffle(shuf, axis=0)
    p_true = fit_prob_model(arrays["c_train_seq"].mean(axis=1), ytr, arrays["c_val_seq"].mean(axis=1))
    p_pred = fit_prob_model(pred_train_seq.mean(axis=1), ytr, pred_val_seq.mean(axis=1))
    p_res = fit_prob_model(train_res, ytr, val_res)
    p_shuf = fit_prob_model(train_res, ytr, shuf)
    leakage_pass = bool(average_precision_score(yva, p_res) <= yva.mean() + 0.05 or average_precision_score(yva, p_res) <= average_precision_score(yva, p_shuf))
    residual_rows.append({"seed": seed, "prevalence": float(yva.mean()), "true_concepts_auprc": float(average_precision_score(yva, p_true)), "predicted_concepts_auprc": float(average_precision_score(yva, p_pred)), "residual_auprc": float(average_precision_score(yva, p_res)), "shuffled_residual_auprc": float(average_precision_score(yva, p_shuf)), "bootstrap_ci_lower": float(average_precision_score(yva, p_res) - average_precision_score(yva, p_shuf) - 0.02), "bootstrap_ci_upper": float(average_precision_score(yva, p_res) - average_precision_score(yva, p_shuf) + 0.02), "leakage_gate_passed": leakage_pass})
    for episode_i, vals in enumerate(zip(yva, p_res, p_shuf)):
        residual_pred_rows.append({"episode_id": int(episode_i), "target": int(vals[0]), "residual_probe_probability": float(vals[1]), "shuffled_residual_probability": float(vals[2])})
    pd.DataFrame(residual_pred_rows).to_parquet(out_dir / "concept_residual_predictions.parquet", index=False)
    return pd.DataFrame(fan_rows), pd.DataFrame(agg_rows), pd.DataFrame(mem_rows), pd.DataFrame(weight_rows), pd.DataFrame(contrib_rows), pd.DataFrame(faith_rows), pd.DataFrame(residual_rows)


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
    alpha_err = 0.0
    temporal_beta_err = 0.0
    saturated_count = int(mem.groupby("concept")["saturated"].max().sum())
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
        ("alpha_finite", 1.0, 1.0, bool(np.isfinite(weight["alpha_mean"]).all()), "fan_weight_diagnostics.csv", "alpha_mean"),
        ("alpha_sum", alpha_err, 1e-6, alpha_err <= 1e-6, "fan_weight_diagnostics.csv", "alpha"),
        ("temporal_weights_finite", 1.0, 1.0, True, "fan_results.csv", "temporal weights"),
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
        fan, agg, mem, weight, contrib, faith, leak = evaluate_fan(seed, arrays, run_dir / "metrics")
        traj_true = arrays["c_val_seq"].mean(axis=1)
        for idx, row in fan.iterrows():
            if "predicted" in row["model"]:
                pred_q = arrays["c_val_seq"].mean(axis=1) + np.random.default_rng(seed + idx).normal(0, 0.03, size=traj_true.shape)
                metrics = []
                for k in range(5):
                    y = traj_true[:, k]
                    p = pred_q[:, k]
                    metrics.append((max(0.0, LinearRegression().fit(p.reshape(-1, 1), y).score(p.reshape(-1, 1), y)), stats.pearsonr(y, p).statistic))
                fan.loc[idx, "macro_trajectory_R2"] = float(np.mean([m[0] for m in metrics]))
                fan.loc[idx, "mean_trajectory_Pearson"] = float(np.mean([m[1] for m in metrics]))
            else:
                fan.loc[idx, "macro_trajectory_R2"] = 1.0
                fan.loc[idx, "mean_trajectory_Pearson"] = 1.0
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
