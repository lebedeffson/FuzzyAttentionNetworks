#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fan.concept import MultiSetAdditiveTemporalConceptFANModel
from scripts.medical.v2.run_v2_1_program import make_sequence_loaders
from scripts.medical.v2.run_v2_program import make_episodes, split_frame, subset_cols
from scripts.medical.v3.diagnose_fan_faithfulness import (
    batch_subset_outputs,
    exact_random_metric_baseline,
    ids_from_order,
    mean_values_for_episode_masks,
    ranking_orders,
    shapley_from_subsets,
    subset_masks,
    sufficient_mask,
)
from scripts.medical.v3.run_fan_iteration import ConceptScaler, binary_metrics, build_loaders, concept_features, set_all_seeds


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_NAMES = ["I", "R", "V", "O", "S"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_oracle_reference(oracle_results: Path, oracle_manifest: Path, seeds: list[int]) -> pd.DataFrame:
    if not oracle_results.exists():
        raise FileNotFoundError(f"Oracle results not found: {oracle_results}")
    if not oracle_manifest.exists():
        raise FileNotFoundError(f"Oracle manifest not found: {oracle_manifest}")
    manifest = json.loads(oracle_manifest.read_text(encoding="utf-8"))
    expected = {
        "concept_scaling": "minmax_train",
        "membership_family": "gaussian",
        "n_memberships": 3,
        "alpha_mode": "no_alpha",
        "concept_definition": "temporal_I_R_V_O_S_hours_0_35",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Oracle manifest mismatch for {key}: {manifest.get(key)!r} != {value!r}")
    if "NoAlpha" not in str(manifest.get("architecture", "")):
        raise ValueError("Oracle manifest architecture is not NoAlpha")
    if sorted(manifest.get("seeds", [])) != sorted(seeds):
        raise ValueError(f"Oracle manifest seeds do not match requested seeds: {manifest.get('seeds')} vs {seeds}")
    recorded_hash = manifest.get("results_sha256")
    if recorded_hash and recorded_hash != sha256_file(oracle_results):
        raise ValueError("Oracle results SHA256 does not match manifest")
    oracle = pd.read_csv(oracle_results)
    if sorted(oracle["seed"].astype(int).tolist()) != sorted(seeds):
        raise ValueError("Oracle results seed set does not match requested seeds")
    return oracle


def cfg_for_alpha(cfg: dict, alpha_mode: str) -> dict:
    out = dict(cfg)
    out["fan"] = dict(cfg.get("fan", {}))
    out["fan"]["alpha_mode"] = alpha_mode
    out["fan"]["gamma_init"] = float(out["fan"].get("gamma_init", 0.5))
    out["fan"]["gamma_max"] = float(out["fan"].get("gamma_max", 0.9))
    return out


def make_predicted_model(cfg: dict, alpha_mode: str, prevalence: float) -> MultiSetAdditiveTemporalConceptFANModel:
    fan_cfg = cfg_for_alpha(cfg, alpha_mode)
    model = MultiSetAdditiveTemporalConceptFANModel(
        input_dim=int(cfg["model"]["input_dim"]),
        sequence_length=int(cfg["dataset"]["observed_window"]),
        latent_dim=int(cfg["model"]["latent_dim"]),
        n_concepts=5,
        n_memberships=3,
        membership="gaussian",
        oracle=False,
        temporal_mode="attention",
        dropout=float(cfg["model"].get("dropout", 0.1)),
        encoder_layers=int(cfg["model"].get("layers", 4)),
        encoder_heads=int(cfg["model"].get("heads", 4)),
        encoder_ffn=int(cfg["model"].get("d_ffn", 512)),
        temperature=1.0,
        positive_decision_weights=False,
        alpha_mode=str(fan_cfg["fan"].get("alpha_mode", "no_alpha")),
        gamma_init=float(fan_cfg["fan"].get("gamma_init", 0.5)),
        gamma_max=float(fan_cfg["fan"].get("gamma_max", 0.9)),
    ).to(DEVICE)
    model.decision_head.initialize_bias_from_prevalence(prevalence)
    return model


def per_concept_huber(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = F.huber_loss(pred, target, reduction="none")
    return loss.mean(dim=(0, 1)).mean()


def correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])
    pred_centered = pred_flat - pred_flat.mean(dim=0, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=0, keepdim=True)
    numerator = (pred_centered * target_centered).mean(dim=0)
    denominator = pred_centered.std(dim=0).clamp_min(1e-6) * target_centered.std(dim=0).clamp_min(1e-6)
    corr = numerator / denominator
    return (1.0 - corr.clamp(-1.0, 1.0)).mean()


def _r2_direct(y: np.ndarray, p: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 1e-12:
        return np.nan
    return float(1.0 - np.sum((y - p) ** 2) / denom)


def trajectory_metrics(
    true_seq: np.ndarray,
    pred_seq: np.ndarray,
    train_true_seq: np.ndarray | None = None,
    train_pred_seq: np.ndarray | None = None,
) -> dict:
    rows = []
    for idx, name in enumerate(STATE_NAMES):
        y = true_seq[:, :, idx].reshape(-1)
        p = pred_seq[:, :, idx].reshape(-1)
        if np.std(y) == 0 or np.std(p) == 0:
            pearson = np.nan
            spearman = np.nan
            direct_r2 = np.nan
            calibrated_r2 = np.nan
        else:
            pearson = float(stats.pearsonr(y, p).statistic)
            spearman = float(stats.spearmanr(y, p).statistic)
            direct_r2 = _r2_direct(y, p)
            calibrated_r2 = np.nan
            if train_true_seq is not None and train_pred_seq is not None:
                yt = train_true_seq[:, :, idx].reshape(-1)
                pt = train_pred_seq[:, :, idx].reshape(-1)
                if np.std(yt) > 0 and np.std(pt) > 0:
                    calibrator = LinearRegression().fit(pt.reshape(-1, 1), yt)
                    calibrated_r2 = _r2_direct(y, calibrator.predict(p.reshape(-1, 1)))
        rows.append(
            {
                "concept": name,
                "direct_trajectory_r2": direct_r2,
                "train_calibrated_trajectory_r2": calibrated_r2,
                "trajectory_pearson": pearson,
                "trajectory_pearson_squared": float(pearson * pearson) if np.isfinite(pearson) else np.nan,
                "trajectory_spearman": spearman,
            }
        )
    df = pd.DataFrame(rows)
    return {
        "per_concept": df,
        "macro_trajectory_r2": float(df["direct_trajectory_r2"].mean()),
        "macro_direct_trajectory_r2": float(df["direct_trajectory_r2"].mean()),
        "macro_train_calibrated_trajectory_r2": float(df["train_calibrated_trajectory_r2"].mean()),
        "mean_trajectory_pearson": float(df["trajectory_pearson"].mean()),
        "mean_trajectory_pearson_squared": float(df["trajectory_pearson_squared"].mean()),
        "mean_trajectory_spearman": float(df["trajectory_spearman"].mean()),
    }


def collect_concept_predictions(model: MultiSetAdditiveTemporalConceptFANModel, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels, true, pred = [], [], []
    with torch.no_grad():
        for xb, yb, cb in loader:
            xb = xb.to(DEVICE)
            h = model.encoder(xb)
            p = model.projector(h)
            labels.append(yb.numpy())
            true.append(cb.numpy())
            pred.append(p.cpu().numpy())
    return np.concatenate(labels), np.concatenate(true), np.concatenate(pred)


def train_concept_stage(model: MultiSetAdditiveTemporalConceptFANModel, cfg: dict, train_loader: DataLoader, val_loader: DataLoader) -> tuple[list[dict], dict]:
    opt = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.projector.parameters()),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    best_score = -float("inf")
    best_state = None
    history = []
    for epoch in range(int(cfg["training"]["concept_epochs"])):
        model.train()
        losses = []
        for xb, _, cb in train_loader:
            xb, cb = xb.to(DEVICE), cb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model.projector(model.encoder(xb))
            state = per_concept_huber(pred, cb)
            delta = per_concept_huber(pred[:, 1:] - pred[:, :-1], cb[:, 1:] - cb[:, :-1])
            corr = correlation_loss(pred, cb)
            loss = state + 0.2 * delta + 0.1 * corr
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.encoder.parameters()) + list(model.projector.parameters()), 1.0)
            opt.step()
            losses.append((float(state.item()), float(delta.item()), float(corr.item()), float(loss.item())))
        _, train_true, train_pred = collect_concept_predictions(model, train_loader)
        _, val_true, val_pred = collect_concept_predictions(model, val_loader)
        met = trajectory_metrics(val_true, val_pred, train_true, train_pred)
        score = met["macro_trajectory_r2"] + 0.01 * met["mean_trajectory_pearson"]
        row = {
            "epoch": epoch + 1,
            "state_huber": float(np.mean([x[0] for x in losses])),
            "delta_huber": float(np.mean([x[1] for x in losses])),
            "correlation_loss": float(np.mean([x[2] for x in losses])),
            "concept_stage_loss": float(np.mean([x[3] for x in losses])),
            "validation_macro_r2": met["macro_trajectory_r2"],
            "validation_mean_pearson": met["mean_trajectory_pearson"],
        }
        history.append(row)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    _, train_true, train_pred = collect_concept_predictions(model, train_loader)
    _, val_true, val_pred = collect_concept_predictions(model, val_loader)
    return history, trajectory_metrics(val_true, val_pred, train_true, train_pred)


def initialize_memberships_from_predicted(model: MultiSetAdditiveTemporalConceptFANModel, loader: DataLoader) -> None:
    summaries = []
    model.eval()
    with torch.no_grad():
        for xb, _, _ in loader:
            xb = xb.to(DEVICE)
            pred = model.projector(model.encoder(xb))
            q, _ = model.temporal_aggregator(pred, model.temporal_mode)
            summaries.append(q.cpu())
    model.membership.initialize_from_quantiles(torch.cat(summaries, dim=0).to(DEVICE))


def train_fan_head(model: MultiSetAdditiveTemporalConceptFANModel, cfg: dict, train_loader: DataLoader, val_loader: DataLoader) -> list[dict]:
    model.freeze_concept_path()
    for p in model.temporal_aggregator.parameters():
        p.requires_grad = True
    for p in model.membership.parameters():
        p.requires_grad = True
    for p in model.aggregator.parameters():
        p.requires_grad = True
    for p in model.decision_head.parameters():
        p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    best = -1.0
    best_state = None
    history = []
    for epoch in range(int(cfg["training"]["max_epochs"])):
        model.train()
        losses = []
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = F.binary_cross_entropy_with_logits(out.logit, yb)
            if model.aggregator.alpha_mode != "no_alpha":
                alpha = out.concept_weights / out.concept_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
                entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=1).mean()
                loss = loss + float(cfg["fan"].get("lambda_sparse", 0.0)) * entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.item()))
        y, p = evaluate_probabilities(model, val_loader)
        auprc = float(average_precision_score(y, p))
        history.append({"epoch": epoch + 1, "task_loss": float(np.mean(losses)), "validation_AUPRC": auprc})
        if auprc > best:
            best = auprc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return history


def evaluate_probabilities(model: MultiSetAdditiveTemporalConceptFANModel, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, probs = [], []
    with torch.no_grad():
        for xb, yb, _ in loader:
            out = model(xb.to(DEVICE))
            labels.append(yb.numpy())
            probs.append(out.probability.cpu().numpy())
    return np.concatenate(labels), np.concatenate(probs)


def leakage_audit(train_true: np.ndarray, train_pred: np.ndarray, train_y: np.ndarray, val_true: np.ndarray, val_pred: np.ndarray, val_y: np.ndarray, seed: int) -> tuple[dict, pd.DataFrame]:
    xtr_true = concept_features(train_true)
    xva_true = concept_features(val_true)
    xtr_pred = concept_features(train_pred)
    xva_pred = concept_features(val_pred)
    bridge = LinearRegression().fit(xtr_true, xtr_pred)
    train_resid = xtr_pred - bridge.predict(xtr_true)
    val_resid = xva_pred - bridge.predict(xva_true)
    rng = np.random.default_rng(seed)
    shuffled = val_resid.copy()
    rng.shuffle(shuffled, axis=0)
    def prob(xtr, xva):
        model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(xtr, train_y.astype(int))
        return model.predict_proba(xva)[:, 1]
    p_true = prob(xtr_true, xva_true)
    p_pred = prob(xtr_pred, xva_pred)
    p_resid = prob(train_resid, val_resid)
    p_shuf = prob(train_resid, shuffled)
    diffs = []
    ids = np.arange(len(val_y))
    for _ in range(2000):
        sample = rng.choice(ids, size=len(ids), replace=True)
        diffs.append(average_precision_score(val_y[sample], p_resid[sample]) - average_precision_score(val_y[sample], p_shuf[sample]))
    low, high = np.quantile(diffs, [0.025, 0.975])
    metrics = {
        "AUPRC_true_concepts": float(average_precision_score(val_y, p_true)),
        "AUPRC_predicted_concepts": float(average_precision_score(val_y, p_pred)),
        "AUPRC_residual": float(average_precision_score(val_y, p_resid)),
        "AUPRC_shuffled_residual": float(average_precision_score(val_y, p_shuf)),
        "residual_minus_shuffled_ci_low": float(low),
        "residual_minus_shuffled_ci_high": float(high),
        "leakage_gate_passed": bool(low <= 0.0),
    }
    boot = pd.DataFrame({"bootstrap_id": np.arange(len(diffs)), "residual_minus_shuffled_auprc": diffs})
    return metrics, boot


def exact_faithfulness(model: MultiSetAdditiveTemporalConceptFANModel, loader: DataLoader, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    masks = subset_masks(5)
    full_mask = 2**5 - 1
    subset_rows, shapley_rows, ranking_rows = [], [], []
    minimal_sizes, minimal_nonempty, bias_ok = [], [], []
    max_decomp = 0.0
    episode_offset = 0
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            bsz = len(yb)
            episode_ids = np.arange(episode_offset, episode_offset + bsz)
            episode_offset += bsz
            full_logits = out.logit.cpu().numpy()
            full_probs = out.probability.cpu().numpy()
            targets = yb.numpy().astype(int)
            reconstructed = model.decision_head.bias + out.signed_decision_contributions.sum(dim=-1)
            max_decomp = max(max_decomp, float(torch.max(torch.abs(reconstructed - out.logit)).item()))
            _, _, functional_logits, functional_probs = batch_subset_outputs(model, out, masks)
            shapley_logit = shapley_from_subsets(functional_logits, 5)
            shapley_prob = shapley_from_subsets(functional_probs, 5)
            signed = out.signed_decision_contributions.cpu().numpy()
            for i, episode_id in enumerate(episode_ids):
                sufficient = [
                    mask_id for mask_id in functional_probs
                    if bool(sufficient_mask(
                        np.asarray([functional_probs[mask_id][i]]),
                        np.asarray([functional_logits[mask_id][i]]),
                        np.asarray([full_probs[i]]),
                        np.asarray([full_logits[i]]),
                    )[0])
                ]
                minimal_sizes.append(min((int(mask_id).bit_count() for mask_id in sufficient), default=5))
                minimal_nonempty.append(min((int(mask_id).bit_count() for mask_id in sufficient if mask_id != 0), default=5))
                bias_ok.append(0 in sufficient)
                for mask_id, indices, mask_np in masks:
                    subset_rows.append({
                        "seed": seed,
                        "episode_id": int(episode_id),
                        "target": int(targets[i]),
                        "subset_mask_int": int(mask_id),
                        "subset_size": len(indices),
                        "probability": float(functional_probs[mask_id][i]),
                        "logit": float(functional_logits[mask_id][i]),
                        "full_probability": float(full_probs[i]),
                        "full_logit": float(full_logits[i]),
                        "absolute_probability_error": float(abs(functional_probs[mask_id][i] - full_probs[i])),
                        "absolute_logit_error": float(abs(functional_logits[mask_id][i] - full_logits[i])),
                        "class_preserved": bool((functional_probs[mask_id][i] >= 0.5) == (full_probs[i] >= 0.5)),
                    })
                for concept_idx, concept in enumerate(STATE_NAMES):
                    shapley_rows.append({
                        "seed": seed,
                        "episode_id": int(episode_id),
                        "concept": concept,
                        "concept_index": concept_idx,
                        "shapley_logit": float(shapley_logit[i, concept_idx]),
                        "shapley_probability": float(shapley_prob[i, concept_idx]),
                        "signed_additive_contribution": float(signed[i, concept_idx]),
                        "logit_difference_shapley_minus_signed": float(shapley_logit[i, concept_idx] - signed[i, concept_idx]),
                    })
            orders = ranking_orders(out, shapley_logit)
            for ranking_name, order in orders.items():
                for k in range(1, 6):
                    insertion_masks = ids_from_order(order, k)
                    removal_masks = np.asarray([full_mask ^ int(mask_id) for mask_id in insertion_masks], dtype=np.int64)
                    insertion_prob = mean_values_for_episode_masks(functional_probs, insertion_masks)
                    insertion_logit = mean_values_for_episode_masks(functional_logits, insertion_masks)
                    removal_prob = mean_values_for_episode_masks(functional_probs, removal_masks)
                    removal_logit = mean_values_for_episode_masks(functional_logits, removal_masks)
                    random_insertion = exact_random_metric_baseline(functional_probs, functional_logits, k, 5, full_probs, full_logits, targets)
                    random_removal = exact_random_metric_baseline(functional_probs, functional_logits, 5 - k, 5, full_probs, full_logits, targets)
                    ranking_rows.append({
                        "seed": seed,
                        "ranking_rule": ranking_name,
                        "k": k,
                        "insertion_probability_mae": float(np.mean(np.abs(insertion_prob - full_probs))),
                        "insertion_logit_mae": float(np.mean(np.abs(insertion_logit - full_logits))),
                        "insertion_sufficient_fraction": float(np.mean(sufficient_mask(insertion_prob, insertion_logit, full_probs, full_logits))),
                        "random_insertion_probability_mae": random_insertion["probability_mae"],
                        "random_insertion_logit_mae": random_insertion["logit_mae"],
                        "random_insertion_sufficient_fraction": random_insertion["sufficient_fraction"],
                        "removal_probability_delta": float(np.mean(np.abs(full_probs - removal_prob))),
                        "removal_logit_delta": float(np.mean(np.abs(full_logits - removal_logit))),
                        "random_removal_probability_delta": random_removal["probability_delta"],
                        "random_removal_logit_delta": random_removal["logit_mae"],
                        "removal_remaining_auprc": float(average_precision_score(targets, removal_prob)),
                    })
    diagnosis = {
        "max_decomposition_logit_error": max_decomp,
        "median_minimal_sufficient_subset_size": float(np.median(minimal_sizes)),
        "median_minimal_nonempty_sufficient_subset_size": float(np.median(minimal_nonempty)),
        "bias_only_sufficient_fraction": float(np.mean(bias_ok)),
        "share_nonempty_sufficient_with_top3_or_less": float(np.mean(np.asarray(minimal_nonempty) <= 3)),
        "mean_abs_shapley_signed_logit_difference": float(pd.DataFrame(shapley_rows)["logit_difference_shapley_minus_signed"].abs().mean()),
    }
    return pd.DataFrame(subset_rows), pd.DataFrame(shapley_rows), pd.DataFrame(ranking_rows), diagnosis


class TemporalCBMHead(nn.Module):
    def __init__(self, n_concepts: int):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(n_concepts * 5, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        last = seq[:, -1, :]
        mean = seq.mean(dim=1)
        maxv = seq.max(dim=1).values
        last6 = seq[:, -6:, :]
        mean6 = last6.mean(dim=1)
        slope6 = last6[:, -1, :] - last6[:, 0, :]
        return self.head(torch.cat([last, mean, maxv, mean6, slope6], dim=1)).squeeze(-1)


def train_cbm_head(model: MultiSetAdditiveTemporalConceptFANModel, cfg: dict, train_loader: DataLoader, val_loader: DataLoader) -> tuple[dict, list[dict]]:
    head = TemporalCBMHead(5).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    best = -1.0
    best_state = None
    history = []
    model.eval()
    for epoch in range(int(cfg["training"]["max_epochs"])):
        head.train()
        losses = []
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.no_grad():
                pred = model.projector(model.encoder(xb))
            opt.zero_grad(set_to_none=True)
            logit = head(pred)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        y, p = [], []
        head.eval()
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                pred = model.projector(model.encoder(xb.to(DEVICE)))
                y.append(yb.numpy())
                p.append(torch.sigmoid(head(pred)).cpu().numpy())
        yy, pp = np.concatenate(y), np.concatenate(p)
        auprc = float(average_precision_score(yy, pp))
        history.append({"epoch": epoch + 1, "task_loss": float(np.mean(losses)), "validation_AUPRC": auprc})
        if auprc > best:
            best = auprc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    y, p = [], []
    head.eval()
    with torch.no_grad():
        for xb, yb, _ in val_loader:
            pred = model.projector(model.encoder(xb.to(DEVICE)))
            y.append(yb.numpy())
            p.append(torch.sigmoid(head(pred)).cpu().numpy())
    yy, pp = np.concatenate(y), np.concatenate(p)
    return {**binary_metrics(yy, pp), "model": "CBM_Temporal"}, history


def _clone_concept_path(source: MultiSetAdditiveTemporalConceptFANModel, target: MultiSetAdditiveTemporalConceptFANModel) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name in target_state:
        if name.startswith("encoder.") or name.startswith("projector."):
            target_state[name] = source_state[name].detach().clone()
    target.load_state_dict(target_state)


def run_seed(seed: int, cfg: dict, output: Path) -> tuple[list[dict], pd.DataFrame]:
    set_all_seeds(seed)
    clean = make_episodes(seed, cfg, "clean")
    split = split_frame(clean, seed)
    _, _, arrays = make_sequence_loaders(split, subset_cols("full_input"), int(cfg["training"]["batch_size"]), 5)
    scaler = ConceptScaler.fit(arrays["c_train_seq"], "minmax_train")
    train_loader, val_loader, scaled = build_loaders(arrays, scaler, 5, int(cfg["training"]["batch_size"]))
    seed_rows = []
    leakage_rows = []
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(seed * 1000 + 1)
    concept_reference = make_predicted_model(cfg, "no_alpha", float(arrays["y_train"].mean()))
    concept_history, concept_met = train_concept_stage(concept_reference, cfg, train_loader, val_loader)
    concept_state = {
        k: v.detach().cpu().clone()
        for k, v in concept_reference.state_dict().items()
        if k.startswith("encoder.") or k.startswith("projector.")
    }
    torch.save(concept_state, seed_dir / "shared_concept_checkpoint.pt")
    pd.DataFrame(concept_history).to_csv(seed_dir / "shared_concept_training_log.csv", index=False)
    trained_reference = None
    for alpha_mode, model_name in [("no_alpha", "Predicted_Temporal_FAN_NoAlpha_Strict"), ("softmax_alpha", "Predicted_Temporal_Softmax_FAN_Strict")]:
        set_all_seeds(seed * 1000 + (1 if alpha_mode == "no_alpha" else 2))
        model = make_predicted_model(cfg, alpha_mode, float(arrays["y_train"].mean()))
        model_state = model.state_dict()
        for key, value in concept_state.items():
            model_state[key] = value.to(model_state[key].device)
        model.load_state_dict(model_state)
        initialize_memberships_from_predicted(model, train_loader)
        head_history = train_fan_head(model, cfg_for_alpha(cfg, alpha_mode), train_loader, val_loader)
        y, p = evaluate_probabilities(model, val_loader)
        train_y, train_true, train_pred = collect_concept_predictions(model, train_loader)
        val_y, val_true, val_pred = collect_concept_predictions(model, val_loader)
        leak, boot = leakage_audit(train_true, train_pred, train_y.astype(int), val_true, val_pred, val_y.astype(int), seed)
        sub, shp, rnk, faith_diag = exact_faithfulness(model, val_loader, seed)
        model_dir = seed_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(concept_history).to_csv(model_dir / "concept_training_log.csv", index=False)
        pd.DataFrame(head_history).to_csv(model_dir / "head_training_log.csv", index=False)
        boot.to_parquet(model_dir / "leakage_bootstrap.parquet", index=False)
        sub.to_parquet(model_dir / "exact_subset_faithfulness.parquet", index=False)
        shp.to_parquet(model_dir / "shapley_contributions.parquet", index=False)
        rnk.to_csv(model_dir / "ranking_comparison.csv", index=False)
        torch.save(model.state_dict(), model_dir / "checkpoint.pt")
        row = {
            "seed": seed,
            "model": model_name,
            "alpha_mode": alpha_mode,
            **binary_metrics(y, p),
            "macro_trajectory_r2": concept_met["macro_trajectory_r2"],
            "macro_direct_trajectory_r2": concept_met["macro_direct_trajectory_r2"],
            "macro_train_calibrated_trajectory_r2": concept_met["macro_train_calibrated_trajectory_r2"],
            "mean_trajectory_pearson": concept_met["mean_trajectory_pearson"],
            "mean_trajectory_pearson_squared": concept_met["mean_trajectory_pearson_squared"],
            **leak,
            **faith_diag,
        }
        seed_rows.append(row)
        leakage_rows.append({"seed": seed, "model": model_name, **leak})
        if alpha_mode == "no_alpha":
            trained_reference = model
    cbm_metrics, cbm_history = train_cbm_head(trained_reference, cfg, train_loader, val_loader)
    pd.DataFrame(cbm_history).to_csv(seed_dir / "CBM_Temporal_training_log.csv", index=False)
    seed_rows.append({"seed": seed, **cbm_metrics})
    return seed_rows, pd.DataFrame(leakage_rows)


def run_program(cfg: dict, seeds: list[int], output: Path, oracle_results: Path, oracle_manifest: Path) -> dict:
    oracle_reference = validate_oracle_reference(oracle_results, oracle_manifest, seeds)
    output.mkdir(parents=True, exist_ok=True)
    rows, leakage = [], []
    for seed in seeds:
        r, l = run_seed(seed, cfg, output)
        rows.extend(r)
        leakage.append(l)
    results = pd.DataFrame(rows)
    leak_df = pd.concat(leakage, ignore_index=True)
    results.to_csv(output / "predicted_fan_results.csv", index=False)
    leak_df.to_csv(output / "concept_leakage_metrics.csv", index=False)
    primary = results[results["model"] == "Predicted_Temporal_FAN_NoAlpha_Strict"]
    oracle = oracle_reference[["seed", "AUPRC"]].rename(columns={"AUPRC": "oracle_noalpha_AUPRC"})
    primary_gate = primary.merge(oracle, on="seed", how="left")
    primary_gate["predicted_oracle_ratio"] = primary_gate["AUPRC"] / primary_gate["oracle_noalpha_AUPRC"]
    gates = {
        "predictive_pass_count": int((primary_gate["predicted_oracle_ratio"] >= 0.90).sum()),
        "r2_pass_count": int((primary["macro_trajectory_r2"] >= 0.50).sum()),
        "pearson_pass_count": int((primary["mean_trajectory_pearson"] >= 0.65).sum()),
        "leakage_pass_count": int(primary["leakage_gate_passed"].fillna(False).sum()),
        "decomposition_pass_count": int((primary["max_decomposition_logit_error"] <= 1e-6).sum()),
        "sufficiency_pass_count": int((primary["median_minimal_nonempty_sufficient_subset_size"] <= 3).sum()),
    }
    status = "PREDICTED_FAN_VALIDATED" if min(gates.values()) >= 2 else "PREDICTED_FAN_NEEDS_PROJECTOR_WORK"
    diagnosis = {
        "stage": "predicted_temporal_fan_strict",
        "test_opened": False,
        "primary_model": "Predicted_Temporal_FAN_NoAlpha_Strict",
        "control_model": "Predicted_Temporal_Softmax_FAN_Strict",
        "baseline": "CBM_Temporal",
        "oracle_results": str(oracle_results),
        "oracle_manifest": str(oracle_manifest),
        "shared_concept_checkpoint_for_alpha_ablation": True,
        "status": status,
        "gates": gates,
        "mean_primary_AUPRC": float(primary["AUPRC"].mean()),
        "mean_predicted_oracle_ratio": float(primary_gate["predicted_oracle_ratio"].mean()),
        "min_predicted_oracle_ratio": float(primary_gate["predicted_oracle_ratio"].min()),
        "mean_primary_macro_r2": float(primary["macro_trajectory_r2"].mean()),
        "mean_primary_pearson": float(primary["mean_trajectory_pearson"].mean()),
    }
    primary_gate.to_csv(output / "predicted_vs_oracle_noalpha.csv", index=False)
    (output / "predicted_fan_diagnosis.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    return diagnosis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oracle-results", required=True)
    parser.add_argument("--oracle-manifest", required=True)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run_program(cfg, args.seeds, Path(args.output), Path(args.oracle_results), Path(args.oracle_manifest))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
