#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fan.sctc.adaptive import AdaptiveSCTCTrainConfig, layer_specific_capacity, train_adaptive_sctc
from fan.sctc.evaluation import fidelity_metrics
from med_circuitbench.planted.model import EDGES, NODE_LAYERS, STATE_NAMES, PlantedCircuitModel
from scripts.medical.v2.run_v2_program import make_episodes, split_frame, subset_cols
from scripts.medical.v3.run_research_program import DEVICE, prepare_arrays, sctc_fidelity_predictions


CORRELATION_THRESHOLD = 0.80
COSINE_THRESHOLD = 0.50
NODE_EFFECT_THRESHOLD = 1e-4
TOPK_TOLERANCE = 0.25


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pred = p >= 0.5
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return {
        "AUROC": float(roc_auc_score(y, p)) if np.unique(y).size > 1 else float("nan"),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "ECE": ece,
        "F1": float(f1_score(y, pred)) if np.unique(pred).size > 1 else 0.0,
        "positive_prediction_fraction": float(pred.mean()),
    }


def bh_q_values(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


class ScalarCalibrator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_scale = torch.nn.Parameter(torch.zeros(()))
        self.bias = torch.nn.Parameter(torch.zeros(()))

    @property
    def scale(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_scale) + 1e-6

    def forward(self, raw_logit: torch.Tensor) -> torch.Tensor:
        return self.scale * raw_logit + self.bias


def calibrate_planted_head(raw_train_logit: np.ndarray, y_train: np.ndarray, seed: int, epochs: int = 300) -> tuple[ScalarCalibrator, dict]:
    torch.manual_seed(seed + 9100)
    raw = torch.from_numpy(raw_train_logit.astype(np.float32)).to(DEVICE)
    target = torch.from_numpy(y_train.astype(np.float32)).to(DEVICE)
    model = ScalarCalibrator().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=0.05)
    best = None
    best_loss = float("inf")
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(raw), target)
        loss.backward()
        opt.step()
        if float(loss.item()) < best_loss:
            best_loss = float(loss.item())
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best is not None:
        model.load_state_dict(best)
    with torch.no_grad():
        prob = torch.sigmoid(model(raw)).detach().cpu().numpy()
    metrics = binary_metrics(y_train, prob)
    metrics.update({"scale": float(model.scale.detach().cpu().item()), "bias": float(model.bias.detach().cpu().item()), "loss": best_loss})
    return model.eval(), metrics


def calibrate_np(calibrator: ScalarCalibrator, raw_logit: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(raw_logit.astype(np.float32)).to(DEVICE)
        return calibrator(x).detach().cpu().numpy()


def registered_candidate_capacities(rank_capacity: int, method_cfg: dict) -> list[int]:
    allowed = sorted(
        {
            int(item["n_features"])
            for item in method_cfg["adaptive_sctc"]["planted_grid"]
        }
        | {int(method_cfg["adaptive_sctc"]["layer_specific_capacity"]["min_features"])}
        | {int(method_cfg["adaptive_sctc"]["layer_specific_capacity"]["max_features"])}
        | {16, 24, 32, 48, 64, 96}
    )
    lower = [x for x in allowed if x <= rank_capacity]
    upper = [x for x in allowed if x >= rank_capacity]
    candidates = {
        lower[-1] if lower else allowed[0],
        min(allowed, key=lambda x: abs(x - rank_capacity)),
        upper[0] if upper else allowed[-1],
    }
    return sorted(candidates)


def target_top_k_for_capacity(n_features: int, method_cfg: dict) -> int:
    registered = {int(item["n_features"]): int(item["top_k"]) for item in method_cfg["adaptive_sctc"]["planted_grid"]}
    if int(n_features) in registered:
        return registered[int(n_features)]
    if n_features <= 32:
        return 8
    if n_features <= 48:
        return 12
    return 16


def final_activation_stats(transcoder, activation: np.ndarray, target_top_k: int, dead_threshold: float) -> dict:
    transcoder.eval()
    transcoder.set_active_top_k(target_top_k)
    device = next(transcoder.parameters()).device
    with torch.no_grad():
        z = transcoder(torch.from_numpy(activation).float().to(device))["z"].detach().cpu()
    freq = (z > 0).float().mean(dim=(0, 1)).numpy()
    l0 = float((z > 0).float().sum(dim=-1).mean().item())
    if l0 > target_top_k + TOPK_TOLERANCE:
        raise RuntimeError(f"final L0 invariant failed: L0={l0:.6f}, final_top_k={target_top_k}")
    quantiles = np.quantile(freq, [0.0, 0.25, 0.50, 0.75, 1.0])
    return {
        "final_active_top_k": int(transcoder.active_top_k),
        "final_L0_per_token": l0,
        "final_dead_feature_fraction": float(np.mean(freq < dead_threshold)),
        "feature_frequency_q0": float(quantiles[0]),
        "feature_frequency_q25": float(quantiles[1]),
        "feature_frequency_q50": float(quantiles[2]),
        "feature_frequency_q75": float(quantiles[3]),
        "feature_frequency_q100": float(quantiles[4]),
        "effective_active_feature_count": int(np.sum(freq >= dead_threshold)),
    }


def flatten_layer(model: PlantedCircuitModel, states: np.ndarray, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        out = model(torch.from_numpy(states).float().to(DEVICE))
    return (
        out.layers[layer].detach().cpu().numpy(),
        out.logit.detach().cpu().numpy(),
        out.probability.detach().cpu().numpy(),
        out.nodes.detach().cpu().numpy(),
    )


def node_matching(model: PlantedCircuitModel, transcoder, activation: np.ndarray, nodes: np.ndarray, layer: int, seed: int) -> pd.DataFrame:
    device = next(transcoder.parameters()).device
    with torch.no_grad():
        z = transcoder(torch.from_numpy(activation).float().to(device))["z"].detach().cpu().numpy().reshape(-1, transcoder.n_features)
    decoder = transcoder.decoder.weight.detach().cpu().T.numpy()
    layer_nodes = [name for name, node_layer in NODE_LAYERS.items() if node_layer == layer]
    if not layer_nodes:
        return pd.DataFrame()
    flat_nodes = nodes.reshape(-1, nodes.shape[-1])
    score = np.zeros((transcoder.n_features, len(layer_nodes)), dtype=np.float64)
    corr_mat = np.zeros_like(score)
    cos_mat = np.zeros_like(score)
    for feature_id in range(transcoder.n_features):
        for node_col, node in enumerate(layer_nodes):
            idx = STATE_NAMES.index(node)
            target = flat_nodes[:, idx]
            corr = 0.0 if np.std(z[:, feature_id]) == 0 or np.std(target) == 0 else abs(float(stats.pearsonr(z[:, feature_id], target).statistic))
            direction = model.directions[idx].detach().cpu().numpy()
            cosine = abs(float(np.dot(decoder[feature_id], direction) / ((np.linalg.norm(decoder[feature_id]) + 1e-8) * (np.linalg.norm(direction) + 1e-8))))
            corr_mat[feature_id, node_col] = corr
            cos_mat[feature_id, node_col] = cosine
            score[feature_id, node_col] = corr * cosine
    row_ind, col_ind = linear_sum_assignment(-score)
    primary = {(int(r), int(c)) for r, c in zip(row_ind, col_ind)}
    rows = []
    for feature_id in range(transcoder.n_features):
        best_col = int(np.argmax(score[feature_id]))
        for node_col, node in enumerate(layer_nodes):
            corr = corr_mat[feature_id, node_col]
            cosine = cos_mat[feature_id, node_col]
            if (feature_id, node_col) in primary:
                match_type = "PRIMARY_HUNGARIAN_MATCH"
            elif corr >= CORRELATION_THRESHOLD:
                match_type = "REDUNDANT_CORRELATED_FEATURE"
            elif cosine < COSINE_THRESHOLD:
                match_type = "REJECTED_LOW_DIRECTION_COSINE"
            else:
                match_type = "CORRELATION_MATCH_ONLY" if node_col == best_col else "REJECTED_NONPRIMARY_NODE"
            rows.append(
                {
                    "seed": seed,
                    "layer": layer,
                    "feature_id": feature_id,
                    "node": node,
                    "activation_correlation": float(corr),
                    "decoder_cosine": float(cosine),
                    "intervention_effect": np.nan,
                    "null_q99": np.nan,
                    "p_value": np.nan,
                    "q_value": np.nan,
                    "match_type": match_type,
                    "accepted": False,
                }
            )
    return pd.DataFrame(rows)


def intervention_validate_matches(
    matches: pd.DataFrame,
    model: PlantedCircuitModel,
    transcoder,
    activation: np.ndarray,
    nodes: np.ndarray,
    layer: int,
    seed: int,
    run_interventions: bool,
    run_negative_controls: bool,
    n_nulls: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if matches.empty:
        return matches, pd.DataFrame()
    if not run_interventions:
        matches = matches.copy()
        matches["match_type"] = np.where(matches["match_type"].eq("PRIMARY_HUNGARIAN_MATCH"), "CORRELATION_MATCH_ONLY", matches["match_type"])
        return matches, pd.DataFrame()

    rng = np.random.default_rng(seed + 7100 + layer)
    device = next(transcoder.parameters()).device
    model = model.to(device)
    with torch.no_grad():
        act_t = torch.from_numpy(activation).float().to(device)
        out = transcoder(act_t)
        z = out["z"]
        directions = transcoder.decoder.weight.detach().T
        base_nodes = torch.from_numpy(nodes).float().to(device)
        base_forward = model.downstream_from_layer(layer, act_t)
        base_logit = base_forward.logit
        base_probability = base_forward.probability
    evidence_rows = []
    updated = []
    primary = matches[matches["match_type"].eq("PRIMARY_HUNGARIAN_MATCH")].copy()
    p_values = []
    effects = []
    null_q99s = []
    for _, row in primary.iterrows():
        feature_id = int(row["feature_id"])
        node = str(row["node"])
        node_idx = STATE_NAMES.index(node)
        direction = directions[feature_id]
        coeff = z[..., feature_id]
        intervened = act_t - coeff.unsqueeze(-1) * direction.view(1, 1, -1)
        push_scale = float(z[..., feature_id].detach().std().cpu().item())
        pushed = act_t + push_scale * direction.view(1, 1, -1)
        with torch.no_grad():
            after_forward = model.downstream_from_layer(layer, intervened)
            push_forward = model.downstream_from_layer(layer, pushed)
            after = after_forward.nodes
            pushed_nodes = push_forward.nodes
        target_nulls: dict[str, list[float]] = {target: [] for target in STATE_NAMES}
        node_effect = float((after[..., node_idx] - base_nodes[..., node_idx]).mean().detach().cpu().item())
        node_push_effect = float((pushed_nodes[..., node_idx] - base_nodes[..., node_idx]).mean().detach().cpu().item())
        for null_id in range(n_nulls):
            rand = rng.normal(size=direction.numel()).astype(np.float32)
            rand = rand / (np.linalg.norm(rand) + 1e-8)
            rand_t = torch.from_numpy(rand).float().to(device)
            coeff_rand = (act_t * rand_t.view(1, 1, -1)).sum(dim=-1)
            random_act = act_t - coeff_rand.unsqueeze(-1) * rand_t.view(1, 1, -1)
            with torch.no_grad():
                random_nodes = model.downstream_from_layer(layer, random_act).nodes
            for target in STATE_NAMES:
                target_idx = STATE_NAMES.index(target)
                null_effect = float((random_nodes[..., target_idx] - base_nodes[..., target_idx]).mean().detach().cpu().item())
                target_nulls[target].append(null_effect)
                evidence_rows.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "feature_id": feature_id,
                        "source_node": node,
                        "target_node": target,
                        "evidence_type": "matched_random_ablation",
                        "sample_id": null_id,
                        "effect": null_effect,
                        "push_effect": np.nan,
                        "logit_effect": np.nan,
                        "probability_effect": np.nan,
                        "null_q99": np.nan,
                        "p_value": np.nan,
                        "q_value": np.nan,
                        "accepted": False,
                    }
                )
        node_nulls = np.asarray(target_nulls[node])
        q99 = float(np.quantile(np.abs(node_nulls), 0.99))
        p_value = float((1 + np.sum(np.abs(node_nulls) >= abs(node_effect))) / (len(node_nulls) + 1))
        for target in STATE_NAMES:
            target_idx = STATE_NAMES.index(target)
            nulls_np = np.asarray(target_nulls[target])
            target_effect = float((after[..., target_idx] - base_nodes[..., target_idx]).mean().detach().cpu().item())
            target_push = float((pushed_nodes[..., target_idx] - base_nodes[..., target_idx]).mean().detach().cpu().item())
            target_q99 = float(np.quantile(np.abs(nulls_np), 0.99))
            target_p = float((1 + np.sum(np.abs(nulls_np) >= abs(target_effect))) / (len(nulls_np) + 1))
            evidence_rows.append(
                {
                    "seed": seed,
                    "layer": layer,
                    "feature_id": feature_id,
                    "source_node": node,
                    "target_node": target,
                    "evidence_type": "true_feature_intervention",
                    "sample_id": -1,
                    "effect": target_effect,
                    "push_effect": target_push,
                    "logit_effect": float((after_forward.logit - base_logit).mean().detach().cpu().item()),
                    "probability_effect": float((after_forward.probability - base_probability).mean().detach().cpu().item()),
                    "null_q99": target_q99,
                    "p_value": target_p,
                    "q_value": np.nan,
                    "accepted": False,
                }
            )
        effects.append(node_effect)
        null_q99s.append(q99)
        p_values.append(p_value)
    q_values = bh_q_values(np.asarray(p_values))
    primary = primary.reset_index(drop=True)
    for idx, row in primary.iterrows():
        corr = float(row["activation_correlation"])
        cosine = float(row["decoder_cosine"])
        accepted = bool(
            corr >= CORRELATION_THRESHOLD
            and cosine >= COSINE_THRESHOLD
            and q_values[idx] <= 0.05
            and abs(effects[idx]) > null_q99s[idx]
            and abs(effects[idx]) > NODE_EFFECT_THRESHOLD
        )
        row = row.copy()
        row["intervention_effect"] = effects[idx]
        row["null_q99"] = null_q99s[idx]
        row["p_value"] = p_values[idx]
        row["q_value"] = float(q_values[idx])
        row["accepted"] = accepted
        if not accepted:
            if cosine < COSINE_THRESHOLD:
                row["match_type"] = "REJECTED_LOW_DIRECTION_COSINE"
            else:
                row["match_type"] = "REJECTED_INTERVENTION"
        updated.append(row)

    non_primary = matches[~matches["match_type"].eq("PRIMARY_HUNGARIAN_MATCH")].copy()
    out = pd.concat([pd.DataFrame(updated), non_primary], ignore_index=True)
    if run_negative_controls:
        neg_rows = []
        for _, row in primary.iterrows():
            wrong_nodes = [n for n in STATE_NAMES if NODE_LAYERS[n] != layer]
            if not wrong_nodes:
                continue
            for control_type, control_node in [
                ("wrong_layer_node_label", wrong_nodes[0]),
                ("permuted_node_label", STATE_NAMES[(STATE_NAMES.index(str(row["node"])) + 2) % len(STATE_NAMES)]),
            ]:
                neg_rows.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "feature_id": int(row["feature_id"]),
                        "source_node": str(row["node"]),
                        "target_node": control_node,
                        "evidence_type": control_type,
                        "sample_id": -1,
                        "effect": np.nan,
                        "push_effect": np.nan,
                        "logit_effect": np.nan,
                        "probability_effect": np.nan,
                        "null_q99": np.nan,
                        "p_value": np.nan,
                        "q_value": np.nan,
                        "accepted": False,
                    }
                )
        evidence_rows.extend(neg_rows)
    evidence = pd.DataFrame(evidence_rows)
    if not evidence.empty:
        mask = evidence["evidence_type"].eq("true_feature_intervention")
        evidence.loc[mask, "q_value"] = bh_q_values(evidence.loc[mask, "p_value"].to_numpy(dtype=float))
        evidence.loc[mask, "accepted"] = (
            (evidence.loc[mask, "q_value"] <= 0.05)
            & (evidence.loc[mask, "effect"].abs() > evidence.loc[mask, "null_q99"])
            & (evidence.loc[mask, "effect"].abs() > NODE_EFFECT_THRESHOLD)
        )
    return out, evidence


def node_recovery_metrics(matches: pd.DataFrame, layer: int) -> dict:
    expected_nodes = {name for name, node_layer in NODE_LAYERS.items() if node_layer == layer}
    accepted = matches[matches.get("accepted", pd.Series(dtype=bool)).fillna(False)] if not matches.empty else pd.DataFrame()
    recovered_nodes = set(accepted["node"].astype(str)) if not accepted.empty else set()
    tp = len(recovered_nodes & expected_nodes)
    fp = len(recovered_nodes - expected_nodes)
    fn = len(expected_nodes - recovered_nodes)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accepted_feature_matches": int(len(accepted)),
        "unique_recovered_nodes": int(tp),
        "redundant_matches": int(matches["match_type"].eq("REDUNDANT_CORRELATED_FEATURE").sum()) if not matches.empty else 0,
        "node_precision": float(precision),
        "node_recall": float(recall),
        "node_f1": float(f1),
    }


def edge_recovery_metrics(interventions: pd.DataFrame, accepted_nodes: pd.DataFrame) -> dict:
    true_edges = {(source, target): float(np.sign(weight)) for source, target, weight in EDGES}
    if interventions.empty or accepted_nodes.empty:
        return {
            "accepted_true_edges": 0,
            "accepted_false_edges": 0,
            "edge_precision": 0.0,
            "edge_recall": 0.0,
            "edge_f1": 0.0,
            "sign_agreement": 0.0,
            "negative_control_fpr": float("nan"),
            "negative_control_status": "NOT_RUN",
            "CircuitF1": 0.0,
        }
    recovered_sources = set(accepted_nodes.loc[accepted_nodes["accepted"].fillna(False), "node"].astype(str))
    true_feature_rows = interventions[
        interventions["evidence_type"].eq("true_feature_intervention")
        & interventions["source_node"].isin(recovered_sources)
    ].copy()
    candidates = true_feature_rows[true_feature_rows["accepted"].fillna(False)].copy()
    accepted_edges: set[tuple[str, str]] = set()
    false_edges: set[tuple[str, str]] = set()
    signs = []
    for _, row in candidates.iterrows():
        edge = (str(row["source_node"]), str(row["target_node"]))
        if edge[0] == edge[1]:
            continue
        predicted_sign = float(np.sign(row["push_effect"])) if np.isfinite(row["push_effect"]) else 0.0
        if edge in true_edges:
            accepted_edges.add(edge)
            signs.append(predicted_sign == true_edges[edge])
        elif predicted_sign != 0:
            false_edges.add(edge)
    tp = len(accepted_edges)
    fp = len(false_edges)
    fn = len(true_edges) - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    controls = true_feature_rows[
        true_feature_rows.apply(lambda r: (str(r["source_node"]), str(r["target_node"])) not in true_edges and str(r["source_node"]) != str(r["target_node"]), axis=1)
    ]
    fpr = float(controls["accepted"].fillna(False).mean()) if len(controls) else float("nan")
    return {
        "accepted_true_edges": int(tp),
        "accepted_false_edges": int(fp),
        "edge_precision": float(precision),
        "edge_recall": float(recall),
        "edge_f1": float(f1),
        "sign_agreement": float(np.mean(signs)) if signs else 0.0,
        "negative_control_fpr": fpr,
        "negative_control_status": "EVALUATED" if len(controls) else "NOT_RUN",
        "CircuitF1": float(0.5 * f1),
    }


def full_gate_status(result: pd.DataFrame, full: bool, run_interventions: bool, run_negative_controls: bool, smoke_reason: str) -> dict:
    if not full:
        smoke_ok = len(result) > 0 and bool(np.isfinite(result.get("delta_AUPRC", pd.Series(dtype=float))).all())
        return {
            "status": "SMOKE_PASS" if smoke_ok else "SMOKE_FAIL",
            "scientific_gate_evaluated": False,
            "reason": smoke_reason,
            "best_final_dead_feature_fraction": float(result["final_dead_feature_fraction"].min()) if len(result) else float("nan"),
            "best_delta_AUPRC": float(result["delta_AUPRC"].min()) if len(result) else float("nan"),
        }
    if not run_interventions or not run_negative_controls:
        return {
            "status": "PLANTED_ADAPTIVE_INCOMPLETE_PROTOCOL",
            "scientific_gate_evaluated": False,
            "reason": "full run requires --run-interventions and --run-negative-controls",
        }
    per_seed = []
    for seed, group in result.groupby("seed"):
        best = group.sort_values(
            ["fidelity_gate_pass", "sparsity_gate_pass", "CircuitF1", "sign_agreement", "negative_control_fpr", "n_features"],
            ascending=[False, False, False, False, True, True],
        ).iloc[0]
        seed_pass = bool(
            best["node_precision"] >= 0.80
            and best["node_recall"] >= 0.80
            and best["edge_precision"] >= 0.80
            and best["edge_recall"] >= 0.80
            and best["CircuitF1"] >= 0.80
            and best["sign_agreement"] >= 0.90
            and best["negative_control_fpr"] <= 0.05
            and best["final_dead_feature_fraction"] < 0.50
            and 8 <= best["final_L0_per_token"] <= 32
            and best["delta_AUROC"] <= 0.01
            and best["delta_AUPRC"] <= 0.01
            and best["probability_MAE"] <= 0.02
        )
        per_seed.append({"seed": int(seed), "pass": seed_pass, "selected_layer": int(best["layer"]), "selected_n_features": int(best["n_features"])})
    pass_count = sum(item["pass"] for item in per_seed)
    return {
        "status": "PLANTED_ADAPTIVE_PASS" if pass_count >= 2 else "PLANTED_ADAPTIVE_NEGATIVE",
        "scientific_gate_evaluated": True,
        "seed_pass_count": int(pass_count),
        "seed_results": per_seed,
        "best_final_dead_feature_fraction": float(result["final_dead_feature_fraction"].min()) if len(result) else float("nan"),
        "best_delta_AUPRC": float(result["delta_AUPRC"].min()) if len(result) else float("nan"),
    }


def run_seed(
    seed: int,
    cfg: dict,
    method_cfg: dict,
    output: Path,
    max_epochs: int | None = None,
    limit_layers: int | None = None,
    limit_candidates: int | None = None,
    run_interventions: bool = False,
    run_negative_controls: bool = False,
) -> pd.DataFrame:
    frame = make_episodes(seed, cfg, "clean")
    arrays = prepare_arrays(split_frame(frame, seed), subset_cols("full_input"))
    model = PlantedCircuitModel(seed=seed, d_model=int(cfg["model"]["d_model"])).to(DEVICE)
    rows = []
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    layers = list(range(4))
    if limit_layers is not None:
        layers = layers[: int(limit_layers)]
    for layer in layers:
        train_act, train_logit, _, _ = flatten_layer(model, arrays["c_train_seq"], layer)
        val_act, val_logit, val_prob, val_nodes = flatten_layer(model, arrays["c_val_seq"], layer)
        calibrator, calibration_metrics = calibrate_planted_head(train_logit.max(axis=1), arrays["y_train"], seed)
        (seed_dir / f"layer{layer}_planted_calibration.json").write_text(json.dumps(calibration_metrics, indent=2), encoding="utf-8")
        calibrated_train_logit = calibrate_np(calibrator, train_logit.max(axis=1))
        calibrated_val_logit = calibrate_np(calibrator, val_logit.max(axis=1))
        calibrated_val_prob = 1.0 / (1.0 + np.exp(-calibrated_val_logit))
        rank_capacity = layer_specific_capacity(
            torch.from_numpy(train_act).float(),
            multiplier=float(method_cfg["adaptive_sctc"]["layer_specific_capacity"]["effective_rank_multiplier"][1]),
            minimum=int(method_cfg["adaptive_sctc"]["layer_specific_capacity"]["min_features"]),
            maximum=int(method_cfg["adaptive_sctc"]["layer_specific_capacity"]["max_features"]),
        )
        candidates = registered_candidate_capacities(rank_capacity, method_cfg)
        if limit_candidates is not None:
            candidates = candidates[: int(limit_candidates)]
        for n_features in candidates:
            target_top_k = target_top_k_for_capacity(n_features, method_cfg)

            def forward_from_layer(repl: torch.Tensor, layer: int = layer) -> torch.Tensor:
                raw = model.downstream_from_layer(layer, repl).logit.max(dim=1).values
                return calibrator(raw)

            train_cfg = AdaptiveSCTCTrainConfig(
                n_features=n_features,
                target_top_k=target_top_k,
                epochs=int(max_epochs or method_cfg["adaptive_sctc"]["epochs"]["max"]),
                patience=int(method_cfg["adaptive_sctc"]["epochs"]["early_stopping_patience"]),
                resample_every_steps=int(method_cfg["adaptive_sctc"]["dead_feature_resampling"]["every_steps"]),
            )
            transcoder, log = train_adaptive_sctc(
                torch.from_numpy(train_act).float(),
                torch.from_numpy(calibrated_train_logit).float(),
                forward_from_layer,
                train_cfg,
                DEVICE,
            )
            transcoder.set_active_top_k(target_top_k)
            ckpt = seed_dir / f"layer{layer}_{n_features}_adaptive.pt"
            torch.save({"model_state_dict": transcoder.state_dict(), "seed": seed, "layer": layer, "n_features": n_features, "top_k": target_top_k}, ckpt)
            log.to_csv(seed_dir / f"layer{layer}_{n_features}_adaptive_training.csv", index=False)
            dummy = np.zeros((val_act.shape[0], val_act.shape[1], 1), dtype=np.float32)
            pred = sctc_fidelity_predictions(
                transcoder,
                val_act,
                dummy,
                arrays["y_val"],
                calibrated_val_logit,
                calibrated_val_prob,
                lambda _x, repl, layer=layer: model.downstream_from_layer(layer, repl).logit.max(dim=1).values,
            )
            pred["reconstructed_logit"] = calibrate_np(calibrator, pred["reconstructed_logit"].to_numpy())
            pred["reconstructed_probability"] = 1.0 / (1.0 + np.exp(-pred["reconstructed_logit"].to_numpy()))
            pred.to_parquet(seed_dir / f"layer{layer}_{n_features}_adaptive_predictions.parquet", index=False)
            met = fidelity_metrics(pred)
            final_stats = final_activation_stats(
                transcoder,
                val_act,
                target_top_k,
                float(method_cfg["adaptive_sctc"]["dead_feature_resampling"]["frequency_threshold"]),
            )
            matches = node_matching(model, transcoder, val_act, val_nodes, layer, seed)
            matches, intervention_evidence = intervention_validate_matches(
                matches,
                model,
                transcoder,
                val_act,
                val_nodes,
                layer,
                seed,
                run_interventions=run_interventions,
                run_negative_controls=run_negative_controls,
                n_nulls=1000 if run_interventions else 0,
            )
            matches.to_parquet(seed_dir / f"layer{layer}_{n_features}_adaptive_node_matching.parquet", index=False)
            intervention_evidence.to_parquet(seed_dir / f"layer{layer}_{n_features}_adaptive_interventions.parquet", index=False)
            node_met = node_recovery_metrics(matches, layer)
            edge_met = edge_recovery_metrics(intervention_evidence, matches)
            edge_met["CircuitF1"] = float(0.5 * node_met["node_f1"] + 0.5 * edge_met["edge_f1"])
            warmup_dead = float(log["dead_feature_fraction"].iloc[0]) if len(log) else float("nan")
            annealed_log = log[log["active_top_k"].astype(int) <= int(target_top_k)] if len(log) else pd.DataFrame()
            after_annealing_reached = bool(len(annealed_log))
            after_annealing_dead = float(annealed_log["dead_feature_fraction"].iloc[-1]) if after_annealing_reached else float("nan")
            fidelity_gate = bool(met["delta_AUROC"] <= 0.01 and met["delta_AUPRC"] <= 0.01 and met["probability_MAE"] <= 0.02)
            sparsity_gate = bool(8 <= final_stats["final_L0_per_token"] <= 32 and final_stats["final_dead_feature_fraction"] < 0.50)
            rows.append(
                {
                    "seed": seed,
                    "layer": layer,
                    "n_features": n_features,
                    "top_k": target_top_k,
                    "effective_rank_capacity": rank_capacity,
                    "candidate_capacities": json.dumps(candidates),
                    "epochs_trained": int(len(log)),
                    "resampling_count": int(log["resampled_features"].sum()) if len(log) else 0,
                    "warmup_dead_feature_fraction": warmup_dead,
                    "after_annealing_reached": after_annealing_reached,
                    "after_annealing_dead_feature_fraction": after_annealing_dead,
                    "dead_feature_fraction": final_stats["final_dead_feature_fraction"],
                    "L0_per_token": final_stats["final_L0_per_token"],
                    "final_top_k_invariant_pass": bool(final_stats["final_L0_per_token"] <= final_stats["final_active_top_k"] + TOPK_TOLERANCE),
                    "fidelity_gate_pass": fidelity_gate,
                    "sparsity_gate_pass": sparsity_gate,
                    "checkpoint": str(ckpt),
                    **final_stats,
                    **node_met,
                    **edge_met,
                    **met,
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medical/v3/full.yaml")
    parser.add_argument("--method-config", default="configs/medical/v3_1/method_improvements.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", default="artifacts/medical/v3_1/planted_adaptive_sctc")
    parser.add_argument("--max-epochs", type=int, help="Runtime override for smoke checks; full registered run omits this.")
    parser.add_argument("--limit-layers", type=int, help="Runtime override for smoke checks; full registered run omits this.")
    parser.add_argument("--limit-candidates", type=int, help="Runtime override for smoke checks; full registered run omits this.")
    parser.add_argument("--full", action="store_true", help="Evaluate the registered scientific gate; requires full layers/candidates and interventions/controls.")
    parser.add_argument("--run-interventions", action="store_true", help="Run downstream-forward feature interventions.")
    parser.add_argument("--run-negative-controls", action="store_true", help="Run explicit wrong-layer/permuted-label negative controls.")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    method_cfg = yaml.safe_load(Path(args.method_config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    smoke_limited = bool(args.max_epochs is not None or args.limit_layers is not None or args.limit_candidates is not None or len(args.seeds) < 3)
    full = bool(args.full and not smoke_limited)
    frames = [
        run_seed(
            seed,
            cfg,
            method_cfg,
            output,
            args.max_epochs,
            args.limit_layers,
            args.limit_candidates,
            run_interventions=args.run_interventions,
            run_negative_controls=args.run_negative_controls,
        )
        for seed in args.seeds
    ]
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output / "planted_adaptive_sctc_grid.csv", index=False)
    if smoke_limited:
        reason = "single seed, limited layers/candidates/epochs, or warm-up top-k smoke; scientific planted gate not evaluated"
    else:
        reason = "full registered run"
    gate = full_gate_status(result, full, args.run_interventions, args.run_negative_controls, reason)
    gate.update(
        {
            "config": str(args.method_config),
            "full_requested": bool(args.full),
            "full_evaluated": full,
            "run_interventions": bool(args.run_interventions),
            "run_negative_controls": bool(args.run_negative_controls),
        }
    )
    (output / "planted_adaptive_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
