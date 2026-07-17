#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fan.sctc.evaluation import fidelity_metrics
from fan.sctc.joint_causal import (
    JointCausalSCTC,
    JointCausalTrainConfig,
    decoder_incoherence,
    train_joint_causal_sctc,
)
from med_circuitbench.planted.model import EDGES, NODE_LAYERS, STATE_NAMES, PlantedCircuitModel
from scripts.medical.v2.run_v2_program import make_episodes, split_frame, subset_cols
from scripts.medical.v3.run_research_program import DEVICE, prepare_arrays, sctc_fidelity_predictions
from scripts.medical.v3_1.run_planted_adaptive_sctc import (
    COSINE_THRESHOLD,
    CORRELATION_THRESHOLD,
    NODE_EFFECT_THRESHOLD,
    binary_metrics,
    bh_q_values,
    calibrate_np,
    calibrate_planted_head,
    edge_recovery_metrics,
    node_recovery_metrics,
)


def flatten_layers(model: PlantedCircuitModel, states: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        out = model(torch.from_numpy(states).float().to(DEVICE))
    return (
        [out.layers[layer].detach().cpu().numpy() for layer in range(4)],
        out.logit.detach().cpu().numpy(),
        out.probability.detach().cpu().numpy(),
        out.nodes.detach().cpu().numpy(),
    )


def behavior_forward_factory(model: PlantedCircuitModel, calibrator):
    def behavior_forward(layer: int, replacement: torch.Tensor) -> torch.Tensor:
        raw = model.downstream_from_layer(layer, replacement).logit.max(dim=1).values
        return calibrator(raw)

    return behavior_forward


def encode_layers(joint: JointCausalSCTC, activations: list[np.ndarray]) -> list[np.ndarray]:
    joint.eval()
    zs = []
    with torch.no_grad():
        for layer, activation in enumerate(activations):
            z = joint.transcoders[layer](torch.from_numpy(activation).float().to(DEVICE))["z"]
            zs.append(z.detach().cpu().numpy())
    return zs


def joint_node_matching(model: PlantedCircuitModel, joint: JointCausalSCTC, activations: list[np.ndarray], nodes: np.ndarray, seed: int, stage: str, config_id: str) -> pd.DataFrame:
    rows = []
    flat_nodes = nodes.reshape(-1, nodes.shape[-1])
    for layer, transcoder in enumerate(joint.transcoders):
        with torch.no_grad():
            z = transcoder(torch.from_numpy(activations[layer]).float().to(DEVICE))["z"].detach().cpu().numpy().reshape(-1, transcoder.n_features)
            decoder = transcoder.original_decoder_directions().detach().cpu().numpy()
        layer_nodes = [name for name, node_layer in NODE_LAYERS.items() if node_layer == layer]
        score = np.zeros((transcoder.n_features, len(layer_nodes)), dtype=np.float64)
        corr_mat = np.zeros_like(score)
        spear_mat = np.zeros_like(score)
        cos_mat = np.zeros_like(score)
        temporal_mat = np.zeros_like(score)
        for feature_id in range(transcoder.n_features):
            feature = z[:, feature_id]
            feature_seq = z.reshape(activations[layer].shape[0], activations[layer].shape[1], transcoder.n_features)[:, :, feature_id].mean(axis=0)
            for node_col, node in enumerate(layer_nodes):
                idx = STATE_NAMES.index(node)
                target = flat_nodes[:, idx]
                corr = 0.0 if np.std(feature) == 0 or np.std(target) == 0 else abs(float(stats.pearsonr(feature, target).statistic))
                spear = 0.0 if np.std(feature) == 0 or np.std(target) == 0 else abs(float(stats.spearmanr(feature, target).statistic))
                target_seq = nodes[:, :, idx].mean(axis=0)
                temporal = 0.0 if np.std(feature_seq) == 0 or np.std(target_seq) == 0 else abs(float(stats.pearsonr(feature_seq, target_seq).statistic))
                direction = model.directions[idx].detach().cpu().numpy()
                cosine = abs(float(np.dot(decoder[feature_id], direction) / ((np.linalg.norm(decoder[feature_id]) + 1e-8) * (np.linalg.norm(direction) + 1e-8))))
                corr_mat[feature_id, node_col] = corr
                spear_mat[feature_id, node_col] = spear
                cos_mat[feature_id, node_col] = cosine
                temporal_mat[feature_id, node_col] = temporal
                score[feature_id, node_col] = corr * cosine * max(temporal, 1e-6)
        row_ind, col_ind = linear_sum_assignment(-score)
        primary = {(int(r), int(c)) for r, c in zip(row_ind, col_ind)}
        for feature_id in range(transcoder.n_features):
            best_col = int(np.argmax(score[feature_id])) if len(layer_nodes) else 0
            for node_col, node in enumerate(layer_nodes):
                corr = corr_mat[feature_id, node_col]
                cosine = cos_mat[feature_id, node_col]
                if (feature_id, node_col) in primary:
                    match_type = "PRIMARY_HUNGARIAN_MATCH"
                elif corr >= CORRELATION_THRESHOLD:
                    match_type = "REDUNDANT_CORRELATED_FEATURE"
                elif corr < CORRELATION_THRESHOLD:
                    match_type = "REJECTED_LOW_CORRELATION"
                elif cosine < COSINE_THRESHOLD:
                    match_type = "REJECTED_LOW_DIRECTION_COSINE"
                else:
                    match_type = "REJECTED_WRONG_LAYER" if node_col != best_col else "REJECTED_LOW_CORRELATION"
                rows.append(
                    {
                        "seed": seed,
                        "stage": stage,
                        "config_id": config_id,
                        "layer": layer,
                        "feature_id": feature_id,
                        "node": node,
                        "activation_correlation": float(corr),
                        "activation_spearman": float(spear_mat[feature_id, node_col]),
                        "decoder_cosine": float(cosine),
                        "temporal_profile_correlation": float(temporal_mat[feature_id, node_col]),
                        "intervention_effect": np.nan,
                        "push_effect": np.nan,
                        "null_q99": np.nan,
                        "p_value": np.nan,
                        "q_value": np.nan,
                        "match_type": match_type,
                        "accepted": False,
                    }
                )
    return pd.DataFrame(rows)


def joint_interventions(
    model: PlantedCircuitModel,
    joint: JointCausalSCTC,
    activations: list[np.ndarray],
    nodes: np.ndarray,
    matches: pd.DataFrame,
    seed: int,
    stage: str,
    config_id: str,
    n_nulls: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    updated = []
    rng = np.random.default_rng(seed + 8400)
    base_nodes = torch.from_numpy(nodes).float().to(DEVICE)
    for _, match in matches[matches["match_type"].eq("PRIMARY_HUNGARIAN_MATCH")].iterrows():
        layer = int(match["layer"])
        feature_id = int(match["feature_id"])
        source_node = str(match["node"])
        source_idx = STATE_NAMES.index(source_node)
        transcoder = joint.transcoders[layer]
        act_t = torch.from_numpy(activations[layer]).float().to(DEVICE)
        with torch.no_grad():
            z = transcoder(act_t)["z"]
            z_ablated = z.clone()
            z_ablated[..., feature_id] = 0.0
            feature_std = z[..., feature_id].std().clamp_min(1e-6)
            z_pushed = z.clone()
            z_pushed[..., feature_id] = z_pushed[..., feature_id] + feature_std
            after = model.downstream_from_layer(layer, transcoder.decode(z_ablated))
            pushed = model.downstream_from_layer(layer, transcoder.decode(z_pushed))
            base = model.downstream_from_layer(layer, act_t)
        nulls_by_target = {target: [] for target in STATE_NAMES}
        for null_id in range(n_nulls):
            rand = torch.from_numpy(rng.normal(size=transcoder.n_features).astype(np.float32)).to(DEVICE)
            rand = rand / rand.norm().clamp_min(1e-8)
            coeff = (z * rand.view(1, 1, -1)).sum(dim=-1, keepdim=True)
            z_random = z - coeff * rand.view(1, 1, -1)
            with torch.no_grad():
                random_after = model.downstream_from_layer(layer, transcoder.decode(z_random))
            for target in STATE_NAMES:
                target_idx = STATE_NAMES.index(target)
                effect = float((random_after.nodes[..., target_idx] - base_nodes[..., target_idx]).mean().detach().cpu().item())
                nulls_by_target[target].append(effect)
                rows.append(
                    {
                        "seed": seed,
                        "stage": stage,
                        "config_id": config_id,
                        "layer": layer,
                        "feature_id": feature_id,
                        "source_node": source_node,
                        "target_node": target,
                        "evidence_type": "matched_random_ablation",
                        "control_type": "random_orthogonal_direction",
                        "sample_id": null_id,
                        "effect": effect,
                        "push_effect": np.nan,
                        "logit_effect": np.nan,
                        "probability_effect": np.nan,
                        "null_q99": np.nan,
                        "p_value": np.nan,
                        "q_value": np.nan,
                        "accepted": False,
                    }
                )
        p_values = []
        true_rows = []
        for target in STATE_NAMES:
            target_idx = STATE_NAMES.index(target)
            effect = float((after.nodes[..., target_idx] - base_nodes[..., target_idx]).mean().detach().cpu().item())
            push_effect = float((pushed.nodes[..., target_idx] - base.nodes[..., target_idx]).mean().detach().cpu().item())
            null_np = np.asarray(nulls_by_target[target])
            q99 = float(np.quantile(np.abs(null_np), 0.99))
            p_value = float((1 + np.sum(np.abs(null_np) >= abs(effect))) / (len(null_np) + 1))
            p_values.append(p_value)
            true_rows.append(
                {
                    "seed": seed,
                    "stage": stage,
                    "config_id": config_id,
                    "layer": layer,
                    "feature_id": feature_id,
                    "source_node": source_node,
                    "target_node": target,
                    "evidence_type": "true_feature_intervention",
                    "control_type": "true_or_non_edge_pair",
                    "sample_id": -1,
                    "effect": effect,
                    "push_effect": push_effect,
                    "logit_effect": float((after.logit - base.logit).mean().detach().cpu().item()),
                    "probability_effect": float((after.probability - base.probability).mean().detach().cpu().item()),
                    "null_q99": q99,
                    "p_value": p_value,
                    "q_value": np.nan,
                    "accepted": False,
                }
            )
        q_values = bh_q_values(np.asarray(p_values))
        for row, q_value in zip(true_rows, q_values):
            accepted = bool(
                q_value <= 0.05
                and abs(row["effect"]) > row["null_q99"]
                and abs(row["effect"]) > NODE_EFFECT_THRESHOLD
                and np.sign(row["push_effect"]) == -np.sign(row["effect"])
            )
            row["q_value"] = float(q_value)
            row["accepted"] = accepted
            rows.append(row)
        node_row = match.copy()
        node_effect = true_rows[source_idx]["effect"]
        node_push = true_rows[source_idx]["push_effect"]
        node_q = true_rows[source_idx]["q_value"]
        node_q99 = true_rows[source_idx]["null_q99"]
        accepted_node = bool(
            match["activation_correlation"] >= CORRELATION_THRESHOLD
            and match["decoder_cosine"] >= COSINE_THRESHOLD
            and node_q <= 0.05
            and abs(node_effect) > node_q99
            and np.sign(node_push) == -np.sign(node_effect)
        )
        node_row["intervention_effect"] = node_effect
        node_row["push_effect"] = node_push
        node_row["null_q99"] = node_q99
        node_row["p_value"] = true_rows[source_idx]["p_value"]
        node_row["q_value"] = node_q
        node_row["accepted"] = accepted_node
        if not accepted_node:
            if match["decoder_cosine"] < COSINE_THRESHOLD:
                node_row["match_type"] = "REJECTED_LOW_DIRECTION_COSINE"
            elif node_q > 0.05 or abs(node_effect) <= node_q99:
                node_row["match_type"] = "REJECTED_NULL_SIGNIFICANCE"
            else:
                node_row["match_type"] = "REJECTED_ABLATION"
        updated.append(node_row)
    non_primary = matches[~matches["match_type"].eq("PRIMARY_HUNGARIAN_MATCH")].copy()
    return pd.concat([pd.DataFrame(updated), non_primary], ignore_index=True), pd.DataFrame(rows)


def transition_tables(joint: JointCausalSCTC, seed: int, stage: str, config_id: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    matrices = {}
    for pair_id, matrix in enumerate(joint.transitions):
        arr = matrix.detach().cpu().numpy()
        matrices[f"A{pair_id}{pair_id + 1}"] = arr
        density = float(np.mean(np.abs(arr) > 1e-3))
        rows.append(
            {
                "seed": seed,
                "stage": stage,
                "config_id": config_id,
                "source_layer": pair_id,
                "target_layer": pair_id + 1,
                "density_abs_gt_1e_3": density,
                "l1_mean": float(np.mean(np.abs(arr))),
                "max_abs": float(np.max(np.abs(arr))),
            }
        )
    return pd.DataFrame(rows), matrices


def evaluate_joint(seed: int, stage: str, config_id: str, cfg: dict, method_cfg: dict, output: Path, train_cfg: JointCausalTrainConfig) -> dict:
    frame = make_episodes(seed, cfg, "clean")
    arrays = prepare_arrays(split_frame(frame, seed), subset_cols("full_input"))
    model = PlantedCircuitModel(seed=seed, d_model=int(cfg["model"]["d_model"])).to(DEVICE)
    train_acts, train_logit, _, _ = flatten_layers(model, arrays["c_train_seq"])
    val_acts, val_logit, val_prob, val_nodes = flatten_layers(model, arrays["c_val_seq"])
    calibrator, calibration = calibrate_planted_head(train_logit.max(axis=1), arrays["y_train"], seed)
    calibrated_train_logit = calibrate_np(calibrator, train_logit.max(axis=1))
    calibrated_val_logit = calibrate_np(calibrator, val_logit.max(axis=1))
    calibrated_val_prob = 1.0 / (1.0 + np.exp(-calibrated_val_logit))
    seed_dir = output / stage / f"seed_{seed}" / config_id
    seed_dir.mkdir(parents=True, exist_ok=True)
    joint, train_log = train_joint_causal_sctc(
        [torch.from_numpy(x).float() for x in train_acts],
        behavior_forward_factory(model, calibrator),
        torch.from_numpy(calibrated_train_logit).float(),
        train_cfg,
        DEVICE,
    )
    train_log["seed"] = seed
    train_log["config_id"] = config_id
    train_log.to_parquet(seed_dir / "joint_training_metrics.parquet", index=False)
    torch.save({"model_state_dict": joint.state_dict(), "config": asdict(train_cfg), "seed": seed}, seed_dir / "joint_causal_sctc.pt")

    dummy = np.zeros((val_acts[3].shape[0], val_acts[3].shape[1], 1), dtype=np.float32)
    pred = sctc_fidelity_predictions(
        joint.transcoders[3],
        val_acts[3],
        dummy,
        arrays["y_val"],
        calibrated_val_logit,
        calibrated_val_prob,
        lambda _x, repl: model.downstream_from_layer(3, repl).logit.max(dim=1).values,
    )
    pred["reconstructed_logit"] = calibrate_np(calibrator, pred["reconstructed_logit"].to_numpy())
    pred["reconstructed_probability"] = 1.0 / (1.0 + np.exp(-pred["reconstructed_logit"].to_numpy()))
    pred.to_parquet(seed_dir / "model_predictions.parquet", index=False)
    fidelity = fidelity_metrics(pred)

    activity_rows = []
    reconstruction_rows = []
    with torch.no_grad():
        for layer, transcoder in enumerate(joint.transcoders):
            out = transcoder(torch.from_numpy(val_acts[layer]).float().to(DEVICE))
            freq = (out["z"] > 0).float().mean(dim=(0, 1)).detach().cpu().numpy()
            for feature_id, f in enumerate(freq):
                activity_rows.append({"seed": seed, "stage": stage, "config_id": config_id, "layer": layer, "feature_id": feature_id, "activation_frequency": float(f)})
            reconstruction_rows.append(
                {
                    "seed": seed,
                    "stage": stage,
                    "config_id": config_id,
                    "layer": layer,
                    "reconstruction_mse": float(torch.mean((out["reconstructed"] - torch.from_numpy(val_acts[layer]).float().to(DEVICE)) ** 2).item()),
                    "L0": float((out["z"] > 0).float().sum(dim=-1).mean().item()),
                    "dead_feature_fraction": float(np.mean(freq < 1e-5)),
                    "n_features": transcoder.n_features,
                    "top_k": transcoder.target_top_k,
                    "white_dim": transcoder.whitening.white_dim,
                }
            )
    pd.DataFrame(activity_rows).to_parquet(seed_dir / "feature_activity.parquet", index=False)
    layer_metrics = pd.DataFrame(reconstruction_rows)
    layer_metrics.to_csv(seed_dir / "layer_reconstruction_metrics.csv", index=False)

    matches = joint_node_matching(model, joint, val_acts, val_nodes, seed, stage, config_id)
    matches, interventions = joint_interventions(model, joint, val_acts, val_nodes, matches, seed, stage, config_id)
    matches.to_parquet(seed_dir / "node_matching.parquet", index=False)
    interventions.to_parquet(seed_dir / "interventions.parquet", index=False)
    node_metrics = []
    for layer in range(4):
        node_metrics.append(node_recovery_metrics(matches[matches["layer"].eq(layer)], layer))
    node_precision = float(np.mean([m["node_precision"] for m in node_metrics]))
    node_recall = float(np.mean([m["node_recall"] for m in node_metrics]))
    node_f1 = float(np.mean([m["node_f1"] for m in node_metrics]))
    edge_metrics = edge_recovery_metrics(interventions, matches)
    circuit_f1 = 0.5 * node_f1 + 0.5 * edge_metrics["edge_f1"]
    trans_df, matrices = transition_tables(joint, seed, stage, config_id)
    trans_df.to_csv(seed_dir / "transition_sparsity.csv", index=False)
    np.savez(seed_dir / "transition_matrices.npz", **matrices)
    incoherence = float(np.mean([float(decoder_incoherence(t).detach().cpu()) for t in joint.transcoders]))
    mean_l0 = float(layer_metrics["L0"].mean())
    aggregate_l0 = float(layer_metrics["L0"].sum())
    dead = float(layer_metrics["dead_feature_fraction"].mean())
    transition_density = float(trans_df["density_abs_gt_1e_3"].mean())
    row = {
        "seed": seed,
        "stage": stage,
        "config_id": config_id,
        **asdict(train_cfg),
        **fidelity,
        "mean_L0": mean_l0,
        "aggregate_L0": aggregate_l0,
        "dead_feature_fraction": dead,
        "decoder_incoherence": incoherence,
        "transition_density": transition_density,
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": node_f1,
        **edge_metrics,
        "CircuitF1": circuit_f1,
        "fidelity_gate_pass": bool(fidelity["delta_AUROC"] <= 0.01 and fidelity["delta_AUPRC"] <= 0.01 and fidelity["probability_MAE"] <= 0.02),
        "sparsity_gate_pass": bool(8 <= aggregate_l0 <= 32 and dead < 0.50),
        "checkpoint": str(seed_dir / "joint_causal_sctc.pt"),
    }
    pd.DataFrame([row]).to_csv(seed_dir / "recovery_metrics.csv", index=False)
    return row


def stage_configs(method_cfg: dict) -> list[tuple[str, str, JointCausalTrainConfig]]:
    base = method_cfg["joint_causal_sctc"]
    train = base["training"]
    configs = []
    for multiplier in base["stages"]["B1_compact_incoherent"]["capacity_multiplier"]:
        for incoh in base["stages"]["B1_compact_incoherent"]["lambda_incoherence"]:
            cfg = JointCausalTrainConfig(
                stage="B1_compact_incoherent",
                capacity_multiplier=float(multiplier),
                lambda_incoherence=float(incoh),
                lambda_decorrelation=float(base["stages"]["B1_compact_incoherent"]["lambda_decorrelation"][0]),
                epochs=int(train["epochs"]),
                batch_size=int(train["batch_size"]),
                learning_rate=float(train["learning_rate"]),
            )
            configs.append((cfg.stage, f"m{multiplier}_inc{incoh}", cfg))
    for value in base["stages"]["B2_joint_transition"]["lambda_transition"]:
        cfg = JointCausalTrainConfig(
            stage="B2_joint_transition",
            capacity_multiplier=2.0,
            lambda_incoherence=0.01,
            lambda_decorrelation=0.001,
            lambda_transition=float(value),
            lambda_edge_sparse=float(base["stages"]["B2_joint_transition"]["lambda_edge_sparse"][0]),
            epochs=int(train["epochs"]),
            batch_size=int(train["batch_size"]),
            learning_rate=float(train["learning_rate"]),
        )
        configs.append((cfg.stage, f"tr{value}", cfg))
    for value in base["stages"]["B3_interventional_causal"]["lambda_interventional"]:
        cfg = JointCausalTrainConfig(
            stage="B3_interventional_causal",
            capacity_multiplier=2.0,
            lambda_incoherence=0.01,
            lambda_decorrelation=0.001,
            lambda_transition=0.03,
            lambda_edge_sparse=0.001,
            lambda_interventional=float(value),
            epochs=int(train["epochs"]),
            batch_size=int(train["batch_size"]),
            learning_rate=float(train["learning_rate"]),
        )
        configs.append((cfg.stage, f"int{value}", cfg))
    return configs[:10]


def aggregate_outputs(output: Path, rows: list[dict], baseline: Path | None) -> dict:
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "recovery_metrics.csv", index=False)
    metrics.to_csv(output / "method_comparison.csv", index=False)
    for artifact, pattern, writer in [
        ("joint_training_metrics.parquet", "*/*/*/joint_training_metrics.parquet", "parquet"),
        ("layer_reconstruction_metrics.csv", "*/*/*/layer_reconstruction_metrics.csv", "csv"),
        ("feature_activity.parquet", "*/*/*/feature_activity.parquet", "parquet"),
        ("node_matching.parquet", "*/*/*/node_matching.parquet", "parquet"),
        ("node_interventions.parquet", "*/*/*/interventions.parquet", "parquet"),
        ("edge_interventions.parquet", "*/*/*/interventions.parquet", "parquet"),
        ("random_null.parquet", "*/*/*/interventions.parquet", "parquet"),
        ("negative_controls.parquet", "*/*/*/interventions.parquet", "parquet"),
        ("transition_sparsity.csv", "*/*/*/transition_sparsity.csv", "csv"),
    ]:
        dfs = []
        for path in output.glob(pattern):
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            if artifact == "edge_interventions.parquet":
                df = df[df.get("evidence_type", "").eq("true_feature_intervention") if "evidence_type" in df else []]
            elif artifact == "random_null.parquet":
                df = df[df.get("evidence_type", "").eq("matched_random_ablation") if "evidence_type" in df else []]
            elif artifact == "negative_controls.parquet":
                if "evidence_type" in df:
                    true_edges = {(source, target) for source, target, _ in EDGES}
                    df = df[df["evidence_type"].eq("true_feature_intervention") & df.apply(lambda r: (str(r["source_node"]), str(r["target_node"])) not in true_edges and str(r["source_node"]) != str(r["target_node"]), axis=1)]
            elif artifact == "node_interventions.parquet":
                if "evidence_type" in df:
                    df = df[df["evidence_type"].eq("true_feature_intervention") & df["source_node"].astype(str).eq(df["target_node"].astype(str))]
            dfs.append(df)
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            if writer == "csv":
                merged.to_csv(output / artifact, index=False)
            else:
                merged.to_parquet(output / artifact, index=False)
    matrix_files = list(output.glob("*/*/*/transition_matrices.npz"))
    if matrix_files:
        shutil.copy2(matrix_files[0], output / "transition_matrices.npz")
    if "decoder_incoherence" in metrics:
        metrics[["seed", "stage", "config_id", "decoder_incoherence"]].to_csv(output / "decoder_incoherence.csv", index=False)
    node_path = output / "node_matching.parquet"
    if node_path.exists():
        node = pd.read_parquet(node_path)
        redundancy = (
            node.groupby(["seed", "stage", "config_id", "layer", "node", "match_type"])
            .size()
            .reset_index(name="count")
        )
        redundancy.to_parquet(output / "feature_redundancy.parquet", index=False)
    training_path = output / "joint_training_metrics.parquet"
    if training_path.exists():
        training = pd.read_parquet(training_path)
        cols = [c for c in ["seed", "stage", "config_id", "epoch", "transition_loss", "edge_sparse_loss"] if c in training.columns]
        training[cols].to_parquet(output / "transition_prediction.parquet", index=False)
    edge_path = output / "edge_interventions.parquet"
    if edge_path.exists():
        edge = pd.read_parquet(edge_path)
        edge[edge.get("stage", pd.Series(dtype=str)).astype(str).str.startswith("B3")].to_parquet(output / "interventional_training_pairs.parquet", index=False)
    best = metrics.sort_values(
        ["fidelity_gate_pass", "sparsity_gate_pass", "CircuitF1", "sign_agreement", "negative_control_fpr", "transition_density"],
        ascending=[False, False, False, False, True, True],
    ).iloc[0]
    selected = output / "selected_checkpoints"
    selected.mkdir(exist_ok=True)
    for seed, group in metrics.groupby("seed"):
        row = group.sort_values(["fidelity_gate_pass", "sparsity_gate_pass", "CircuitF1"], ascending=[False, False, False]).iloc[0]
        src = Path(str(row["checkpoint"]))
        if src.exists():
            shutil.copy2(src, selected / f"seed{int(seed)}_{row['stage']}_{row['config_id']}.pt")
    pass_by_seed = []
    for seed, group in metrics.groupby("seed"):
        row = group.sort_values(["fidelity_gate_pass", "sparsity_gate_pass", "CircuitF1"], ascending=[False, False, False]).iloc[0]
        seed_pass = bool(
            row["fidelity_gate_pass"]
            and row["sparsity_gate_pass"]
            and row["node_precision"] >= 0.80
            and row["node_recall"] >= 0.80
            and row["edge_precision"] >= 0.80
            and row["edge_recall"] >= 0.80
            and row["CircuitF1"] >= 0.80
            and row["sign_agreement"] >= 0.90
            and row["negative_control_fpr"] <= 0.05
            and row["transition_density"] <= 0.25
        )
        pass_by_seed.append({"seed": int(seed), "pass": seed_pass, "stage": row["stage"], "config_id": row["config_id"]})
    pass_count = sum(x["pass"] for x in pass_by_seed)
    status = "JOINT_CAUSAL_SCTC_PASS" if pass_count >= 2 else "UNSUPERVISED_CAUSAL_BASIS_NOT_IDENTIFIABLE"
    if status != "JOINT_CAUSAL_SCTC_PASS":
        if metrics["negative_control_fpr"].max() > 0.05:
            reason = "HIGH_NEGATIVE_CONTROL_FPR"
        elif metrics["CircuitF1"].max() < 0.80:
            reason = "RECOVERY_FAIL"
        else:
            reason = "SPARSITY_OR_FIDELITY_FAIL"
    else:
        reason = "GATE_PASS"
    gate = {
        "status": status,
        "reason": reason,
        "seed_pass_count": int(pass_count),
        "seed_results": pass_by_seed,
        "best_stage": str(best["stage"]),
        "best_config_id": str(best["config_id"]),
        "best_CircuitF1": float(metrics["CircuitF1"].max()),
        "best_negative_control_fpr": float(metrics["negative_control_fpr"].min()),
        "baseline": str(baseline) if baseline else None,
    }
    (output / "final_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return gate


def copy_baseline(baseline: Path, output: Path) -> None:
    dest = output / "B0_adaptive_baseline"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ["planted_adaptive_gate.json", "planted_adaptive_sctc_grid.csv", "planted_adaptive_metrics.csv"]:
        src = baseline / name
        if src.exists():
            shutil.copy2(src, dest / name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medical/v3/full.yaml")
    parser.add_argument("--method-config", default="configs/medical/v3_1/method_improvements.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--baseline", default="artifacts/medical/v3_1/planted_adaptive_sctc")
    parser.add_argument("--output", default="artifacts/medical/v3_1/joint_causal_sctc")
    parser.add_argument("--continue-until-terminal", action="store_true")
    parser.add_argument("--max-configs", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    method_cfg = yaml.safe_load(Path(args.method_config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    baseline = Path(args.baseline)
    copy_baseline(baseline, output)
    configs = stage_configs(method_cfg)
    if args.max_configs is not None:
        configs = configs[: int(args.max_configs)]
    rows = []
    for stage, config_id, train_cfg in configs:
        for seed in args.seeds:
            row = evaluate_joint(seed, stage, config_id, cfg, method_cfg, output, train_cfg)
            rows.append(row)
    gate = aggregate_outputs(output, rows, baseline)
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
