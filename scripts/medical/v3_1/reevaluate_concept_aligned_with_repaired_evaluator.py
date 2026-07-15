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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fan.sctc.concept_aligned import ConceptAlignedInterventionalTrainConfig, build_concept_aligned_model  # noqa: E402
from med_circuitbench.planted.interventions import (  # noqa: E402
    DESCENDANTS,
    DIRECT_EDGES,
    STATE_INDEX,
    descendant_pairs,
    direct_edge_counterfactual,
    standardized_effect,
    target_train_std,
    total_effect_counterfactual,
)
from med_circuitbench.planted.model import STATE_NAMES, PlantedCircuitModel  # noqa: E402
from scripts.medical.v2.run_v2_program import make_episodes, split_frame, subset_cols  # noqa: E402
from scripts.medical.v3.run_research_program import DEVICE, prepare_arrays  # noqa: E402
from scripts.medical.v3_1.repair_planted_causal_evaluator import MARGIN, paired_bootstrap  # noqa: E402
from scripts.medical.v3_1.run_joint_causal_sctc import flatten_layers, joint_node_matching  # noqa: E402
from scripts.medical.v3_1.run_planted_adaptive_sctc import (  # noqa: E402
    COSINE_THRESHOLD,
    CORRELATION_THRESHOLD,
    bh_q_values,
    node_recovery_metrics,
)


def _episode_effect(nodes_after: torch.Tensor, nodes_before: torch.Tensor, target: str, train_std: torch.Tensor) -> np.ndarray:
    return standardized_effect(nodes_after, nodes_before, target, train_std).detach().cpu().numpy().mean(axis=1)


def _graph_metrics(rows: pd.DataFrame, truth: set[tuple[str, str]], prefix: str) -> dict:
    accepted = rows[rows["accepted"].fillna(False)]
    accepted_pairs = {(str(r.source_node), str(r.target_node)) for r in accepted.itertuples()}
    tp = len(accepted_pairs & truth)
    fp = len(accepted_pairs - truth)
    fn = len(truth - accepted_pairs)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    controls = rows[~rows.apply(lambda r: (str(r["source_node"]), str(r["target_node"])) in truth, axis=1)]
    return {
        f"{prefix}_TP": int(tp),
        f"{prefix}_FP": int(fp),
        f"{prefix}_FN": int(fn),
        f"{prefix}_precision": float(precision),
        f"{prefix}_recall": float(recall),
        f"{prefix}_F1": float(f1),
        f"{prefix}_negative_control_fpr": float(controls["accepted"].fillna(False).mean()) if len(controls) else float("nan"),
    }


def load_frozen_model(row: pd.Series, cfg: dict):
    seed = int(row["seed"])
    checkpoint = torch.load(str(row["checkpoint"]), map_location=DEVICE, weights_only=False)
    train_cfg = ConceptAlignedInterventionalTrainConfig(**checkpoint["config"])
    frame = make_episodes(seed, cfg, "clean")
    arrays = prepare_arrays(split_frame(frame, seed), subset_cols("full_input"))
    planted = PlantedCircuitModel(seed=seed, d_model=int(cfg["model"]["d_model"])).to(DEVICE)
    train_acts, _, _, train_nodes = flatten_layers(planted, arrays["c_train_seq"])
    val_acts, _, _, val_nodes = flatten_layers(planted, arrays["c_val_seq"])
    model, _, _ = build_concept_aligned_model([torch.from_numpy(x).float() for x in train_acts], train_cfg)
    model = model.to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return planted, model, train_nodes, val_nodes, train_acts, val_acts


def evaluate_row(row: pd.Series, cfg: dict, output: Path) -> dict:
    seed = int(row["seed"])
    stage = str(row["stage"])
    config_id = str(row["config_id"])
    planted, model, train_nodes_np, val_nodes_np, _, val_acts = load_frozen_model(row, cfg)
    train_nodes = torch.from_numpy(train_nodes_np).float().to(DEVICE)
    val_nodes = torch.from_numpy(val_nodes_np).float().to(DEVICE)
    matches = joint_node_matching(planted, model, val_acts, val_nodes_np, seed, stage, config_id)

    direct_rows = []
    total_rows = []
    updated_matches = []
    for _, match in matches[matches["match_type"].eq("PRIMARY_HUNGARIAN_MATCH")].iterrows():
        layer = int(match["layer"])
        feature_id = int(match["feature_id"])
        source = str(match["node"])
        act_t = torch.from_numpy(val_acts[layer]).float().to(DEVICE)
        transcoder = model.transcoders[layer]
        with torch.no_grad():
            z = transcoder(act_t)["z"]
            z_ablated = z.clone()
            z_ablated[..., feature_id] = 0.0
            feature_std = z[..., feature_id].std().clamp_min(1e-8)
            z_pushed = z.clone()
            z_pushed[..., feature_id] = z_pushed[..., feature_id] + feature_std
            decoded_ablated = transcoder.decode(z_ablated)
            decoded_pushed = transcoder.decode(z_pushed)
            source_after = planted.recover_layer_nodes(layer, decoded_ablated)[source]
            source_push = planted.recover_layer_nodes(layer, decoded_pushed)[source]

        direct_p_values = []
        total_p_values = []
        direct_batch = []
        total_batch = []
        for target in STATE_NAMES:
            if target == source:
                continue
            std = target_train_std(train_nodes, target)
            direct_after = direct_edge_counterfactual(val_nodes, source, target, source_after).nodes
            direct_push = direct_edge_counterfactual(val_nodes, source, target, source_push).nodes
            direct_effect = _episode_effect(direct_after, val_nodes, target, std)
            direct_push_effect = _episode_effect(direct_push, val_nodes, target, std)
            direct_boot = paired_bootstrap(direct_effect, seed + 4000 + 101 * feature_id + 17 * STATE_INDEX[target])
            direct_p_values.append(direct_boot["p_value"])
            direct_batch.append(
                {
                    "seed": seed,
                    "stage": stage,
                    "config_id": config_id,
                    "layer": layer,
                    "feature_id": feature_id,
                    "source_node": source,
                    "target_node": target,
                    "relation": "direct_edge" if (source, target) in DIRECT_EDGES else "direct_non_edge_control",
                    "effect_mean": direct_boot["mean"],
                    "ci_lower": direct_boot["ci_lower"],
                    "ci_upper": direct_boot["ci_upper"],
                    "p_value": direct_boot["p_value"],
                    "q_value": np.nan,
                    "push_effect_mean": float(direct_push_effect.mean()),
                    "accepted": False,
                    "directional_null_status": "NOT_APPLICABLE_LOW_RANK",
                }
            )
            total_after = total_effect_counterfactual(val_nodes, source, source_after).nodes
            total_push = total_effect_counterfactual(val_nodes, source, source_push).nodes
            total_effect = _episode_effect(total_after, val_nodes, target, std)
            total_push_effect = _episode_effect(total_push, val_nodes, target, std)
            total_boot = paired_bootstrap(total_effect, seed + 5000 + 101 * feature_id + 17 * STATE_INDEX[target])
            total_p_values.append(total_boot["p_value"])
            total_batch.append(
                {
                    "seed": seed,
                    "stage": stage,
                    "config_id": config_id,
                    "layer": layer,
                    "feature_id": feature_id,
                    "source_node": source,
                    "target_node": target,
                    "relation": "descendant_total_effect" if target in DESCENDANTS[source] else "non_descendant_control",
                    "effect_mean": total_boot["mean"],
                    "ci_lower": total_boot["ci_lower"],
                    "ci_upper": total_boot["ci_upper"],
                    "p_value": total_boot["p_value"],
                    "q_value": np.nan,
                    "push_effect_mean": float(total_push_effect.mean()),
                    "accepted": False,
                    "directional_null_status": "NOT_APPLICABLE_LOW_RANK",
                }
            )
        for batch, q_values in [(direct_batch, bh_q_values(np.asarray(direct_p_values))), (total_batch, bh_q_values(np.asarray(total_p_values)))]:
            for effect_row, q_value in zip(batch, q_values):
                excludes_zero = effect_row["ci_lower"] > 0.0 or effect_row["ci_upper"] < 0.0
                sign_ok = np.sign(effect_row["push_effect_mean"]) == -np.sign(effect_row["effect_mean"])
                effect_row["q_value"] = float(q_value)
                effect_row["accepted"] = bool(excludes_zero and abs(effect_row["effect_mean"]) > MARGIN and sign_ok and q_value <= 0.05)
        direct_rows.extend(direct_batch)
        total_rows.extend(total_batch)
        own_effect = _episode_effect(total_effect_counterfactual(val_nodes, source, source_after).nodes, val_nodes, source, target_train_std(train_nodes, source))
        own_boot = paired_bootstrap(own_effect, seed + 6000 + feature_id)
        node_row = match.copy()
        node_row["intervention_effect"] = own_boot["mean"]
        node_row["p_value"] = own_boot["p_value"]
        node_row["q_value"] = own_boot["p_value"]
        node_row["accepted"] = bool(
            match["activation_correlation"] >= CORRELATION_THRESHOLD
            and match["decoder_cosine"] >= COSINE_THRESHOLD
            and (own_boot["ci_lower"] > 0.0 or own_boot["ci_upper"] < 0.0)
            and abs(own_boot["mean"]) > MARGIN
        )
        if not bool(node_row["accepted"]):
            node_row["match_type"] = "REJECTED_REPAIRED_NODE_EFFECT"
        updated_matches.append(node_row)

    non_primary = matches[~matches["match_type"].eq("PRIMARY_HUNGARIAN_MATCH")].copy()
    node_df = pd.concat([pd.DataFrame(updated_matches), non_primary], ignore_index=True)
    direct_df = pd.DataFrame(direct_rows)
    total_df = pd.DataFrame(total_rows)
    node_metrics = [node_recovery_metrics(node_df[node_df["layer"].eq(layer)], layer) for layer in range(4)]
    node_precision = float(np.mean([m["node_precision"] for m in node_metrics]))
    node_recall = float(np.mean([m["node_recall"] for m in node_metrics]))
    node_f1 = float(np.mean([m["node_f1"] for m in node_metrics]))
    direct_metrics = _graph_metrics(direct_df, set(DIRECT_EDGES), "direct") if len(direct_df) else {}
    total_metrics = _graph_metrics(total_df, descendant_pairs(), "total") if len(total_df) else {}
    return {
        "row": {
            "seed": seed,
            "stage": stage,
            "config_id": config_id,
            "node_precision": node_precision,
            "node_recall": node_recall,
            "node_f1": node_f1,
            **direct_metrics,
            **total_metrics,
            "fidelity_gate_pass": bool(row["fidelity_gate_pass"]),
            "sparsity_gate_pass": bool(row["sparsity_gate_pass"]),
            "checkpoint": str(row["checkpoint"]),
        },
        "nodes": node_df,
        "direct": direct_df,
        "total": total_df,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medical/v3/full.yaml")
    parser.add_argument("--run-dir", default="artifacts/medical/v3_1/concept_aligned_interventional_sctc")
    parser.add_argument("--output", default="artifacts/medical/v3_1/concept_aligned_interventional_sctc/repaired_recovery")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(run_dir / "recovery_metrics.csv")
    results = [evaluate_row(row, cfg, output) for _, row in metrics.iterrows()]
    rows = pd.DataFrame([item["row"] for item in results])
    nodes = pd.concat([item["nodes"] for item in results], ignore_index=True)
    direct = pd.concat([item["direct"] for item in results], ignore_index=True)
    total = pd.concat([item["total"] for item in results], ignore_index=True)
    rows.to_csv(output / "repaired_recovery_metrics.csv", index=False)
    nodes.to_parquet(output / "repaired_node_matching.parquet", index=False)
    direct.to_parquet(output / "repaired_direct_edge_effects.parquet", index=False)
    total.to_parquet(output / "repaired_total_effects.parquet", index=False)

    correct = rows[rows["stage"].eq("correct_concepts")]
    controls = rows[rows["stage"].isin(["permuted_concepts", "random_targets"])]
    correct_direct = float(correct["direct_F1"].mean()) if "direct_F1" in correct and len(correct) else 0.0
    control_direct = float(controls["direct_F1"].max()) if "direct_F1" in controls and len(controls) else 0.0
    if len(correct) and (correct["direct_F1"] >= 0.80).sum() >= 2 and correct_direct > control_direct:
        status = "WEAK_CONCEPT_ALIGNMENT_IMPROVES_MECHANISTIC_RECOVERY"
    elif correct_direct > control_direct or (len(correct) and correct["node_recall"].mean() > controls["node_recall"].mean()):
        status = "WEAK_ALIGNMENT_IMPROVES_SEMANTIC_LOCALIZATION_BUT_NOT_STRICT_DIRECT_EDGE_RECOVERY"
    else:
        status = "CAUSAL_BASIS_NOT_RECOVERED_WITH_WEAK_ALIGNMENT"
    gate = {
        "status": status,
        "correct_mean_direct_F1": correct_direct,
        "control_best_direct_F1": control_direct,
        "correct_mean_total_F1": float(correct["total_F1"].mean()) if "total_F1" in correct and len(correct) else 0.0,
        "control_best_total_F1": float(controls["total_F1"].max()) if "total_F1" in controls and len(controls) else 0.0,
        "oracle_evaluator_required_status": "ORACLE_CAUSAL_EVALUATOR_PASS",
        "model_selection_performed": False,
    }
    (output / "repaired_recovery_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
