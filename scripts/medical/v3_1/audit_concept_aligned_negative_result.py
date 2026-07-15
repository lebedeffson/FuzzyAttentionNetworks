#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fan.sctc.concept_aligned import (  # noqa: E402
    ConceptAlignedInterventionalTrainConfig,
    build_concept_aligned_model,
    concept_losses,
    make_control_targets,
)
from med_circuitbench.planted.model import EDGES, NODE_LAYERS, STATE_NAMES, PlantedCircuitModel  # noqa: E402
from scripts.medical.v2.run_v2_program import make_episodes, split_frame, subset_cols  # noqa: E402
from scripts.medical.v3.run_research_program import DEVICE, prepare_arrays  # noqa: E402
from scripts.medical.v3_1.run_concept_aligned_interventional_sctc import replication_evaluation  # noqa: E402
from scripts.medical.v3_1.run_joint_causal_sctc import flatten_layers  # noqa: E402
from scripts.medical.v3_1.run_planted_adaptive_sctc import (  # noqa: E402
    COSINE_THRESHOLD,
    CORRELATION_THRESHOLD,
    NODE_EFFECT_THRESHOLD,
    bh_q_values,
    edge_recovery_metrics,
    node_recovery_metrics,
)


def _sha256_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(stats.pearsonr(a, b).statistic)


def oracle_recovery_for_seed(seed: int, cfg: dict, output: Path, n_nulls: int = 1000) -> dict:
    frame = make_episodes(seed, cfg, "clean")
    arrays = prepare_arrays(split_frame(frame, seed), subset_cols("full_input"))
    planted = PlantedCircuitModel(seed=seed, d_model=int(cfg["model"]["d_model"])).to(DEVICE)
    val_acts, _, _, val_nodes = flatten_layers(planted, arrays["c_val_seq"])
    matches = []
    interventions = []
    rng = np.random.default_rng(seed + 17000)
    directions = planted.directions.detach().cpu().numpy()
    true_edges = {(source, target) for source, target, _ in EDGES}
    for layer, activation_np in enumerate(val_acts):
        activation = torch.from_numpy(activation_np).float().to(DEVICE)
        base = planted.downstream_from_layer(layer, activation)
        layer_nodes = [node for node, node_layer in NODE_LAYERS.items() if node_layer == layer]
        for feature_id, node in enumerate(layer_nodes):
            node_idx = STATE_NAMES.index(node)
            direction = torch.from_numpy(directions[node_idx]).float().to(DEVICE)
            coeff = activation @ direction
            target_flat = val_nodes[..., node_idx].reshape(-1)
            coeff_flat = coeff.detach().cpu().numpy().reshape(-1)
            corr = abs(_safe_corr(coeff_flat, target_flat))
            spearman = 0.0 if np.std(coeff_flat) <= 1e-12 or np.std(target_flat) <= 1e-12 else abs(float(stats.spearmanr(coeff_flat, target_flat).statistic))
            z_ablated_activation = activation - coeff.unsqueeze(-1) * direction.view(1, 1, -1)
            feature_std = coeff.std().clamp_min(1e-6)
            z_pushed_activation = activation + feature_std * direction.view(1, 1, -1)
            after = planted.downstream_from_layer(layer, z_ablated_activation)
            pushed = planted.downstream_from_layer(layer, z_pushed_activation)
            nulls_by_target = {target: [] for target in STATE_NAMES}
            for null_id in range(n_nulls):
                rand = torch.from_numpy(rng.normal(size=activation.shape[-1]).astype(np.float32)).to(DEVICE)
                rand = rand - torch.dot(rand, direction) * direction
                rand = rand / rand.norm().clamp_min(1e-8)
                random_coeff = activation @ rand
                random_activation = activation - random_coeff.unsqueeze(-1) * rand.view(1, 1, -1)
                random_after = planted.downstream_from_layer(layer, random_activation)
                for target in STATE_NAMES:
                    target_idx = STATE_NAMES.index(target)
                    effect = float((random_after.nodes[..., target_idx] - base.nodes[..., target_idx]).mean().detach().cpu())
                    nulls_by_target[target].append(effect)
                    interventions.append(
                        {
                            "seed": seed,
                            "layer": layer,
                            "feature_id": feature_id,
                            "source_node": node,
                            "target_node": target,
                            "evidence_type": "matched_random_ablation",
                            "control_type": "oracle_original_space_orthogonal_random",
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
                effect = float((after.nodes[..., target_idx] - base.nodes[..., target_idx]).mean().detach().cpu())
                push_effect = float((pushed.nodes[..., target_idx] - base.nodes[..., target_idx]).mean().detach().cpu())
                null_np = np.asarray(nulls_by_target[target])
                q99 = float(np.quantile(np.abs(null_np), 0.99))
                p_value = float((1 + np.sum(np.abs(null_np) >= abs(effect))) / (len(null_np) + 1))
                p_values.append(p_value)
                true_rows.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "feature_id": feature_id,
                        "source_node": node,
                        "target_node": target,
                        "evidence_type": "true_feature_intervention",
                        "control_type": "oracle_true_or_non_edge_pair",
                        "sample_id": -1,
                        "effect": effect,
                        "push_effect": push_effect,
                        "logit_effect": float((after.logit - base.logit).mean().detach().cpu()),
                        "probability_effect": float((after.probability - base.probability).mean().detach().cpu()),
                        "null_q99": q99,
                        "p_value": p_value,
                        "q_value": np.nan,
                        "accepted": False,
                    }
                )
            q_values = bh_q_values(np.asarray(p_values))
            for row, q_value in zip(true_rows, q_values):
                row["q_value"] = float(q_value)
                row["accepted"] = bool(
                    q_value <= 0.05
                    and abs(row["effect"]) > row["null_q99"]
                    and abs(row["effect"]) > NODE_EFFECT_THRESHOLD
                    and np.sign(row["push_effect"]) == -np.sign(row["effect"])
                )
                interventions.append(row)
            own = true_rows[node_idx]
            accepted_node = bool(corr >= CORRELATION_THRESHOLD and 1.0 >= COSINE_THRESHOLD and own["accepted"])
            matches.append(
                {
                    "seed": seed,
                    "layer": layer,
                    "feature_id": feature_id,
                    "node": node,
                    "activation_correlation": corr,
                    "activation_spearman": spearman,
                    "decoder_cosine": 1.0,
                    "temporal_profile_correlation": 1.0,
                    "intervention_effect": own["effect"],
                    "push_effect": own["push_effect"],
                    "null_q99": own["null_q99"],
                    "p_value": own["p_value"],
                    "q_value": own["q_value"],
                    "match_type": "PRIMARY_HUNGARIAN_MATCH" if accepted_node else "REJECTED_ORACLE_EVALUATOR",
                    "accepted": accepted_node,
                }
            )
    matches_df = pd.DataFrame(matches)
    interventions_df = pd.DataFrame(interventions)
    matches_df.to_parquet(output / f"oracle_seed{seed}_node_matching.parquet", index=False)
    interventions_df.to_parquet(output / f"oracle_seed{seed}_interventions.parquet", index=False)
    node_metrics = [node_recovery_metrics(matches_df[matches_df["layer"].eq(layer)], layer) for layer in range(4)]
    node_precision = float(np.mean([m["node_precision"] for m in node_metrics]))
    node_recall = float(np.mean([m["node_recall"] for m in node_metrics]))
    node_f1 = float(np.mean([m["node_f1"] for m in node_metrics]))
    edge = edge_recovery_metrics(interventions_df, matches_df)
    accepted_non_edges = interventions_df[
        interventions_df["evidence_type"].eq("true_feature_intervention")
        & interventions_df["accepted"].fillna(False)
        & interventions_df.apply(lambda r: (str(r["source_node"]), str(r["target_node"])) not in true_edges and str(r["source_node"]) != str(r["target_node"]), axis=1)
    ]
    return {
        "seed": seed,
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": node_f1,
        **edge,
        "CircuitF1": float(0.5 * node_f1 + 0.5 * edge["edge_f1"]),
        "accepted_transitive_or_non_edge_count": int(len(accepted_non_edges)),
    }


def run_oracle_recovery(cfg: dict, seeds: list[int], output: Path) -> dict:
    rows = [oracle_recovery_for_seed(seed, cfg, output) for seed in seeds]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "oracle_recovery_sanity.csv", index=False)
    pass_rows = (
        (frame["node_precision"] >= 0.99)
        & (frame["node_recall"] >= 0.99)
        & (frame["edge_precision"] >= 0.99)
        & (frame["edge_recall"] >= 0.99)
        & (frame["CircuitF1"] >= 0.99)
        & (frame["sign_agreement"] >= 0.99)
        & (frame["negative_control_fpr"] <= 0.05)
    )
    result = {
        "status": "PASS" if bool(pass_rows.all()) else "FAIL",
        "seed_pass_count": int(pass_rows.sum()),
        "metrics": rows,
        "interpretation": "oracle planted directions pass evaluator" if bool(pass_rows.all()) else "oracle planted directions do not pass evaluator; recovery conclusions require evaluator repair",
    }
    (output / "oracle_recovery_sanity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.detach().pow(2).sum().cpu())
    return float(np.sqrt(total))


def gradient_audit_for_checkpoint(row: pd.Series, cfg: dict) -> dict:
    seed = int(row["seed"])
    checkpoint = torch.load(str(row["checkpoint"]), map_location=DEVICE, weights_only=False)
    train_cfg = ConceptAlignedInterventionalTrainConfig(**checkpoint["config"])
    frame = make_episodes(seed, cfg, "clean")
    arrays = prepare_arrays(split_frame(frame, seed), subset_cols("full_input"))
    planted = PlantedCircuitModel(seed=seed, d_model=int(cfg["model"]["d_model"])).to(DEVICE)
    train_acts, _, _, train_nodes = flatten_layers(planted, arrays["c_train_seq"])
    model, _, _ = build_concept_aligned_model([torch.from_numpy(x).float() for x in train_acts], train_cfg)
    model = model.to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train()
    activations = [torch.from_numpy(x[: train_cfg.batch_size]).float().to(DEVICE) for x in train_acts]
    concepts = torch.from_numpy(train_nodes[: train_cfg.batch_size]).float().to(DEVICE)
    concepts = make_control_targets(concepts, str(row["stage"]), seed + 991)
    for param in model.parameters():
        param.grad = None
    out = model(activations)
    concept, matching, _ = concept_losses(model, out["z"], concepts, train_cfg.matching_temperature)
    concept.backward(retain_graph=True)
    concept_encoder = _grad_norm([p for t in model.transcoders for p in t.encoder.parameters()])
    concept_decoder = _grad_norm([p for t in model.transcoders for p in t.decoder.parameters()])
    concept_readout = _grad_norm([p for r in model.concept_readouts for p in r.parameters()])
    concept_transition = _grad_norm(list(model.transitions))
    for param in model.parameters():
        param.grad = None
    matching.backward()
    matching_encoder = _grad_norm([p for t in model.transcoders for p in t.encoder.parameters()])
    matching_decoder = _grad_norm([p for t in model.transcoders for p in t.decoder.parameters()])
    return {
        "seed": seed,
        "stage": str(row["stage"]),
        "config_id": str(row["config_id"]),
        "concept_loss": float(concept.detach().cpu()),
        "matching_loss": float(matching.detach().cpu()),
        "concept_grad_encoder_norm": concept_encoder,
        "concept_grad_decoder_norm": concept_decoder,
        "concept_grad_readout_norm": concept_readout,
        "concept_grad_transition_norm": concept_transition,
        "matching_grad_encoder_norm": matching_encoder,
        "matching_grad_decoder_norm": matching_decoder,
        "alignment_reaches_encoder": bool(concept_encoder > 1e-10 or matching_encoder > 1e-10),
        "alignment_reaches_decoder": bool(concept_decoder > 1e-10 or matching_decoder > 1e-10),
    }


def run_gradient_audit(run_dir: Path, cfg: dict, output: Path) -> pd.DataFrame:
    metrics = pd.read_csv(run_dir / "recovery_metrics.csv")
    rows = [gradient_audit_for_checkpoint(row, cfg) for _, row in metrics.iterrows()]
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "alignment_gradient_audit.parquet", index=False)
    return frame


def summarize_continuous_metrics(run_dir: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    concept = pd.read_csv(run_dir / "concept_alignment_metrics.csv")
    nodes = pd.read_parquet(run_dir / "node_matching.parquet")
    edges = pd.read_parquet(run_dir / "edge_interventions.parquet")
    true_edges = {(source, target) for source, target, _ in EDGES}
    concept_summary = concept.groupby(["stage", "seed"]).agg(
        concept_R2=("R2", "mean"),
        concept_Pearson=("Pearson", "mean"),
        concept_matching_score=("matching_score", "mean"),
    )
    primary_nodes = nodes[nodes["match_type"].astype(str).str.contains("PRIMARY|REJECTED", regex=True)].copy()
    primary_nodes["hungarian_score"] = (
        primary_nodes["activation_correlation"].abs()
        * primary_nodes["decoder_cosine"].abs()
        * primary_nodes["temporal_profile_correlation"].abs()
    )
    node_summary = primary_nodes.groupby(["stage", "seed"]).agg(
        feature_to_concept_corr=("activation_correlation", "mean"),
        decoder_cosine=("decoder_cosine", "mean"),
        hungarian_matching_score=("hungarian_score", "mean"),
        node_acceptance_rate=("accepted", "mean"),
    )
    true_rows = edges[edges["evidence_type"].eq("true_feature_intervention")].copy()
    true_rows["ablation_margin_over_null"] = true_rows["effect"].abs() - true_rows["null_q99"].abs()
    true_rows["is_true_edge"] = true_rows.apply(lambda r: (str(r["source_node"]), str(r["target_node"])) in true_edges, axis=1)
    edge_summary = true_rows.groupby(["stage", "seed"]).agg(
        ablation_margin_over_null=("ablation_margin_over_null", "mean"),
        best_ablation_margin_over_null=("ablation_margin_over_null", "max"),
        push_effect_abs=("push_effect", lambda x: float(np.mean(np.abs(x)))),
        accepted_fraction=("accepted", "mean"),
    )
    edge_only = true_rows[true_rows["is_true_edge"]].groupby(["stage", "seed"]).agg(
        true_edge_margin=("ablation_margin_over_null", "mean"),
        true_edge_accepted_fraction=("accepted", "mean"),
    )
    summary = concept_summary.join(node_summary, how="outer").join(edge_summary, how="outer").join(edge_only, how="outer").reset_index()
    summary.to_csv(output / "control_separation_metrics.csv", index=False)
    boot_rows = []
    rng = np.random.default_rng(20260715)
    metrics = [col for col in summary.columns if col not in {"stage", "seed"}]
    for control in ["permuted_concepts", "random_targets"]:
        merged = summary[summary["stage"].eq("correct_concepts")].merge(
            summary[summary["stage"].eq(control)],
            on="seed",
            suffixes=("_correct", "_control"),
        )
        for metric in metrics:
            diffs = (merged[f"{metric}_correct"] - merged[f"{metric}_control"]).dropna().to_numpy(dtype=float)
            if diffs.size == 0:
                continue
            samples = [float(rng.choice(diffs, size=diffs.size, replace=True).mean()) for _ in range(2000)]
            boot_rows.append(
                {
                    "control": control,
                    "metric": metric,
                    "observed_delta": float(diffs.mean()),
                    "ci_lower": float(np.quantile(samples, 0.025)),
                    "ci_upper": float(np.quantile(samples, 0.975)),
                    "n_pairs": int(diffs.size),
                }
            )
    boot = pd.DataFrame(boot_rows)
    boot.to_parquet(output / "control_separation_bootstrap.parquet", index=False)
    return summary, boot


def audit_replication_freeze(run_dir: Path, output: Path) -> dict:
    required = [
        "replication_model_predictions.parquet",
        "replication_sctc_predictions.parquet",
        "replication_selected_interventions.parquet",
        "replication_metrics.csv",
        "replication_manifest.json",
    ]
    files = {name: (run_dir / name).exists() for name in required}
    metrics = pd.read_csv(run_dir / "replication_metrics.csv") if files["replication_metrics.csv"] else pd.DataFrame()
    predictions = pd.read_parquet(run_dir / "replication_model_predictions.parquet") if files["replication_model_predictions.parquet"] else pd.DataFrame()
    interventions = pd.read_parquet(run_dir / "replication_selected_interventions.parquet") if files["replication_selected_interventions.parquet"] else pd.DataFrame()
    source = inspect.getsource(replication_evaluation)
    forbidden = ["optimizer.step", ".backward(", "train_concept_aligned_interventional_sctc(", "train_joint_causal_sctc(", "train_adaptive_sctc("]
    forbidden_hits = [item for item in forbidden if item in source]
    fit_hits = [item for item in ["calibrate_planted_head(", "build_concept_aligned_model("] if item in source]
    validation_interventions = pd.read_parquet(run_dir / "edge_interventions.parquet") if (run_dir / "edge_interventions.parquet").exists() else pd.DataFrame()
    effects_differ = None
    if not interventions.empty and not validation_interventions.empty and "effect" in interventions and "effect" in validation_interventions:
        n = min(len(interventions), len(validation_interventions), 5000)
        effects_differ = bool(not np.allclose(interventions["effect"].head(n).to_numpy(), validation_interventions["effect"].head(n).to_numpy()))
    clean_pass = all(files.values()) and not forbidden_hits and bool(effects_differ) and not metrics.empty and bool((metrics["training_after_replication_generation"] == False).all())
    if clean_pass and fit_hits:
        status = "PASS_WITH_FREEZE_RISK"
    elif clean_pass:
        status = "PASS"
    else:
        status = "FAIL"
    audit = {
        "status": status,
        "files": files,
        "prediction_rows": int(len(predictions)),
        "intervention_rows": int(len(interventions)),
        "training_after_replication_generation_all_false": bool((metrics["training_after_replication_generation"] == False).all()) if not metrics.empty else False,
        "forbidden_training_calls_in_replication_function": forbidden_hits,
        "freeze_risk_calls_in_replication_function": fit_hits,
        "replication_effects_differ_from_validation_prefix": effects_differ,
        "dataset_sha256_values": sorted(metrics["replication_dataset_state_sha256"].astype(str).unique().tolist()) if "replication_dataset_state_sha256" in metrics else [],
    }
    (output / "replication_freeze_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def build_validity_gate(oracle: dict, gradients: pd.DataFrame, continuous: pd.DataFrame, bootstrap: pd.DataFrame, replication: dict, output: Path) -> dict:
    concept_correct = continuous[continuous["stage"].eq("correct_concepts")]
    controls = continuous[continuous["stage"].isin(["permuted_concepts", "random_targets"])]
    alignment_grad_pass = bool(gradients["alignment_reaches_encoder"].any())
    decoder_grad_pass = bool(gradients["alignment_reaches_decoder"].any())
    continuous_pass = False
    if not bootstrap.empty:
        candidates = bootstrap[bootstrap["metric"].isin(["concept_R2", "concept_Pearson", "concept_matching_score", "hungarian_matching_score"])]
        continuous_pass = bool((candidates["ci_lower"] > 0).any())
    concept_objective_pass = bool(not concept_correct.empty and not controls.empty and concept_correct["concept_R2"].mean() > controls["concept_R2"].mean())
    if oracle["status"] != "PASS":
        status = "EVALUATION_PROTOCOL_INVALID"
        reason = "ORACLE_PLANTED_DIRECTIONS_FAIL_EVALUATOR"
    elif not alignment_grad_pass:
        status = "ALIGNMENT_SIGNAL_INEFFECTIVE"
        reason = "NO_ALIGNMENT_GRADIENT_TO_ENCODER"
    elif not concept_objective_pass:
        status = "ALIGNMENT_SIGNAL_INEFFECTIVE"
        reason = "CORRECT_CONCEPT_OBJECTIVE_NOT_BETTER_THAN_CONTROLS"
    elif not continuous_pass:
        status = "CAUSAL_BASIS_NOT_RECOVERED_WITH_WEAK_ALIGNMENT"
        reason = "NO_CONTINUOUS_CONTROL_SEPARATION"
    elif replication["status"] != "PASS":
        status = "PROVISIONAL_NEGATIVE_REQUIRES_FREEZE_FIX"
        reason = "REPLICATION_FREEZE_AUDIT_FAILED"
    else:
        status = "CAUSAL_BASIS_NOT_RECOVERED_WITH_WEAK_ALIGNMENT"
        reason = "ORACLE_AND_ALIGNMENT_AUDITS_PASS_RECOVERY_STILL_FAILS"
    gate = {
        "status": status,
        "reason": reason,
        "scientific_gate": {
            "fidelity": "PASS",
            "replication_fidelity": "PASS" if replication.get("prediction_rows", 0) >= 30000 else "FAIL",
            "oracle_evaluator": oracle["status"],
            "alignment_gradient_to_encoder": "PASS" if alignment_grad_pass else "FAIL",
            "alignment_gradient_to_decoder": "PASS" if decoder_grad_pass else "FAIL",
            "correct_concept_objective_vs_controls": "PASS" if concept_objective_pass else "FAIL",
            "continuous_control_separation": "PASS" if continuous_pass else "FAIL",
            "replication_freeze_audit": replication["status"],
            "thresholded_node_recovery": "FAIL",
            "thresholded_edge_recovery": "FAIL",
        },
        "notes": [
            "Decoder gradient from concept loss can be zero because the current concept readout uses encoded features only.",
            "If oracle_evaluator fails, repair recovery/null protocol before strengthening mechanistic negative claims.",
        ],
    }
    (output / "negative_result_validity_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medical/v3/full.yaml")
    parser.add_argument("--run-dir", default="artifacts/medical/v3_1/concept_aligned_interventional_sctc")
    parser.add_argument("--output", default="artifacts/medical/v3_1/concept_aligned_interventional_sctc/diagnostic_closure")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    oracle = run_oracle_recovery(cfg, args.seeds, output)
    gradients = run_gradient_audit(run_dir, cfg, output)
    continuous, bootstrap = summarize_continuous_metrics(run_dir, output)
    replication = audit_replication_freeze(run_dir, output)
    gate = build_validity_gate(oracle, gradients, continuous, bootstrap, replication, output)
    manifest = {
        "run_dir": str(run_dir),
        "output": str(output),
        "artifacts": {
            "oracle_recovery_sanity": "oracle_recovery_sanity.json",
            "alignment_gradient_audit": "alignment_gradient_audit.parquet",
            "control_separation_metrics": "control_separation_metrics.csv",
            "control_separation_bootstrap": "control_separation_bootstrap.parquet",
            "replication_freeze_audit": "replication_freeze_audit.json",
            "negative_result_validity_gate": "negative_result_validity_gate.json",
        },
        "source_run_state_sha256": _sha256_array(pd.read_csv(run_dir / "recovery_metrics.csv").to_numpy(dtype=str)),
    }
    (output / "diagnostic_closure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
