#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, replace
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
    ConceptAlignedInterventionalSCTC,
    ConceptAlignedInterventionalTrainConfig,
    LAYER_CONCEPT_INDICES,
    build_concept_aligned_model,
    correlation_matrix,
    sinkhorn,
    train_concept_aligned_interventional_sctc,
)
from fan.sctc.evaluation import fidelity_metrics  # noqa: E402
from fan.sctc.joint_causal import decoder_incoherence  # noqa: E402
from med_circuitbench.planted.model import EDGES, STATE_NAMES, PlantedCircuitModel  # noqa: E402
from scripts.medical.v2.run_v2_program import make_episodes, split_frame, subset_cols  # noqa: E402
from scripts.medical.v3.run_research_program import DEVICE, prepare_arrays, sctc_fidelity_predictions  # noqa: E402
from scripts.medical.v3_1.run_joint_causal_sctc import (  # noqa: E402
    flatten_layers,
    joint_interventions,
    joint_node_matching,
    transition_tables,
)
from scripts.medical.v3_1.run_planted_adaptive_sctc import (  # noqa: E402
    calibrate_np,
    calibrate_planted_head,
    edge_recovery_metrics,
    node_recovery_metrics,
)


def behavior_forward_factory(model: PlantedCircuitModel, calibrator):
    def behavior_forward(layer: int, replacement: torch.Tensor) -> torch.Tensor:
        raw = model.downstream_from_layer(layer, replacement).logit.max(dim=1).values
        return calibrator(raw)

    return behavior_forward


def downstream_activation_forward_factory(model: PlantedCircuitModel):
    def downstream_activation(layer: int, replacement: torch.Tensor) -> torch.Tensor:
        if layer >= 3:
            raise ValueError("layer 3 has no downstream transition target")
        return model.downstream_from_layer(layer, replacement).layers[layer + 1]

    return downstream_activation


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def concept_alignment_metrics(
    joint: ConceptAlignedInterventionalSCTC,
    activations: list[np.ndarray],
    concepts: np.ndarray,
    seed: int,
    stage: str,
    config_id: str,
) -> pd.DataFrame:
    rows = []
    joint.eval()
    with torch.no_grad():
        for layer, transcoder in enumerate(joint.transcoders):
            act = torch.from_numpy(activations[layer]).float().to(DEVICE)
            z = transcoder(act)["z"]
            pred = joint.concept_readouts[layer](z).detach().cpu().numpy()
            target = concepts[..., list(LAYER_CONCEPT_INDICES[layer])]
            corr = correlation_matrix(z, torch.from_numpy(target).float().to(DEVICE)).detach().cpu()
            assign = sinkhorn(corr.abs() / 0.10).detach().cpu().numpy()
            for local_idx, concept_idx in enumerate(LAYER_CONCEPT_INDICES[layer]):
                flat_true = target[..., local_idx].reshape(-1)
                flat_pred = pred[..., local_idx].reshape(-1)
                pearson = 0.0 if np.std(flat_true) == 0 or np.std(flat_pred) == 0 else float(stats.pearsonr(flat_true, flat_pred).statistic)
                rows.append(
                    {
                        "seed": seed,
                        "stage": stage,
                        "config_id": config_id,
                        "layer": layer,
                        "concept": STATE_NAMES[concept_idx],
                        "R2": _r2(flat_true, flat_pred),
                        "Pearson": pearson,
                        "MAE": float(np.mean(np.abs(flat_true - flat_pred))),
                        "matching_score": float(np.max(np.abs(corr.numpy()[:, local_idx]))),
                        "assignment_mass": float(assign[:, local_idx].sum()),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_concept_aligned(
    seed: int,
    stage: str,
    config_id: str,
    cfg: dict,
    train_cfg: ConceptAlignedInterventionalTrainConfig,
    output: Path,
) -> dict:
    frame = make_episodes(seed, cfg, "clean")
    arrays = prepare_arrays(split_frame(frame, seed), subset_cols("full_input"))
    planted = PlantedCircuitModel(seed=seed, d_model=int(cfg["model"]["d_model"])).to(DEVICE)
    train_acts, train_logit, _, train_nodes = flatten_layers(planted, arrays["c_train_seq"])
    val_acts, val_logit, val_prob, val_nodes = flatten_layers(planted, arrays["c_val_seq"])
    calibrator, calibration = calibrate_planted_head(train_logit.max(axis=1), arrays["y_train"], seed)
    calibrated_train_logit = calibrate_np(calibrator, train_logit.max(axis=1))
    calibrated_val_logit = calibrate_np(calibrator, val_logit.max(axis=1))
    calibrated_val_prob = 1.0 / (1.0 + np.exp(-calibrated_val_logit))

    seed_dir = output / stage / f"seed_{seed}" / config_id
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    joint, train_log = train_concept_aligned_interventional_sctc(
        [torch.from_numpy(x).float() for x in train_acts],
        torch.from_numpy(train_nodes).float(),
        behavior_forward_factory(planted, calibrator),
        downstream_activation_forward_factory(planted),
        torch.from_numpy(calibrated_train_logit).float(),
        train_cfg,
        DEVICE,
        control_seed=seed + 991,
    )
    train_log["seed"] = seed
    train_log["config_id"] = config_id
    train_log.to_parquet(seed_dir / "concept_aligned_training_metrics.parquet", index=False)
    torch.save(
        {"model_state_dict": joint.state_dict(), "config": asdict(train_cfg), "seed": seed, "stage": stage, "config_id": config_id},
        seed_dir / "concept_aligned_interventional_sctc.pt",
    )

    dummy = np.zeros((val_acts[3].shape[0], val_acts[3].shape[1], 1), dtype=np.float32)
    pred = sctc_fidelity_predictions(
        joint.transcoders[3],
        val_acts[3],
        dummy,
        arrays["y_val"],
        calibrated_val_logit,
        calibrated_val_prob,
        lambda _x, repl: planted.downstream_from_layer(3, repl).logit.max(dim=1).values,
    )
    pred["reconstructed_logit"] = calibrate_np(calibrator, pred["reconstructed_logit"].to_numpy())
    pred["reconstructed_probability"] = 1.0 / (1.0 + np.exp(-pred["reconstructed_logit"].to_numpy()))
    pred.to_parquet(seed_dir / "predictions.parquet", index=False)
    fidelity = fidelity_metrics(pred)

    activity_rows = []
    reconstruction_rows = []
    with torch.no_grad():
        for layer, transcoder in enumerate(joint.transcoders):
            out = transcoder(torch.from_numpy(val_acts[layer]).float().to(DEVICE))
            freq = (out["z"] > 0).float().mean(dim=(0, 1)).detach().cpu().numpy()
            for feature_id, f in enumerate(freq):
                activity_rows.append(
                    {
                        "seed": seed,
                        "stage": stage,
                        "config_id": config_id,
                        "layer": layer,
                        "feature_id": feature_id,
                        "activation_frequency": float(f),
                    }
                )
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

    concept_metrics = concept_alignment_metrics(joint, val_acts, val_nodes, seed, stage, config_id)
    concept_metrics.to_csv(seed_dir / "concept_alignment_metrics.csv", index=False)

    matches = joint_node_matching(planted, joint, val_acts, val_nodes, seed, stage, config_id)
    matches, interventions = joint_interventions(planted, joint, val_acts, val_nodes, matches, seed, stage, config_id)
    matches.to_parquet(seed_dir / "node_matching.parquet", index=False)
    interventions.to_parquet(seed_dir / "interventions.parquet", index=False)

    node_metrics = [node_recovery_metrics(matches[matches["layer"].eq(layer)], layer) for layer in range(4)]
    node_precision = float(np.mean([m["node_precision"] for m in node_metrics]))
    node_recall = float(np.mean([m["node_recall"] for m in node_metrics]))
    node_f1 = float(np.mean([m["node_f1"] for m in node_metrics]))
    edge_metrics = edge_recovery_metrics(interventions, matches)
    circuit_f1 = 0.5 * node_f1 + 0.5 * edge_metrics["edge_f1"]
    trans_df, matrices = transition_tables(joint, seed, stage, config_id)
    trans_df.to_csv(seed_dir / "transition_sparsity.csv", index=False)
    np.savez(seed_dir / "transition_matrices.npz", **matrices)
    row = {
        "seed": seed,
        "stage": stage,
        "config_id": config_id,
        **asdict(train_cfg),
        **fidelity,
        "mean_L0": float(layer_metrics["L0"].mean()),
        "aggregate_L0": float(layer_metrics["L0"].sum()),
        "dead_feature_fraction": float(layer_metrics["dead_feature_fraction"].mean()),
        "decoder_incoherence": float(np.mean([float(decoder_incoherence(t).detach().cpu()) for t in joint.transcoders])),
        "transition_density": float(trans_df["density_abs_gt_1e_3"].mean()),
        "concept_R2": float(concept_metrics["R2"].mean()),
        "concept_Pearson": float(concept_metrics["Pearson"].mean()),
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": node_f1,
        **edge_metrics,
        "CircuitF1": circuit_f1,
        "fidelity_gate_pass": bool(fidelity["delta_AUROC"] <= 0.01 and fidelity["delta_AUPRC"] <= 0.01 and fidelity["probability_MAE"] <= 0.02),
        "sparsity_gate_pass": bool(0 < layer_metrics["L0"].sum() <= 32 and layer_metrics["dead_feature_fraction"].mean() < 0.50),
        "checkpoint": str(seed_dir / "concept_aligned_interventional_sctc.pt"),
    }
    pd.DataFrame([row]).to_csv(seed_dir / "recovery_metrics.csv", index=False)
    return row


def replication_evaluation(rows: pd.DataFrame, cfg: dict, output: Path) -> pd.DataFrame:
    rep_seed = 20260717
    rep_frame = make_episodes(rep_seed, cfg, "clean")
    states = np.stack(rep_frame["states"].map(lambda x: np.asarray(x, dtype=np.float32)).to_numpy())[:, :36, :5]
    y = rep_frame["target"].to_numpy(dtype=np.float32)
    records = []
    prediction_frames = []
    intervention_frames = []
    sctc_frames = []
    for _, row in rows.iterrows():
        seed = int(row["seed"])
        planted = PlantedCircuitModel(seed=seed, d_model=int(cfg["model"]["d_model"])).to(DEVICE)
        rep_acts, rep_logit, rep_prob, rep_nodes = flatten_layers(planted, states)
        train_arrays = prepare_arrays(split_frame(make_episodes(seed, cfg, "clean"), seed), subset_cols("full_input"))
        train_acts, train_logit, _, _ = flatten_layers(planted, train_arrays["c_train_seq"])
        calibrator, _ = calibrate_planted_head(train_logit.max(axis=1), train_arrays["y_train"], seed)
        calibrated_rep_logit = calibrate_np(calibrator, rep_logit.max(axis=1))
        calibrated_rep_prob = 1.0 / (1.0 + np.exp(-calibrated_rep_logit))
        checkpoint = torch.load(str(row["checkpoint"]), map_location=DEVICE, weights_only=False)
        train_cfg = ConceptAlignedInterventionalTrainConfig(**checkpoint["config"])
        joint, _, _ = build_concept_aligned_model([torch.from_numpy(x).float() for x in train_acts], train_cfg)
        joint = joint.to(DEVICE)
        joint.load_state_dict(checkpoint["model_state_dict"])
        joint.eval()
        dummy = np.zeros((rep_acts[3].shape[0], rep_acts[3].shape[1], 1), dtype=np.float32)
        pred = sctc_fidelity_predictions(
            joint.transcoders[3],
            rep_acts[3],
            dummy,
            y,
            calibrated_rep_logit,
            calibrated_rep_prob,
            lambda _x, repl, planted=planted: planted.downstream_from_layer(3, repl).logit.max(dim=1).values,
        )
        pred["reconstructed_logit"] = calibrate_np(calibrator, pred["reconstructed_logit"].to_numpy())
        pred["reconstructed_probability"] = 1.0 / (1.0 + np.exp(-pred["reconstructed_logit"].to_numpy()))
        pred["seed"] = seed
        pred["stage"] = row["stage"]
        pred["config_id"] = row["config_id"]
        pred["replication_seed"] = rep_seed
        prediction_frames.append(pred)
        sctc_frames.append(pred.copy())
        matches = joint_node_matching(planted, joint, rep_acts, rep_nodes, seed, "replication", str(row["config_id"]))
        matches, interventions = joint_interventions(planted, joint, rep_acts, rep_nodes, matches, seed, "replication", str(row["config_id"]))
        interventions["replication_seed"] = rep_seed
        intervention_frames.append(interventions)
        rep_fidelity = fidelity_metrics(pred)
        records.append(
            {
                "replication_seed": rep_seed,
                "episodes": int(states.shape[0]),
                "training_after_replication_generation": False,
                "seed": seed,
                "stage": row["stage"],
                "config_id": row["config_id"],
                **rep_fidelity,
                "validation_CircuitF1": float(row["CircuitF1"]),
                "validation_fidelity_pass": bool(row["fidelity_gate_pass"]),
                "validation_node_recall": float(row["node_recall"]),
                "validation_edge_recall": float(row["edge_recall"]),
                "replication_dataset_target_prevalence": float(y.mean()),
                "replication_dataset_state_sha256": hashlib.sha256(np.ascontiguousarray(states).tobytes()).hexdigest(),
            }
        )
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_parquet(output / "replication_model_predictions.parquet", index=False)
    if sctc_frames:
        pd.concat(sctc_frames, ignore_index=True).to_parquet(output / "replication_sctc_predictions.parquet", index=False)
    if intervention_frames:
        pd.concat(intervention_frames, ignore_index=True).to_parquet(output / "replication_selected_interventions.parquet", index=False)
    rep = pd.DataFrame(records)
    rep.to_csv(output / "replication_metrics.csv", index=False)
    manifest = {
        "seed": rep_seed,
        "episodes": int(states.shape[0]),
        "prevalence": float(y.mean()),
        "models_changed_after_dataset_generation": False,
        "note": "Frozen replication dataset generated after validation training; this table records frozen selected validation checkpoints without additional model selection.",
    }
    (output / "replication_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return rep


def aggregate_outputs(output: Path, rows: list[dict], baselines: dict[str, Path], cfg: dict) -> dict:
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "recovery_metrics.csv", index=False)
    metrics.to_csv(output / "method_comparison.csv", index=False)
    for artifact, pattern, writer in [
        ("concept_aligned_training_metrics.parquet", "*/*/*/concept_aligned_training_metrics.parquet", "parquet"),
        ("concept_alignment_metrics.csv", "*/*/*/concept_alignment_metrics.csv", "csv"),
        ("feature_activity.parquet", "*/*/*/feature_activity.parquet", "parquet"),
        ("layer_reconstruction_metrics.csv", "*/*/*/layer_reconstruction_metrics.csv", "csv"),
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
            if "interventions" in path.name:
                pass
            if artifact == "edge_interventions.parquet" and "evidence_type" in df:
                df = df[df["evidence_type"].eq("true_feature_intervention")]
            elif artifact == "random_null.parquet" and "evidence_type" in df:
                df = df[df["evidence_type"].eq("matched_random_ablation")]
            elif artifact == "negative_controls.parquet" and "evidence_type" in df:
                true_edges = {(source, target) for source, target, _ in EDGES}
                df = df[df["evidence_type"].eq("true_feature_intervention") & df.apply(lambda r: (str(r["source_node"]), str(r["target_node"])) not in true_edges and str(r["source_node"]) != str(r["target_node"]), axis=1)]
            elif artifact == "node_interventions.parquet" and "evidence_type" in df:
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
    if (output / "node_matching.parquet").exists():
        node = pd.read_parquet(output / "node_matching.parquet")
        node.groupby(["seed", "stage", "config_id", "layer", "node", "match_type"]).size().reset_index(name="count").to_parquet(
            output / "feature_redundancy.parquet", index=False
        )
    if (output / "concept_aligned_training_metrics.parquet").exists():
        training = pd.read_parquet(output / "concept_aligned_training_metrics.parquet")
        cols = [c for c in ["seed", "stage", "config_id", "epoch", "transition_loss", "interventional_loss"] if c in training.columns]
        training[cols].to_parquet(output / "transition_prediction.parquet", index=False)
        training[cols].to_parquet(output / "interventional_training_pairs.parquet", index=False)

    selected = output / "selected_checkpoints"
    selected.mkdir(exist_ok=True)
    correct = metrics[metrics["stage"].eq("correct_concepts")].copy()
    if not correct.empty and not correct["fidelity_gate_pass"].all():
        fallback = metrics[metrics["stage"].eq("correct_concepts_fidelity_fallback")]
        if not fallback.empty:
            correct = pd.concat([correct[correct["fidelity_gate_pass"]], fallback], ignore_index=True)
    for seed, group in correct.groupby("seed"):
        row = group.sort_values(["fidelity_gate_pass", "sparsity_gate_pass"], ascending=[False, False]).iloc[0]
        src = Path(str(row["checkpoint"]))
        if src.exists():
            shutil.copy2(src, selected / f"seed{int(seed)}_{row['stage']}_{row['config_id']}.pt")

    control = metrics[metrics["stage"].isin(["permuted_concepts", "random_targets"])]
    correct_best = correct.sort_values(["fidelity_gate_pass", "sparsity_gate_pass"], ascending=[False, False]).groupby("seed").head(1)
    control_best_circuit = float(control["CircuitF1"].max()) if not control.empty else float("nan")
    correct_best_circuit = float(correct_best["CircuitF1"].mean()) if not correct_best.empty else 0.0
    control_separated = bool(np.isnan(control_best_circuit) or correct_best_circuit > control_best_circuit + 0.05)

    seed_results = []
    for seed, group in correct_best.groupby("seed"):
        row = group.iloc[0]
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
        )
        seed_results.append({"seed": int(seed), "pass": seed_pass, "stage": row["stage"], "config_id": row["config_id"]})
    pass_count = sum(item["pass"] for item in seed_results)
    if pass_count >= 2 and control_separated:
        status = "WEAKLY_SUPERVISED_CAUSAL_BASIS_RECOVERED"
        reason = "GATE_PASS"
    elif not control_separated:
        status = "CAUSAL_BASIS_NOT_RECOVERED_WITH_WEAK_ALIGNMENT"
        reason = "CONCEPT_CONTROLS_NOT_SEPARATED"
    else:
        status = "CAUSAL_BASIS_NOT_RECOVERED_WITH_WEAK_ALIGNMENT"
        reason = "RECOVERY_OR_FPR_GATE_FAIL"

    rep = replication_evaluation(correct_best, cfg, output)
    gate = {
        "status": status,
        "reason": reason,
        "seed_pass_count": int(pass_count),
        "seed_results": seed_results,
        "correct_mean_CircuitF1": correct_best_circuit,
        "control_best_CircuitF1": control_best_circuit,
        "control_separated": control_separated,
        "replication_rows": int(len(rep)),
        "baselines": {name: str(path) for name, path in baselines.items()},
    }
    (output / "final_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return gate


def copy_baseline(path: Path, output: Path, name: str) -> None:
    dest = output / name
    dest.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return
    for item in path.glob("*.json"):
        shutil.copy2(item, dest / item.name)
    for item in path.glob("*.csv"):
        shutil.copy2(item, dest / item.name)


def train_configs(method_cfg: dict, max_epochs: int | None = None) -> list[tuple[str, str, ConceptAlignedInterventionalTrainConfig]]:
    base = method_cfg.get("concept_aligned_interventional_sctc", {})
    epochs = int(max_epochs or base.get("training", {}).get("epochs", 25))
    primary = ConceptAlignedInterventionalTrainConfig(
        stage="correct_concepts",
        lambda_concept=float(base.get("lambda_concept", 0.10)),
        lambda_matching=float(base.get("lambda_matching", 0.03)),
        lambda_transition=float(base.get("lambda_transition", 0.03)),
        lambda_interventional=float(base.get("lambda_interventional", 0.03)),
        lambda_incoherence=float(base.get("lambda_incoherence", 0.01)),
        epochs=epochs,
        batch_size=int(base.get("training", {}).get("batch_size", 128)),
        learning_rate=float(base.get("training", {}).get("learning_rate", 0.001)),
    )
    controls = [
        replace(primary, stage="permuted_concepts"),
        replace(primary, stage="random_targets"),
    ]
    return [
        ("correct_concepts", "primary_lam0.10", primary),
        ("permuted_concepts", "control_permuted", controls[0]),
        ("random_targets", "control_random", controls[1]),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medical/v3/full.yaml")
    parser.add_argument("--method-config", default="configs/medical/v3_1/method_improvements.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--baseline-adaptive", default="artifacts/medical/v3_1/planted_adaptive_sctc")
    parser.add_argument("--baseline-joint", default="artifacts/medical/v3_1/joint_causal_sctc")
    parser.add_argument("--output", default="artifacts/medical/v3_1/concept_aligned_interventional_sctc")
    parser.add_argument("--continue-until-terminal", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    method_cfg = yaml.safe_load(Path(args.method_config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    baselines = {"adaptive": Path(args.baseline_adaptive), "joint_causal": Path(args.baseline_joint)}
    for name, path in baselines.items():
        copy_baseline(path, output, f"baseline_{name}")

    rows: list[dict] = []
    for stage, config_id, train_cfg in train_configs(method_cfg, args.max_epochs):
        for seed in args.seeds:
            row = evaluate_concept_aligned(seed, stage, config_id, cfg, train_cfg, output)
            rows.append(row)

    primary = pd.DataFrame([r for r in rows if r["stage"] == "correct_concepts"])
    if not primary.empty and not primary["fidelity_gate_pass"].all():
        fallback_cfg = replace(train_configs(method_cfg, args.max_epochs)[0][2], stage="correct_concepts_fidelity_fallback", lambda_concept=0.03)
        for seed in args.seeds:
            row = evaluate_concept_aligned(seed, fallback_cfg.stage, "fallback_lam0.03", cfg, fallback_cfg, output)
            rows.append(row)

    gate = aggregate_outputs(output, rows, baselines, cfg)
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
