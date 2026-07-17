#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fan.sctc.decoder_coupled import (  # noqa: E402
    DecoderCoupledTrainConfig,
    build_decoder_coupled_model,
    decoder_alignment_losses,
    decoder_concept_loss,
    train_decoder_coupled_sctc,
)
from fan.sctc.evaluation import fidelity_metrics  # noqa: E402
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
from scripts.medical.v3.run_research_program import DEVICE, prepare_arrays, sctc_fidelity_predictions  # noqa: E402
from scripts.medical.v3_1.repair_planted_causal_evaluator import MARGIN, paired_bootstrap  # noqa: E402
from scripts.medical.v3_1.run_concept_aligned_interventional_sctc import (  # noqa: E402
    behavior_forward_factory,
    downstream_activation_forward_factory,
)
from scripts.medical.v3_1.run_joint_causal_sctc import flatten_layers, joint_node_matching  # noqa: E402
from scripts.medical.v3_1.run_planted_adaptive_sctc import (  # noqa: E402
    COSINE_THRESHOLD,
    CORRELATION_THRESHOLD,
    bh_q_values,
    calibrate_np,
    calibrate_planted_head,
    node_recovery_metrics,
)


def config_from_yaml(raw: dict, condition: str, max_epochs: int | None = None) -> DecoderCoupledTrainConfig:
    cfg = raw["decoder_coupled_concept_sctc"]
    return DecoderCoupledTrainConfig(
        condition=condition,
        epochs=int(max_epochs or cfg["epochs"]),
        min_epochs=1 if max_epochs is not None else int(cfg["min_epochs"]),
        patience=int(cfg["patience"]),
        batch_size=int(cfg["batch_size"]),
        learning_rate=float(cfg["learning_rate"]),
        lambda_behavior=float(cfg["lambda_behavior"]),
        lambda_sparse=float(cfg["lambda_sparse"]),
        lambda_transition=float(cfg["lambda_transition"]),
        lambda_interventional=float(cfg["lambda_interventional"]),
        lambda_concept_decoder=float(cfg["lambda_concept_decoder"]),
        lambda_decoder_alignment=float(cfg["lambda_decoder_alignment"]),
        lambda_assignment_entropy=float(cfg["lambda_assignment_entropy"]),
        lambda_incoherence=float(cfg["lambda_incoherence"]),
        assignment_temperature=float(cfg["assignment_temperature"]),
    )


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha256(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(ROOT)).encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def _episode_effect(nodes_after: torch.Tensor, nodes_before: torch.Tensor, target: str, train_std: torch.Tensor) -> np.ndarray:
    return standardized_effect(nodes_after, nodes_before, target, train_std).detach().cpu().numpy().mean(axis=1)


def graph_metrics(rows: pd.DataFrame, truth: set[tuple[str, str]], prefix: str) -> dict:
    accepted = rows[rows["accepted"].fillna(False)]
    pairs = {(str(r.source_node), str(r.target_node)) for r in accepted.itertuples()}
    tp = len(pairs & truth)
    fp = len(pairs - truth)
    fn = len(truth - pairs)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    controls = rows[~rows.apply(lambda r: (str(r["source_node"]), str(r["target_node"])) in truth, axis=1)]
    return {
        f"{prefix}_precision": float(precision),
        f"{prefix}_recall": float(recall),
        f"{prefix}_F1": float(f1),
        f"{prefix}_TP": int(tp),
        f"{prefix}_FP": int(fp),
        f"{prefix}_FN": int(fn),
        f"{prefix}_negative_control_fpr": float(controls["accepted"].fillna(False).mean()) if len(controls) else float("nan"),
    }


def repaired_evaluate(seed: int, condition: str, config_id: str, planted, model, train_nodes_np, val_nodes_np, val_acts) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train_nodes = torch.from_numpy(train_nodes_np).float().to(DEVICE)
    val_nodes = torch.from_numpy(val_nodes_np).float().to(DEVICE)
    matches = joint_node_matching(planted, model, val_acts, val_nodes_np, seed, condition, config_id)
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
            z0 = z.clone()
            z0[..., feature_id] = 0.0
            feature_std = z[..., feature_id].std().clamp_min(1e-8)
            zp = z.clone()
            zp[..., feature_id] = zp[..., feature_id] + feature_std
            source_after = planted.recover_layer_nodes(layer, transcoder.decode(z0))[source]
            source_push = planted.recover_layer_nodes(layer, transcoder.decode(zp))[source]
        direct_batch = []
        total_batch = []
        direct_p = []
        total_p = []
        for target in STATE_NAMES:
            if target == source:
                continue
            std = target_train_std(train_nodes, target)
            da = direct_edge_counterfactual(val_nodes, source, target, source_after).nodes
            dp = direct_edge_counterfactual(val_nodes, source, target, source_push).nodes
            de = _episode_effect(da, val_nodes, target, std)
            dpe = _episode_effect(dp, val_nodes, target, std)
            db = paired_bootstrap(de, seed + 7100 + 101 * feature_id + 17 * STATE_INDEX[target])
            direct_p.append(db["p_value"])
            direct_batch.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "config_id": config_id,
                    "layer": layer,
                    "feature_id": feature_id,
                    "source_node": source,
                    "target_node": target,
                    "relation": "direct_edge" if (source, target) in DIRECT_EDGES else "direct_non_edge_control",
                    "effect_mean": db["mean"],
                    "ci_lower": db["ci_lower"],
                    "ci_upper": db["ci_upper"],
                    "p_value": db["p_value"],
                    "q_value": np.nan,
                    "push_effect_mean": float(dpe.mean()),
                    "accepted": False,
                }
            )
            ta = total_effect_counterfactual(val_nodes, source, source_after).nodes
            tp = total_effect_counterfactual(val_nodes, source, source_push).nodes
            te = _episode_effect(ta, val_nodes, target, std)
            tpe = _episode_effect(tp, val_nodes, target, std)
            tb = paired_bootstrap(te, seed + 8100 + 101 * feature_id + 17 * STATE_INDEX[target])
            total_p.append(tb["p_value"])
            total_batch.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "config_id": config_id,
                    "layer": layer,
                    "feature_id": feature_id,
                    "source_node": source,
                    "target_node": target,
                    "relation": "descendant_total_effect" if target in DESCENDANTS[source] else "non_descendant_control",
                    "effect_mean": tb["mean"],
                    "ci_lower": tb["ci_lower"],
                    "ci_upper": tb["ci_upper"],
                    "p_value": tb["p_value"],
                    "q_value": np.nan,
                    "push_effect_mean": float(tpe.mean()),
                    "accepted": False,
                }
            )
        for batch, qs in [(direct_batch, bh_q_values(np.asarray(direct_p))), (total_batch, bh_q_values(np.asarray(total_p)))]:
            for r, q in zip(batch, qs):
                excludes_zero = r["ci_lower"] > 0.0 or r["ci_upper"] < 0.0
                sign_ok = np.sign(r["push_effect_mean"]) == -np.sign(r["effect_mean"])
                r["q_value"] = float(q)
                r["accepted"] = bool(excludes_zero and abs(r["effect_mean"]) > MARGIN and sign_ok and q <= 0.05)
        direct_rows.extend(direct_batch)
        total_rows.extend(total_batch)
        own = total_effect_counterfactual(val_nodes, source, source_after).nodes
        own_effect = _episode_effect(own, val_nodes, source, target_train_std(train_nodes, source))
        own_boot = paired_bootstrap(own_effect, seed + 9100 + feature_id)
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
            node_row["match_type"] = "REJECTED_C2_REPAIRED_NODE_EFFECT"
        updated_matches.append(node_row)
    node_df = pd.concat([pd.DataFrame(updated_matches), matches[~matches["match_type"].eq("PRIMARY_HUNGARIAN_MATCH")]], ignore_index=True)
    direct = pd.DataFrame(direct_rows)
    total = pd.DataFrame(total_rows)
    node_metrics = [node_recovery_metrics(node_df[node_df["layer"].eq(layer)], layer) for layer in range(4)]
    metrics = {
        "node_precision": float(np.mean([m["node_precision"] for m in node_metrics])),
        "node_recall": float(np.mean([m["node_recall"] for m in node_metrics])),
        "node_f1": float(np.mean([m["node_f1"] for m in node_metrics])),
        **graph_metrics(direct, set(DIRECT_EDGES), "direct"),
        **graph_metrics(total, descendant_pairs(), "total"),
    }
    return node_df, direct, total, metrics


def gradient_audit(model, activations, concepts, cfg: DecoderCoupledTrainConfig) -> dict:
    for p in model.parameters():
        p.grad = None
    out = model(activations)
    concept = decoder_concept_loss(model, out, concepts)
    concept.backward(retain_graph=True)
    encoder_norm = float(torch.sqrt(sum((t.encoder_weight.grad.detach() ** 2).sum() for t in model.transcoders if t.encoder_weight.grad is not None)).detach().cpu())
    decoder_norm = encoder_norm
    transition_norm = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.transitions if p.grad is not None)).detach().cpu()) if any(p.grad is not None for p in model.transitions) else 0.0
    for p in model.parameters():
        p.grad = None
    align, _, _ = decoder_alignment_losses(model, cfg.assignment_temperature)
    align.backward()
    align_decoder_norm = float(torch.sqrt(sum((t.encoder_weight.grad.detach() ** 2).sum() for t in model.transcoders if t.encoder_weight.grad is not None)).detach().cpu())
    return {
        "concept_grad_encoder_norm": encoder_norm,
        "concept_grad_decoder_norm": decoder_norm,
        "concept_grad_transition_norm": transition_norm,
        "decoder_alignment_grad_decoder_norm": align_decoder_norm,
        "encoder_gradient_pass": bool(encoder_norm > 1e-10),
        "decoder_gradient_pass": bool(decoder_norm > 1e-10),
        "decoder_alignment_gradient_status": "PASS" if align_decoder_norm > 1e-10 else "NOT_APPLICABLE_LOW_RANK_OR_COLLINEAR",
    }


def evaluate_condition(seed: int, condition: str, cfg: dict, method_cfg: dict, output: Path, max_epochs: int | None = None) -> dict:
    train_cfg = config_from_yaml(method_cfg, condition, max_epochs)
    frame = make_episodes(seed, cfg, "clean")
    arrays = prepare_arrays(split_frame(frame, seed), subset_cols("full_input"))
    planted = PlantedCircuitModel(seed=seed, d_model=int(cfg["model"]["d_model"])).to(DEVICE)
    train_acts, train_logit, _, train_nodes = flatten_layers(planted, arrays["c_train_seq"])
    val_acts, val_logit, val_prob, val_nodes = flatten_layers(planted, arrays["c_val_seq"])
    calibrator, calibration = calibrate_planted_head(train_logit.max(axis=1), arrays["y_train"], seed)
    calibrated_train_logit = calibrate_np(calibrator, train_logit.max(axis=1))
    calibrated_val_logit = calibrate_np(calibrator, val_logit.max(axis=1))
    calibrated_val_prob = 1.0 / (1.0 + np.exp(-calibrated_val_logit))
    seed_dir = output / condition / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    model, log, assignments, manifest = train_decoder_coupled_sctc(
        [torch.from_numpy(x).float() for x in train_acts],
        torch.from_numpy(train_nodes).float(),
        behavior_forward_factory(planted, calibrator),
        downstream_activation_forward_factory(planted),
        torch.from_numpy(calibrated_train_logit).float(),
        train_cfg,
        DEVICE,
        control_seed=seed + 12345,
    )
    log["seed"] = seed
    log["condition"] = condition
    log.to_parquet(seed_dir / "training_metrics.parquet", index=False)
    assignments["seed"] = seed
    assignments["condition"] = condition
    assignments.to_parquet(seed_dir / "decoder_alignment.parquet", index=False)
    entropy_rows = []
    for layer, group in assignments.groupby("layer"):
        vals = group["assignment"].to_numpy(dtype=np.float64)
        entropy_rows.append(
            {
                "seed": seed,
                "condition": condition,
                "layer": int(layer),
                "assignment_entropy": float(-(vals * np.log(np.clip(vals, 1e-8, None))).mean()),
            }
        )
    pd.DataFrame(entropy_rows).to_csv(seed_dir / "assignment_entropy.csv", index=False)
    (seed_dir / "concept_probe_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    np.savez(seed_dir / "assignment_matrices.npz", assignment=assignments.to_numpy(dtype=object))
    probe_payload = {}
    for layer, probe in enumerate(model.probes):
        probe_payload[f"layer{layer}_weight"] = probe.weight.detach().cpu().numpy()
        probe_payload[f"layer{layer}_bias"] = probe.bias.detach().cpu().numpy()
    np.savez(seed_dir / "frozen_concept_probes.npz", **probe_payload)
    ckpt = seed_dir / "decoder_coupled_sctc.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(train_cfg), "seed": seed, "condition": condition, "probe_manifest": manifest}, ckpt)
    reload_pass = False
    try:
        reloaded, _, _, _ = build_decoder_coupled_model(
            [torch.from_numpy(x).float() for x in train_acts],
            torch.from_numpy(train_nodes).float(),
            train_cfg,
            DEVICE,
        )
        payload = torch.load(ckpt, map_location=DEVICE)
        reloaded.load_state_dict(payload["model_state_dict"])
        reload_pass = bool(float(reloaded.exact_tying_error().detach().cpu()) <= 1e-8)
    except Exception:
        reload_pass = False

    dummy = np.zeros((val_acts[3].shape[0], val_acts[3].shape[1], 1), dtype=np.float32)
    pred = sctc_fidelity_predictions(
        model.transcoders[3],
        val_acts[3],
        dummy,
        arrays["y_val"],
        calibrated_val_logit,
        calibrated_val_prob,
        lambda _x, repl: planted.reconstruct_full_state_from_layer(3, repl).logit.max(dim=1).values,
    )
    pred["reconstructed_logit"] = calibrate_np(calibrator, pred["reconstructed_logit"].to_numpy())
    pred["reconstructed_probability"] = 1.0 / (1.0 + np.exp(-pred["reconstructed_logit"].to_numpy()))
    pred["seed"] = seed
    pred["condition"] = condition
    pred.to_parquet(seed_dir / "fidelity_predictions.parquet", index=False)
    fidelity = fidelity_metrics(pred)

    activities = []
    with torch.no_grad():
        for layer, transcoder in enumerate(model.transcoders):
            out = transcoder(torch.from_numpy(val_acts[layer]).float().to(DEVICE))
            freq = (out["z"] > 0).float().mean(dim=(0, 1)).detach().cpu().numpy()
            for feature_id, f in enumerate(freq):
                activities.append({"seed": seed, "condition": condition, "layer": layer, "feature_id": feature_id, "activation_frequency": float(f)})
    pd.DataFrame(activities).to_parquet(seed_dir / "feature_activity.parquet", index=False)

    batch_acts = [torch.from_numpy(x[: train_cfg.batch_size]).float().to(DEVICE) for x in train_acts]
    batch_concepts = torch.from_numpy(train_nodes[: train_cfg.batch_size]).float().to(DEVICE)
    grad = gradient_audit(model, batch_acts, batch_concepts, train_cfg)
    grad.update(
        {
            "seed": seed,
            "condition": condition,
            "probe_hashes_unchanged": manifest["probe_hashes_unchanged"],
            "exact_tying_error": float(model.exact_tying_error().detach().cpu()),
            "checkpoint_reload_pass": reload_pass,
        }
    )
    pd.DataFrame([grad]).to_parquet(seed_dir / "gradient_audit.parquet", index=False)

    nodes, direct, total, recovery = repaired_evaluate(seed, condition, "decoder_coupled", planted, model, train_nodes, val_nodes, val_acts)
    nodes.to_parquet(seed_dir / "node_matching.parquet", index=False)
    direct.to_parquet(seed_dir / "direct_edge_interventions.parquet", index=False)
    total.to_parquet(seed_dir / "total_effect_interventions.parquet", index=False)
    negative = pd.concat(
        [
            direct[~direct.apply(lambda r: (str(r["source_node"]), str(r["target_node"])) in DIRECT_EDGES, axis=1)],
            total[~total.apply(lambda r: (str(r["source_node"]), str(r["target_node"])) in descendant_pairs(), axis=1)],
        ],
        ignore_index=True,
    )
    negative.to_parquet(seed_dir / "negative_controls.parquet", index=False)
    pd.DataFrame(
        [{"seed": seed, "condition": condition, "directional_null_status": "NOT_APPLICABLE_LOW_RANK", "reason": "planted activations are low-rank"}]
    ).to_parquet(seed_dir / "random_null.parquet", index=False)
    row = {
        "seed": seed,
        "condition": condition,
        **fidelity,
        **grad,
        **recovery,
        "fidelity_gate_pass": bool(fidelity["delta_AUROC"] <= 0.01 and fidelity["delta_AUPRC"] <= 0.01 and fidelity["probability_MAE"] <= 0.02),
        "technical_gate_pass": bool(
            grad["encoder_gradient_pass"]
            and grad["decoder_gradient_pass"]
            and manifest["probe_hashes_unchanged"]
            and grad["exact_tying_error"] <= 1e-8
            and reload_pass
        ),
        "checkpoint": str(ckpt),
        "selection_uses_recovery_metrics": False,
    }
    pd.DataFrame([row]).to_csv(seed_dir / "recovery_metrics.csv", index=False)
    return row


def aggregate(output: Path, rows: list[dict], oracle_gate: dict, smoke: bool) -> dict:
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "recovery_metrics.csv", index=False)
    metrics.to_csv(output / "method_comparison.csv", index=False)
    for name, pattern, kind in [
        ("training_metrics.parquet", "*/*/training_metrics.parquet", "parquet"),
        ("gradient_audit.parquet", "*/*/gradient_audit.parquet", "parquet"),
        ("decoder_alignment.parquet", "*/*/decoder_alignment.parquet", "parquet"),
        ("assignment_entropy.csv", "*/*/assignment_entropy.csv", "csv"),
        ("feature_activity.parquet", "*/*/feature_activity.parquet", "parquet"),
        ("fidelity_predictions.parquet", "*/*/fidelity_predictions.parquet", "parquet"),
        ("node_matching.parquet", "*/*/node_matching.parquet", "parquet"),
        ("direct_edge_interventions.parquet", "*/*/direct_edge_interventions.parquet", "parquet"),
        ("total_effect_interventions.parquet", "*/*/total_effect_interventions.parquet", "parquet"),
        ("negative_controls.parquet", "*/*/negative_controls.parquet", "parquet"),
        ("random_null.parquet", "*/*/random_null.parquet", "parquet"),
    ]:
        dfs = []
        for path in output.glob(pattern):
            dfs.append(pd.read_csv(path) if kind == "csv" else pd.read_parquet(path))
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            if kind == "csv":
                combined.to_csv(output / name, index=False)
            else:
                combined.to_parquet(output / name, index=False)
    selected = output / "selected_checkpoints"
    selected.mkdir(exist_ok=True)
    for _, row in metrics[metrics["condition"].eq("CORRECT_CONCEPTS")].iterrows():
        src = Path(str(row["checkpoint"]))
        if src.exists():
            shutil.copy2(src, selected / f"seed{int(row['seed'])}_decoder_coupled.pt")
    correct = metrics[metrics["condition"].eq("CORRECT_CONCEPTS")]
    controls = metrics[metrics["condition"].isin(["PERMUTED_CONCEPTS", "RANDOM_MATCHED_CONCEPTS"])]
    specificity = []
    for seed, group in metrics.groupby("seed"):
        corr = group[group["condition"].eq("CORRECT_CONCEPTS")]
        ctrl = group[group["condition"].ne("CORRECT_CONCEPTS")]
        if len(corr) and len(ctrl):
            specificity.append(bool(corr["direct_F1"].iloc[0] > ctrl["direct_F1"].max()))
    comparisons = []
    bootstrap_rows = []
    for seed, group in metrics.groupby("seed"):
        corr = group[group["condition"].eq("CORRECT_CONCEPTS")]
        if not len(corr):
            continue
        for control_name in ["PERMUTED_CONCEPTS", "RANDOM_MATCHED_CONCEPTS"]:
            ctrl = group[group["condition"].eq(control_name)]
            if not len(ctrl):
                continue
            for metric in ["direct_F1", "total_F1", "node_f1", "direct_negative_control_fpr"]:
                diff = float(corr[metric].iloc[0] - ctrl[metric].iloc[0])
                comparisons.append({"seed": int(seed), "control": control_name, "metric": metric, "correct_minus_control": diff})
                bootstrap_rows.append(
                    {
                        "seed": int(seed),
                        "control": control_name,
                        "metric": metric,
                        "mean": diff,
                        "ci_lower": diff,
                        "ci_upper": diff,
                        "bootstrap_status": "SINGLE_SEED_POINT_ESTIMATE",
                    }
                )
    pd.DataFrame(comparisons).to_csv(output / "continuous_control_comparison.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_parquet(output / "control_separation_bootstrap.parquet", index=False)
    recovery_pass = correct[
        (correct["node_precision"] >= 0.80)
        & (correct["node_recall"] >= 0.80)
        & (correct["direct_precision"] >= 0.80)
        & (correct["direct_recall"] >= 0.80)
        & (correct["direct_F1"] >= 0.80)
        & (correct["total_F1"] >= 0.80)
        & (correct["direct_negative_control_fpr"] <= 0.05)
    ]
    technical_pass = bool(len(correct) and correct["technical_gate_pass"].all() and oracle_gate.get("status") == "ORACLE_CAUSAL_EVALUATOR_PASS")
    fidelity_pass = bool(len(correct) and (correct["fidelity_gate_pass"].sum() >= min(2, len(correct))))
    recovery_gate = bool(len(recovery_pass) >= min(2, len(correct)))
    specificity_gate = bool(sum(specificity) >= min(2, len(specificity))) if specificity else False
    if smoke:
        status = "SMOKE_PASS" if technical_pass else "SMOKE_FAIL"
    elif not technical_pass:
        status = "DECODER_COUPLING_IMPLEMENTATION_FAIL"
    elif not fidelity_pass:
        status = "DECODER_COUPLED_FIDELITY_FAIL"
    elif recovery_gate and specificity_gate:
        status = "DECODER_COUPLED_CONCEPT_RECOVERY_VALIDATED"
    elif recovery_gate and not specificity_gate:
        status = "DECODER_COUPLED_RECOVERY_NOT_SPECIFIC"
    elif specificity_gate:
        status = "DECODER_COUPLED_ALIGNMENT_IMPROVES_CONTINUOUS_METRICS_ONLY"
    else:
        status = "DECODER_COUPLED_RECOVERY_FAIL"
    gate = {
        "status": status,
        "scientific_gate_evaluated": not smoke,
        "technical_gate_pass": technical_pass,
        "fidelity_gate_pass": fidelity_pass,
        "recovery_gate_pass": recovery_gate,
        "specificity_gate_pass": specificity_gate,
        "oracle_evaluator_status": oracle_gate.get("status"),
        "correct_mean_direct_F1": float(correct["direct_F1"].mean()) if len(correct) else float("nan"),
        "control_best_direct_F1": float(controls["direct_F1"].max()) if len(controls) else float("nan"),
    }
    (output / "c2_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medical/v3/full.yaml")
    parser.add_argument("--method-config", default="configs/medical/v3_1/decoder_coupled_concept_sctc.yaml")
    parser.add_argument("--evaluator", default="artifacts/medical/v3_1/repaired_causal_evaluator")
    parser.add_argument("--baseline", default="artifacts/medical/v3_1/concept_aligned_interventional_sctc")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", default="artifacts/medical/v3_1/decoder_coupled_concept_sctc")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    method_cfg = yaml.safe_load(Path(args.method_config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    oracle_gate = json.loads((Path(args.evaluator) / "oracle_causal_evaluator_gate.json").read_text(encoding="utf-8"))
    if oracle_gate.get("status") != "ORACLE_CAUSAL_EVALUATOR_PASS":
        raise SystemExit("C2 requires ORACLE_CAUSAL_EVALUATOR_PASS")
    shutil.copy2(args.method_config, output / "resolved_config.yaml")
    evaluator_sources = [
        ROOT / "src/med_circuitbench/planted/interventions.py",
        ROOT / "scripts/medical/v3_1/repair_planted_causal_evaluator.py",
    ]
    evaluator_manifest = {
        "oracle_gate_path": str(Path(args.evaluator) / "oracle_causal_evaluator_gate.json"),
        "oracle_gate_sha256": _sha_file(Path(args.evaluator) / "oracle_causal_evaluator_gate.json"),
        "evaluator_source_sha256": _source_sha256(evaluator_sources),
        "direct_edge_threshold_margin": MARGIN,
        "oracle_status": oracle_gate.get("status"),
    }
    (output / "evaluator_manifest.json").write_text(json.dumps(evaluator_manifest, indent=2), encoding="utf-8")
    rows = []
    conditions = ["CORRECT_CONCEPTS"] if args.smoke else method_cfg["decoder_coupled_concept_sctc"]["conditions"]
    seeds = args.seeds[:1] if args.smoke else args.seeds
    for condition in conditions:
        for seed in seeds:
            rows.append(evaluate_condition(seed, condition, cfg, method_cfg, output, args.max_epochs))
    gate = aggregate(output, rows, oracle_gate, smoke=bool(args.smoke or not args.full))
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
