#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.medical.v2.run_v2_1_program import make_sequence_loaders
from scripts.medical.v2.run_v2_program import make_episodes, split_frame, subset_cols
from scripts.medical.v3.run_fan_iteration import ConceptScaler, build_loaders, make_model, set_all_seeds


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_NAMES = ["I", "R", "V", "O", "S"]


def load_scaler(path: Path) -> ConceptScaler:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ConceptScaler(
        kind=raw["kind"],
        center=np.asarray(raw["center"], dtype=np.float64),
        scale=np.asarray(raw["scale"], dtype=np.float64),
        lower=None if raw.get("lower") is None else np.asarray(raw["lower"], dtype=np.float64),
        upper=None if raw.get("upper") is None else np.asarray(raw["upper"], dtype=np.float64),
    )


def subset_masks(n_concepts: int) -> list[tuple[int, list[int], np.ndarray]]:
    out = []
    for mask_id in range(2**n_concepts):
        indices = [idx for idx in range(n_concepts) if mask_id & (1 << idx)]
        mask = np.zeros(n_concepts, dtype=bool)
        mask[indices] = True
        out.append((mask_id, indices, mask))
    return out


def mask_to_string(mask: np.ndarray) -> str:
    return "".join("1" if bool(v) else "0" for v in mask.tolist())


def prepare_seed(cfg: dict, seed: int, spec: dict, seed_dir: Path):
    set_all_seeds(seed)
    clean = make_episodes(seed, cfg, "clean")
    split = split_frame(clean, seed)
    _, _, arrays = make_sequence_loaders(split, subset_cols("full_input"), int(cfg["training"]["batch_size"]), 5)
    scaler = load_scaler(seed_dir / "concept_scaler.json")
    _, val_loader, scaled = build_loaders(arrays, scaler, 5, int(cfg["training"]["batch_size"]))
    episode_ids = split["validation"]["episode_id"].to_numpy(dtype=np.int64)
    model_cfg = dict(cfg)
    model_cfg["fan"] = dict(cfg.get("fan", {}))
    model_cfg["fan"]["alpha_mode"] = str(spec.get("alpha_mode", model_cfg["fan"].get("alpha_mode", "softmax_alpha")))
    model_cfg["fan"]["gamma_init"] = float(spec.get("gamma_init", model_cfg["fan"].get("gamma_init", 0.5)))
    model_cfg["fan"]["gamma_max"] = float(spec.get("gamma_max", model_cfg["fan"].get("gamma_max", 0.9)))
    model = make_model(
        model_cfg,
        5,
        str(spec["membership_family"]),
        int(spec["n_memberships"]),
        float(spec.get("temperature_init", 1.0)),
        float(arrays["y_train"].mean()),
    )
    checkpoint = torch.load(seed_dir / "oracle_fan_checkpoint.pt", map_location=DEVICE)
    if "decision_head.weight" in checkpoint and "decision_head.raw_weight" not in checkpoint:
        checkpoint["decision_head.raw_weight"] = checkpoint.pop("decision_head.weight")
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model, val_loader, episode_ids, scaled


def batch_subset_outputs(
    model,
    out,
    masks: list[tuple[int, list[int], np.ndarray]],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    bsz, n_concepts = out.concept_evidence.shape
    frozen_logits: dict[int, np.ndarray] = {}
    frozen_probs: dict[int, np.ndarray] = {}
    functional_logits: dict[int, np.ndarray] = {}
    functional_probs: dict[int, np.ndarray] = {}
    for mask_id, _, mask_np in masks:
        mask = torch.from_numpy(mask_np).to(device=out.concept_evidence.device).view(1, n_concepts).expand(bsz, -1)
        frozen_evidence = out.concept_evidence * mask.to(dtype=out.concept_evidence.dtype)
        frozen_logit = model.decision_from_evidence(frozen_evidence)
        if mask_np.any():
            functional_out = model.forward_from_summaries(
                out.latent_sequence,
                out.concept_trajectories,
                out.concept_summaries,
                out.temporal_concept_weights,
                concept_mask=mask,
            )
            functional_logit = functional_out.logit
        else:
            functional_logit = out.logit.new_full((bsz,), float(model.decision_head.bias.detach().cpu()))
        frozen_logits[mask_id] = frozen_logit.detach().cpu().numpy()
        frozen_probs[mask_id] = torch.sigmoid(frozen_logit).detach().cpu().numpy()
        functional_logits[mask_id] = functional_logit.detach().cpu().numpy()
        functional_probs[mask_id] = torch.sigmoid(functional_logit).detach().cpu().numpy()
    return frozen_logits, frozen_probs, functional_logits, functional_probs


def shapley_from_subsets(values: dict[int, np.ndarray], n_concepts: int) -> np.ndarray:
    n = len(next(iter(values.values())))
    phi = np.zeros((n, n_concepts), dtype=np.float64)
    denom = math.factorial(n_concepts)
    for concept_idx in range(n_concepts):
        bit = 1 << concept_idx
        for subset_id in range(2**n_concepts):
            if subset_id & bit:
                continue
            subset_size = int(subset_id.bit_count())
            weight = math.factorial(subset_size) * math.factorial(n_concepts - subset_size - 1) / denom
            phi[:, concept_idx] += weight * (values[subset_id | bit] - values[subset_id])
    return phi


def ranking_orders(out, shapley_logit: np.ndarray) -> dict[str, np.ndarray]:
    alpha = out.concept_weights.detach().cpu().numpy()
    evidence = out.concept_evidence.detach().cpu().numpy()
    signed = out.signed_decision_contributions.detach().cpu().numpy()
    probs = out.probability.detach().cpu().numpy()
    predicted_sign = np.where(probs[:, None] >= 0.5, 1.0, -1.0)
    return {
        "alpha": np.argsort(-alpha, axis=1),
        "concept_evidence": np.argsort(-evidence, axis=1),
        "absolute_signed_contribution": np.argsort(-np.abs(signed), axis=1),
        "predicted_class_signed_contribution": np.argsort(-(predicted_sign * signed), axis=1),
        "positive_class_signed_contribution": np.argsort(-signed, axis=1),
        "absolute_shapley_logit": np.argsort(-np.abs(shapley_logit), axis=1),
        "predicted_class_shapley": np.argsort(-(predicted_sign * shapley_logit), axis=1),
        "positive_class_shapley": np.argsort(-shapley_logit, axis=1),
        "exact_shapley_logit": np.argsort(-shapley_logit, axis=1),
    }


def ids_from_order(order: np.ndarray, k: int) -> np.ndarray:
    masks = np.zeros(order.shape[0], dtype=np.int64)
    for row_idx, concepts in enumerate(order[:, :k]):
        for concept_idx in concepts:
            masks[row_idx] |= 1 << int(concept_idx)
    return masks


def mean_values_for_episode_masks(values: dict[int, np.ndarray], mask_ids: np.ndarray) -> np.ndarray:
    out = np.zeros(len(mask_ids), dtype=np.float64)
    for row_idx, mask_id in enumerate(mask_ids):
        out[row_idx] = values[int(mask_id)][row_idx]
    return out


def sufficient_mask(probs: np.ndarray, logits: np.ndarray, full_probs: np.ndarray, full_logits: np.ndarray) -> np.ndarray:
    return (
        ((probs >= 0.5) == (full_probs >= 0.5))
        & (np.abs(probs - full_probs) <= 0.05)
        & (np.abs(logits - full_logits) <= 0.25)
    )


def exact_random_metric_baseline(
    values: dict[int, np.ndarray],
    logits: dict[int, np.ndarray],
    subset_size: int,
    n_concepts: int,
    full_probs: np.ndarray,
    full_logits: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    matching = [mask_id for mask_id in values if int(mask_id).bit_count() == subset_size]
    maes, logit_maes, auprcs, deltas, sufficient = [], [], [], [], []
    for mask_id in matching:
        probs = values[mask_id]
        sub_logits = logits[mask_id]
        maes.append(float(np.mean(np.abs(probs - full_probs))))
        logit_maes.append(float(np.mean(np.abs(sub_logits - full_logits))))
        deltas.append(float(np.mean(np.abs(full_probs - probs))))
        auprcs.append(float(average_precision_score(targets, probs)))
        sufficient.append(float(np.mean(sufficient_mask(probs, sub_logits, full_probs, full_logits))))
    return {
        "probability_mae": float(np.mean(maes)),
        "logit_mae": float(np.mean(logit_maes)),
        "probability_delta": float(np.mean(deltas)),
        "retention_auprc": float(np.mean(auprcs)),
        "sufficient_fraction": float(np.mean(sufficient)),
    }


def run_seed(seed: int, cfg: dict, spec: dict, iteration_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    seed_dir = iteration_dir / f"seed_{seed}"
    model, val_loader, episode_ids, _ = prepare_seed(cfg, seed, spec, seed_dir)
    masks = subset_masks(5)
    full_mask = 2**5 - 1
    subset_rows: list[dict] = []
    shapley_rows: list[dict] = []
    ranking_rows: list[dict] = []
    max_decomposition_error = 0.0
    minimal_sizes: list[int] = []
    minimal_nonempty_sizes: list[int] = []
    bias_only_sufficient: list[bool] = []
    offset = 0
    with torch.no_grad():
        for xb, yb, cb in val_loader:
            batch_size = len(yb)
            batch_ids = episode_ids[offset : offset + batch_size]
            offset += batch_size
            xb = xb.to(DEVICE)
            cb = cb.to(DEVICE)
            out = model(xb, cb)
            full_logits = out.logit.detach().cpu().numpy()
            full_probs = out.probability.detach().cpu().numpy()
            targets = yb.numpy().astype(int)
            reconstructed = model.decision_head.bias + out.signed_decision_contributions.sum(dim=-1)
            max_decomposition_error = max(max_decomposition_error, float(torch.max(torch.abs(reconstructed - out.logit)).item()))
            frozen_logits, frozen_probs, functional_logits, functional_probs = batch_subset_outputs(model, out, masks)
            shapley_logit = shapley_from_subsets(functional_logits, 5)
            shapley_prob = shapley_from_subsets(functional_probs, 5)
            signed = out.signed_decision_contributions.detach().cpu().numpy()
            for row_idx, episode_id in enumerate(batch_ids):
                sufficient = [
                    mask_id
                    for mask_id in functional_probs
                    if bool(
                        sufficient_mask(
                            np.asarray([functional_probs[mask_id][row_idx]]),
                            np.asarray([functional_logits[mask_id][row_idx]]),
                            np.asarray([full_probs[row_idx]]),
                            np.asarray([full_logits[row_idx]]),
                        )[0]
                    )
                ]
                minimal_sizes.append(min((int(mask_id).bit_count() for mask_id in sufficient), default=5))
                minimal_nonempty_sizes.append(min((int(mask_id).bit_count() for mask_id in sufficient if mask_id != 0), default=5))
                bias_only_sufficient.append(0 in sufficient)
                for mask_id, indices, mask_np in masks:
                    for mode, logits, probs in [
                        ("frozen_evidence", frozen_logits, frozen_probs),
                        ("functional_recomputed_alpha", functional_logits, functional_probs),
                    ]:
                        subset_rows.append(
                            {
                                "seed": seed,
                                "episode_id": int(episode_id),
                                "target": int(targets[row_idx]),
                                "mode": mode,
                                "subset_mask": mask_to_string(mask_np),
                                "subset_mask_int": int(mask_id),
                                "subset_size": int(len(indices)),
                                "logit": float(logits[mask_id][row_idx]),
                                "probability": float(probs[mask_id][row_idx]),
                                "full_logit": float(full_logits[row_idx]),
                                "full_probability": float(full_probs[row_idx]),
                                "absolute_probability_error": float(abs(probs[mask_id][row_idx] - full_probs[row_idx])),
                                "absolute_logit_error": float(abs(logits[mask_id][row_idx] - full_logits[row_idx])),
                                "original_class": int(full_probs[row_idx] >= 0.5),
                            }
                        )
                for concept_idx, concept_name in enumerate(STATE_NAMES):
                    shapley_rows.append(
                        {
                            "seed": seed,
                            "episode_id": int(episode_id),
                            "concept": concept_name,
                            "concept_index": concept_idx,
                            "shapley_logit": float(shapley_logit[row_idx, concept_idx]),
                            "shapley_probability": float(shapley_prob[row_idx, concept_idx]),
                            "signed_additive_contribution": float(signed[row_idx, concept_idx]),
                            "logit_difference_shapley_minus_signed": float(shapley_logit[row_idx, concept_idx] - signed[row_idx, concept_idx]),
                        }
                    )
            orders = ranking_orders(out, shapley_logit)
            for ranking_name, order in orders.items():
                for k in range(1, 6):
                    insertion_masks = ids_from_order(order, k)
                    removal_masks = np.asarray([full_mask ^ int(mask_id) for mask_id in insertion_masks], dtype=np.int64)
                    insertion_prob = mean_values_for_episode_masks(functional_probs, insertion_masks)
                    insertion_logit = mean_values_for_episode_masks(functional_logits, insertion_masks)
                    removal_prob = mean_values_for_episode_masks(functional_probs, removal_masks)
                    removal_logit = mean_values_for_episode_masks(functional_logits, removal_masks)
                    insertion_sufficient = sufficient_mask(insertion_prob, insertion_logit, full_probs, full_logits)
                    random_insertion = exact_random_metric_baseline(functional_probs, functional_logits, k, 5, full_probs, full_logits, targets)
                    random_removal = exact_random_metric_baseline(functional_probs, functional_logits, 5 - k, 5, full_probs, full_logits, targets)
                    ranking_rows.append(
                        {
                            "seed": seed,
                            "ranking_rule": ranking_name,
                            "k": k,
                            "insertion_probability_mae": float(np.mean(np.abs(insertion_prob - full_probs))),
                            "insertion_logit_mae": float(np.mean(np.abs(insertion_logit - full_logits))),
                            "insertion_sufficient_fraction": float(np.mean(insertion_sufficient)),
                            "random_insertion_probability_mae": random_insertion["probability_mae"],
                            "random_insertion_logit_mae": random_insertion["logit_mae"],
                            "random_insertion_sufficient_fraction": random_insertion["sufficient_fraction"],
                            "insertion_retention_auprc": float(average_precision_score(targets, insertion_prob)),
                            "random_insertion_retention_auprc": random_insertion["retention_auprc"],
                            "removal_probability_delta": float(np.mean(np.abs(full_probs - removal_prob))),
                            "removal_logit_delta": float(np.mean(np.abs(full_logits - removal_logit))),
                            "random_removal_probability_delta": random_removal["probability_delta"],
                            "random_removal_logit_delta": random_removal["logit_mae"],
                            "random_removal_sufficient_fraction": random_removal["sufficient_fraction"],
                            "removal_remaining_auprc": float(average_precision_score(targets, removal_prob)),
                            "random_removal_remaining_auprc": random_removal["retention_auprc"],
                        }
                    )
    diagnosis = {
        "seed": seed,
        "max_decomposition_logit_error": max_decomposition_error,
        "median_minimal_sufficient_subset_size": float(np.median(minimal_sizes)),
        "median_minimal_nonempty_sufficient_subset_size": float(np.median(minimal_nonempty_sizes)),
        "bias_only_sufficient_fraction": float(np.mean(bias_only_sufficient)),
        "share_sufficient_with_top3_or_less": float(np.mean(np.asarray(minimal_sizes) <= 3)),
        "share_nonempty_sufficient_with_top3_or_less": float(np.mean(np.asarray(minimal_nonempty_sizes) <= 3)),
    }
    return pd.DataFrame(subset_rows), pd.DataFrame(shapley_rows), pd.DataFrame(ranking_rows), diagnosis


def aggregate_diagnosis(subsets: pd.DataFrame, shapley: pd.DataFrame, rankings: pd.DataFrame, seed_diagnoses: list[dict]) -> dict:
    ranking_summary = (
        rankings.groupby(["ranking_rule", "k"], as_index=False)
        .agg(
            insertion_probability_mae=("insertion_probability_mae", "mean"),
            insertion_logit_mae=("insertion_logit_mae", "mean"),
            insertion_sufficient_fraction=("insertion_sufficient_fraction", "mean"),
            random_insertion_probability_mae=("random_insertion_probability_mae", "mean"),
            random_insertion_logit_mae=("random_insertion_logit_mae", "mean"),
            random_insertion_sufficient_fraction=("random_insertion_sufficient_fraction", "mean"),
            removal_probability_delta=("removal_probability_delta", "mean"),
            removal_logit_delta=("removal_logit_delta", "mean"),
            random_removal_probability_delta=("random_removal_probability_delta", "mean"),
            random_removal_logit_delta=("random_removal_logit_delta", "mean"),
            insertion_retention_auprc=("insertion_retention_auprc", "mean"),
            random_insertion_retention_auprc=("random_insertion_retention_auprc", "mean"),
        )
    )
    best_top3 = ranking_summary[(ranking_summary["ranking_rule"] == "absolute_shapley_logit") & (ranking_summary["k"] == 3)]
    top1 = ranking_summary[(ranking_summary["ranking_rule"] == "absolute_shapley_logit") & (ranking_summary["k"] == 1)]
    shapley_diff = shapley["logit_difference_shapley_minus_signed"].abs()
    return {
        "diagnosis_type": "fan_exact_subset_shapley_train_validation_only",
        "test_opened": False,
        "max_decomposition_logit_error": float(max(d["max_decomposition_logit_error"] for d in seed_diagnoses)),
        "median_minimal_sufficient_subset_size": float(np.median([d["median_minimal_sufficient_subset_size"] for d in seed_diagnoses])),
        "median_minimal_nonempty_sufficient_subset_size": float(np.median([d["median_minimal_nonempty_sufficient_subset_size"] for d in seed_diagnoses])),
        "mean_bias_only_sufficient_fraction": float(np.mean([d["bias_only_sufficient_fraction"] for d in seed_diagnoses])),
        "mean_share_sufficient_with_top3_or_less": float(np.mean([d["share_sufficient_with_top3_or_less"] for d in seed_diagnoses])),
        "mean_share_nonempty_sufficient_with_top3_or_less": float(np.mean([d["share_nonempty_sufficient_with_top3_or_less"] for d in seed_diagnoses])),
        "mean_abs_shapley_signed_logit_difference": float(shapley_diff.mean()),
        "p95_abs_shapley_signed_logit_difference": float(shapley_diff.quantile(0.95)),
        "absolute_shapley_top1_insertion_mae": None if top1.empty else float(top1["insertion_probability_mae"].iloc[0]),
        "absolute_shapley_top3_insertion_mae": None if best_top3.empty else float(best_top3["insertion_probability_mae"].iloc[0]),
        "absolute_shapley_top3_sufficient_fraction": None if best_top3.empty else float(best_top3["insertion_sufficient_fraction"].iloc[0]),
        "top1_is_not_blocking_gate": bool((not best_top3.empty) and (float(best_top3["insertion_probability_mae"].iloc[0]) <= 0.05)),
        "seed_diagnoses": seed_diagnoses,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteration-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    iteration_dir = Path(args.iteration_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    iter_df = pd.read_csv(iteration_dir / "oracle_fan_results.csv")
    subsets, shapley, rankings, diagnoses = [], [], [], []
    for seed in args.seeds:
        row = iter_df[iter_df["seed"] == seed].iloc[0].to_dict()
        sub, shp, rnk, diag = run_seed(seed, cfg, row, iteration_dir)
        subsets.append(sub)
        shapley.append(shp)
        rankings.append(rnk)
        diagnoses.append(diag)
    subset_df = pd.concat(subsets, ignore_index=True)
    shapley_df = pd.concat(shapley, ignore_index=True)
    ranking_df = pd.concat(rankings, ignore_index=True)
    diagnosis = aggregate_diagnosis(subset_df, shapley_df, ranking_df, diagnoses)
    subset_df.to_parquet(output / "exact_subset_faithfulness.parquet", index=False)
    shapley_df.to_parquet(output / "shapley_contributions.parquet", index=False)
    ranking_df.to_csv(output / "ranking_comparison.csv", index=False)
    (output / "faithfulness_diagnosis.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    print(json.dumps(diagnosis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
