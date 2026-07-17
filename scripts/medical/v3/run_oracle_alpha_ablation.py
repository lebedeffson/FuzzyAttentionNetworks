#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.medical.v2.run_v2_1_program import make_sequence_loaders
from scripts.medical.v2.run_v2_program import make_episodes, split_frame, subset_cols
from scripts.medical.v3.diagnose_fan_faithfulness import run_seed as run_exact_faithfulness_seed
from scripts.medical.v3.run_fan_iteration import (
    ConceptScaler,
    binary_metrics,
    build_loaders,
    concept_ceiling,
    evaluate_model,
    initialize_memberships,
    make_model,
    set_all_seeds,
    train_oracle,
)


ALPHA_MODES = ["no_alpha", "uniform_alpha", "softmax_alpha", "residual_floor_alpha"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_seed(data_seed: int, restart: int) -> int:
    return data_seed * 100 + restart


def cfg_for_alpha(cfg: dict, alpha_mode: str) -> dict:
    out = dict(cfg)
    out["fan"] = dict(cfg.get("fan", {}))
    out["fan"]["alpha_mode"] = alpha_mode
    out["fan"]["gamma_init"] = float(out["fan"].get("gamma_init", 0.5))
    out["fan"]["gamma_max"] = float(out["fan"].get("gamma_max", 0.9))
    return out


def effective_concept_count(alpha: np.ndarray, alpha_mode: str) -> float:
    if alpha_mode == "no_alpha":
        return float(alpha.shape[1])
    denom = np.maximum(alpha.sum(axis=1, keepdims=True), 1e-12)
    p = alpha / denom
    entropy = -(p * np.log(p + 1e-12)).sum(axis=1)
    return float(np.exp(entropy).mean())


def pairwise_spearman(vectors: list[np.ndarray]) -> float:
    vals = []
    for left, right in combinations(vectors, 2):
        if np.allclose(left, right):
            vals.append(1.0)
        elif np.std(left) == 0 or np.std(right) == 0:
            vals.append(0.0)
        else:
            vals.append(float(stats.spearmanr(left, right).statistic))
    return float(np.nanmean(vals)) if vals else np.nan


def vector_from_row(row: dict, key: str) -> np.ndarray:
    value = row[key]
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return np.asarray(value, dtype=float)


def pairwise_kendall(vectors: list[np.ndarray]) -> float:
    vals = []
    for left, right in combinations(vectors, 2):
        if np.allclose(left, right):
            vals.append(1.0)
        elif np.std(left) == 0 or np.std(right) == 0:
            vals.append(0.0)
        else:
            vals.append(float(stats.kendalltau(left, right).statistic))
    return float(np.nanmean(vals)) if vals else np.nan


def topk_jaccard(vectors: list[np.ndarray], k: int = 3) -> float:
    vals = []
    for left, right in combinations(vectors, 2):
        lset = set(np.argsort(-np.abs(left))[:k].tolist())
        rset = set(np.argsort(-np.abs(right))[:k].tolist())
        vals.append(len(lset & rset) / max(1, len(lset | rset)))
    return float(np.mean(vals)) if vals else np.nan


def train_selected_seed(seed: int, cfg: dict, alpha_mode: str, variant_dir: Path) -> dict:
    set_all_seeds(seed)
    clean = make_episodes(seed, cfg, "clean")
    split = split_frame(clean, seed)
    _, _, arrays = make_sequence_loaders(split, subset_cols("full_input"), int(cfg["training"]["batch_size"]), 5)
    scaler = ConceptScaler.fit(arrays["c_train_seq"][:, :, :5], "minmax_train")
    train_loader, val_loader, scaled = build_loaders(arrays, scaler, 5, int(cfg["training"]["batch_size"]))
    ceiling, ceiling_df = concept_ceiling(arrays, scaler, 5)
    best = None
    restart_rows = []
    best_state = None
    best_history = None
    model_cfg = cfg_for_alpha(cfg, alpha_mode)
    for restart in [1, 2, 3]:
        mseed = model_seed(seed, restart)
        set_all_seeds(mseed)
        model = make_model(model_cfg, 5, "gaussian", 3, 1.0, float(arrays["y_train"].mean()))
        initialize_memberships(model, scaled["c_train"])
        history = train_oracle(model, model_cfg, train_loader, val_loader, mseed)
        y, p, extras = evaluate_model(model, val_loader)
        auprc = float(average_precision_score(y, p))
        row = {
            "seed": seed,
            "model_seed": mseed,
            "restart": restart,
            "alpha_mode": alpha_mode,
            "membership_family": "gaussian",
            "n_memberships": 3,
            "concept_scaling": "minmax_train",
            "temperature_init": 1.0,
            **binary_metrics(y, p),
            "oracle_ceiling_AUPRC": float(ceiling),
            "ceiling_ratio": float(auprc / max(float(ceiling), 1e-8)),
            "effective_concept_count": effective_concept_count(extras["alpha"], alpha_mode),
            "mean_alpha": extras["alpha"].mean(axis=0).tolist(),
            "decision_weight": model.decision_head.weight.detach().cpu().numpy().tolist(),
            "mean_signed_contribution": extras["signed"].mean(axis=0).tolist(),
            "mean_fuzzy_value": extras["fuzzy_values"].mean(axis=0).tolist(),
            "mean_concept_evidence": extras["evidence"].mean(axis=0).tolist(),
        }
        restart_rows.append(row)
        if best is None or auprc > best["AUPRC"]:
            best = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_history = pd.DataFrame(history)
            best["best_validation_probability"] = p.tolist()
    seed_dir = variant_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(restart_rows).to_csv(seed_dir / "restart_results.csv", index=False)
    ceiling_df.assign(seed=seed, alpha_mode=alpha_mode).to_csv(seed_dir / "concept_ceiling.csv", index=False)
    best_history.to_csv(seed_dir / "training_log_selected.csv", index=False)
    (seed_dir / "concept_scaler.json").write_text(json.dumps(scaler.to_json(), indent=2), encoding="utf-8")
    torch.save(best_state, seed_dir / "oracle_fan_checkpoint.pt")
    selected = {k: v for k, v in best.items() if k != "best_validation_probability"}
    (seed_dir / "selected_model.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return selected


def summarize_exact_outputs(variant_dir: Path, selected_rows: list[dict], cfg: dict) -> dict:
    iter_df = pd.DataFrame(selected_rows)
    iter_df.to_csv(variant_dir / "oracle_fan_results.csv", index=False)
    subsets, shapley, rankings, diagnoses = [], [], [], []
    for row in selected_rows:
        sub, shp, rnk, diag = run_exact_faithfulness_seed(int(row["seed"]), cfg_for_alpha(cfg, row["alpha_mode"]), row, variant_dir)
        subsets.append(sub)
        shapley.append(shp)
        rankings.append(rnk)
        diagnoses.append(diag)
    subset_df = pd.concat(subsets, ignore_index=True)
    shapley_df = pd.concat(shapley, ignore_index=True)
    ranking_df = pd.concat(rankings, ignore_index=True)
    subset_df.to_parquet(variant_dir / "exact_subset_faithfulness.parquet", index=False)
    shapley_df.to_parquet(variant_dir / "shapley_contributions.parquet", index=False)
    ranking_df.to_csv(variant_dir / "ranking_comparison.csv", index=False)
    rank_mean = (
        ranking_df.groupby(["ranking_rule", "k"], as_index=False)
        .agg(
            insertion_probability_mae=("insertion_probability_mae", "mean"),
            insertion_logit_mae=("insertion_logit_mae", "mean"),
            insertion_sufficient_fraction=("insertion_sufficient_fraction", "mean"),
            random_insertion_probability_mae=("random_insertion_probability_mae", "mean"),
            removal_probability_delta=("removal_probability_delta", "mean"),
            removal_logit_delta=("removal_logit_delta", "mean"),
            random_removal_probability_delta=("random_removal_probability_delta", "mean"),
            insertion_retention_auprc=("insertion_retention_auprc", "mean"),
            removal_remaining_auprc=("removal_remaining_auprc", "mean"),
        )
    )
    shapley_abs_gap = shapley_df["logit_difference_shapley_minus_signed"].abs()
    for row in selected_rows:
        seed = int(row["seed"])
        if "decision_weight" not in row:
            checkpoint = torch.load(variant_dir / f"seed_{seed}" / "oracle_fan_checkpoint.pt", map_location="cpu")
            weight = checkpoint.get("decision_head.raw_weight", checkpoint.get("decision_head.weight"))
            row["decision_weight"] = weight.detach().cpu().numpy().tolist()
        if "mean_signed_contribution" not in row:
            signed = (
                shapley_df[shapley_df["seed"] == seed]
                .groupby("concept_index")["signed_additive_contribution"]
                .mean()
                .sort_index()
                .to_numpy()
            )
            row["mean_signed_contribution"] = signed.tolist()
        if "mean_fuzzy_value" not in row:
            row["mean_fuzzy_value"] = row["mean_signed_contribution"]
        if "mean_concept_evidence" not in row:
            row["mean_concept_evidence"] = row["mean_signed_contribution"]
    mean_alpha = [vector_from_row(row, "mean_alpha") for row in selected_rows]
    decision_weight = [vector_from_row(row, "decision_weight") for row in selected_rows]
    mean_signed = [vector_from_row(row, "mean_signed_contribution") for row in selected_rows]
    mean_fuzzy = [vector_from_row(row, "mean_fuzzy_value") for row in selected_rows]
    mean_evidence = [vector_from_row(row, "mean_concept_evidence") for row in selected_rows]
    if selected_rows[0]["alpha_mode"] == "no_alpha":
        alpha_stability = "NOT_APPLICABLE_CONSTANT_ALPHA"
        alpha_spearman = np.nan
    else:
        alpha_stability = "MEASURED"
        alpha_spearman = pairwise_spearman(mean_alpha)
    summary = {
        "alpha_mode": selected_rows[0]["alpha_mode"],
        "mean_AUPRC": float(iter_df["AUPRC"].mean()),
        "mean_ceiling_ratio": float(iter_df["ceiling_ratio"].mean()),
        "min_ceiling_ratio": float(iter_df["ceiling_ratio"].min()),
        "mean_effective_concept_count": float(iter_df["effective_concept_count"].mean()),
        "max_decomposition_logit_error": float(max(d["max_decomposition_logit_error"] for d in diagnoses)),
        "median_minimal_sufficient_subset_size": float(np.median([d["median_minimal_sufficient_subset_size"] for d in diagnoses])),
        "share_sufficient_with_top3_or_less": float(np.mean([d["share_sufficient_with_top3_or_less"] for d in diagnoses])),
        "mean_abs_shapley_signed_logit_difference": float(shapley_abs_gap.mean()),
        "p95_abs_shapley_signed_logit_difference": float(shapley_abs_gap.quantile(0.95)),
        "alpha_stability_status": alpha_stability,
        "cross_seed_mean_alpha_spearman": alpha_spearman,
        "cross_seed_decision_weight_spearman": pairwise_spearman(decision_weight),
        "cross_seed_decision_weight_kendall": pairwise_kendall(decision_weight),
        "cross_seed_signed_contribution_spearman": pairwise_spearman(mean_signed),
        "cross_seed_signed_contribution_kendall": pairwise_kendall(mean_signed),
        "cross_seed_top3_contribution_jaccard": topk_jaccard(mean_signed, 3),
        "cross_seed_fuzzy_value_spearman": pairwise_spearman(mean_fuzzy),
        "cross_seed_evidence_spearman": pairwise_spearman(mean_evidence),
    }
    for ranking_rule in [
        "alpha",
        "absolute_signed_contribution",
        "absolute_shapley_logit",
        "predicted_class_shapley",
        "positive_class_shapley",
    ]:
        rows = rank_mean[rank_mean["ranking_rule"] == ranking_rule]
        if rows.empty:
            continue
        summary[f"{ranking_rule}_insertion_mae_curve_mean"] = float(rows["insertion_probability_mae"].mean())
        summary[f"{ranking_rule}_insertion_logit_mae_curve_mean"] = float(rows["insertion_logit_mae"].mean())
        summary[f"{ranking_rule}_insertion_sufficient_curve_mean"] = float(rows["insertion_sufficient_fraction"].mean())
        summary[f"{ranking_rule}_removal_delta_curve_mean"] = float(rows["removal_probability_delta"].mean())
        summary[f"{ranking_rule}_removal_logit_delta_curve_mean"] = float(rows["removal_logit_delta"].mean())
        for k in [1, 2, 3]:
            krow = rows[rows["k"] == k]
            if not krow.empty:
                summary[f"{ranking_rule}_top{k}_insertion_mae"] = float(krow["insertion_probability_mae"].iloc[0])
                summary[f"{ranking_rule}_top{k}_insertion_logit_mae"] = float(krow["insertion_logit_mae"].iloc[0])
                summary[f"{ranking_rule}_top{k}_insertion_sufficient_fraction"] = float(krow["insertion_sufficient_fraction"].iloc[0])
                summary[f"{ranking_rule}_top{k}_removal_delta"] = float(krow["removal_probability_delta"].iloc[0])
    (variant_dir / "faithfulness_diagnosis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_alpha_ablation(cfg: dict, seeds: list[int], output: Path, alpha_modes: list[str]) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for alpha_mode in alpha_modes:
        if alpha_mode not in ALPHA_MODES:
            raise ValueError(f"Unknown alpha mode {alpha_mode!r}")
        variant_dir = output / alpha_mode
        variant_dir.mkdir(parents=True, exist_ok=True)
        selected_rows = [train_selected_seed(seed, cfg, alpha_mode, variant_dir) for seed in seeds]
        summary = summarize_exact_outputs(variant_dir, selected_rows, cfg)
        summaries.append(summary)
    existing = []
    for alpha_mode in ALPHA_MODES:
        path = output / alpha_mode / "faithfulness_diagnosis.json"
        if path.exists() and alpha_mode not in alpha_modes:
            existing.append(json.loads(path.read_text(encoding="utf-8")))
    summary_df = pd.DataFrame(existing + summaries)
    summary_df.to_csv(output / "alpha_ablation_summary.csv", index=False)
    best = summary_df.sort_values(
        ["mean_AUPRC", "absolute_shapley_logit_top3_insertion_mae", "mean_abs_shapley_signed_logit_difference"],
        ascending=[False, True, True],
    ).iloc[0].to_dict()
    result = {
        "stage": "oracle_alpha_ablation",
        "test_opened": False,
        "seeds": seeds,
        "alpha_modes_run_this_invocation": alpha_modes,
        "alpha_modes_in_summary": summary_df["alpha_mode"].tolist(),
        "deterministic_restarts_per_seed": 3,
        "fixed_membership": "gaussian",
        "fixed_n_memberships": 3,
        "fixed_concept_scaling": "minmax_train",
        "best_alpha_mode": best["alpha_mode"],
        "best_mean_AUPRC": float(best["mean_AUPRC"]),
        "best_mean_ceiling_ratio": float(best["mean_ceiling_ratio"]),
        "summaries": summary_df.to_dict(orient="records"),
    }
    (output / "alpha_ablation_diagnosis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    no_alpha_results = output / "no_alpha" / "oracle_fan_results.csv"
    if no_alpha_results.exists():
        manifest = {
            "stage": "oracle_alpha_ablation_manifest",
            "dataset": "med_circuitbench_clean",
            "split": "train_validation",
            "seeds": seeds,
            "concept_definition": "temporal_I_R_V_O_S_hours_0_35",
            "concept_scaling": "minmax_train",
            "membership_family": "gaussian",
            "n_memberships": 3,
            "architecture": "Multi-Set Additive Temporal Concept FAN-NoAlpha",
            "alpha_mode": "no_alpha",
            "results_file": str(no_alpha_results.relative_to(output)),
            "results_sha256": sha256_file(no_alpha_results),
        }
        (output / "no_alpha" / "oracle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha-modes", nargs="+", default=ALPHA_MODES)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run_alpha_ablation(cfg, args.seeds, Path(args.output), args.alpha_modes)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
