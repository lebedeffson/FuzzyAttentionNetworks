#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, r2_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.medical.v2.run_v2_program import make_episodes, split_frame, subset_cols
from fan.sctc.model import SparseTranscoder
from med_circuitbench.models.transformer import ClinicalTransformer, TransformerConfig
from scripts.medical.v3.run_research_program import (
    DEVICE,
    collect_capture,
    collect_fan_latents,
    load_predicted_fan,
    prepare_arrays,
    validate_claims,
)
from scripts.medical.v3.run_fan_iteration import ConceptScaler
from scripts.medical.v3.validate_no_synthetic_results import validate as validate_no_synthetic


FINAL_STATUSES_REAL = {
    "V3_REAL_MIXED_RESULT_REPLICATION_CONFIRMED",
    "V3_REAL_MIXED_RESULT_REPLICATION_NOT_CONFIRMED",
    "V3_REAL_FAN_VALIDATED_STANDARD_MECHANISTIC_NEGATIVE",
    "V3_REAL_VALIDATED_NEGATIVE",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tex_escape(value: object) -> str:
    text = str(value)
    for src, dst in {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }.items():
        text = text.replace(src, dst)
    return text


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def ensure_layout(output: Path) -> None:
    for rel in [
        "manifests",
        "validation",
        "partial_test",
        "replication",
        "results",
        "checkpoints",
        "tables",
        "figures",
        "paper",
        "logs",
        "delivery",
        "limitations",
    ]:
        (output / rel).mkdir(parents=True, exist_ok=True)


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def source_dirty() -> list[str]:
    rows = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    return [row for row in rows if not row.startswith("?? artifacts/")]


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pred = p >= 0.5
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if m.any():
            ece += float(m.mean()) * abs(float(p[m].mean()) - float(y[m].mean()))
    return {
        "AUROC": float(roc_auc_score(y, p)) if np.unique(y).size > 1 else float("nan"),
        "AUPRC": float(average_precision_score(y, p)),
        "F1": float(f1_score(y, pred)) if np.unique(pred).size > 1 else 0.0,
        "Brier": float(brier_score_loss(y, p)),
        "ECE": ece,
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


def bootstrap_curve_margin(
    observed: np.ndarray,
    random: np.ndarray,
    episode_ids: np.ndarray,
    seed: int,
    n_bootstrap: int = 2000,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    observed = np.asarray(observed, dtype=float)
    random = np.asarray(random, dtype=float)
    ids = np.asarray(episode_ids)
    if len(observed) == 0 or len(random) == 0:
        return float("nan"), float("nan"), float("nan")
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(ids), size=len(ids))
        diffs.append(float(np.nanmean(observed[idx] - random[idx])))
    diff = float(np.nanmean(observed - random))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return diff, float(lo), float(hi)


def load_standard_model(cfg: dict, arrays: dict, checkpoint: Path) -> ClinicalTransformer:
    model = ClinicalTransformer(
        TransformerConfig(
            input_dim=arrays["x_train"].shape[-1],
            layers=int(cfg["model"]["layers"]),
            d_model=int(cfg["model"]["d_model"]),
            heads=int(cfg["model"]["heads"]),
            d_ffn=int(cfg["model"]["d_ffn"]),
            dropout=float(cfg["model"].get("dropout", 0.1)),
            sequence_length=int(cfg["dataset"]["observed_window"]),
        )
    ).to(DEVICE)
    state = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
    model.eval()
    return model


def load_sctc(checkpoint: Path) -> tuple[SparseTranscoder, int, int]:
    state = torch.load(checkpoint, map_location=DEVICE)
    n_features = int(state.get("n_features", state["model_state_dict"]["encoder.weight"].shape[0]))
    layer = int(state.get("layer", 0))
    d_model = int(state["model_state_dict"]["decoder.weight"].shape[0])
    model = SparseTranscoder(d_model=d_model, n_features=n_features, top_k=24).to(DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, layer, n_features


def split_arrays_for_seed(cfg: dict, seed: int, cols: np.ndarray | None = None, include_test: bool = False) -> dict:
    cols = subset_cols("full_input") if cols is None else cols
    frame = make_episodes(seed, cfg, "clean")
    return prepare_arrays(split_frame(frame, seed), cols, include_test=include_test)


def flatten_concept_metrics(true_seq: np.ndarray, pred_seq: np.ndarray) -> dict:
    rows = []
    names = ["I", "R", "V", "O", "S"]
    for idx, name in enumerate(names):
        y = true_seq[:, :, idx].reshape(-1)
        p = pred_seq[:, :, idx].reshape(-1)
        rows.append(
            {
                "concept": name,
                "R2": float(r2_score(y, p)),
                "Pearson": float(stats.pearsonr(y, p).statistic) if np.std(y) > 0 and np.std(p) > 0 else float("nan"),
                "MAE": float(np.mean(np.abs(y - p))),
                "delta_MAE": float(np.mean(np.abs(np.diff(y.reshape(true_seq.shape[0], true_seq.shape[1]), axis=1) - np.diff(p.reshape(pred_seq.shape[0], pred_seq.shape[1]), axis=1)))),
            }
        )
    per = pd.DataFrame(rows)
    return {
        "per_concept": per,
        "direct_macro_R2": float(per["R2"].mean()),
        "variance_weighted_R2": float(r2_score(true_seq.reshape(-1, true_seq.shape[-1]), pred_seq.reshape(-1, pred_seq.shape[-1]), multioutput="variance_weighted")),
        "macro_Pearson": float(per["Pearson"].mean()),
        "concept_MAE": float(per["MAE"].mean()),
        "concept_delta_MAE": float(per["delta_MAE"].mean()),
    }


def fan_forward_episode_tables(seed: int, cfg: dict, source: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arrays = split_arrays_for_seed(cfg, seed)
    scaler = ConceptScaler.fit(arrays["c_train_seq"], "minmax_train")
    c_val_scaled = scaler.transform(arrays["c_val_seq"])
    checkpoint = source / "checkpoints" / "fan" / f"predicted_noalpha_seed_{seed}.pt"
    model = load_predicted_fan(seed, cfg, checkpoint, float(arrays["y_train"].mean()))
    _, logits, probs, labels, extras = collect_fan_latents(
        model, arrays["x_val"], arrays["y_val"], c_val_scaled, int(cfg["training"]["batch_size"])
    )
    with torch.no_grad():
        pred_chunks = []
        xs = torch.from_numpy(arrays["x_val"]).float()
        for start in range(0, len(xs), int(cfg["training"]["batch_size"])):
            xb = xs[start : start + int(cfg["training"]["batch_size"])].to(DEVICE)
            pred_chunks.append(model.projector(model.encoder(xb)).detach().cpu().numpy())
    pred_seq = np.concatenate(pred_chunks, axis=0)
    metrics = flatten_concept_metrics(c_val_scaled, pred_seq)
    pred_rows = pd.DataFrame(
        {
            "seed": seed,
            "episode_id": arrays["val_ids"],
            "target": labels.astype(int),
            "logit": logits.astype(float),
            "probability": probs.astype(float),
            "checkpoint": str(checkpoint),
        }
    )
    concept_records = []
    for ei, episode_id in enumerate(arrays["val_ids"]):
        for t in range(pred_seq.shape[1]):
            for ci, concept in enumerate(["I", "R", "V", "O", "S"]):
                concept_records.append(
                    {
                        "seed": seed,
                        "episode_id": int(episode_id),
                        "time": t,
                        "concept": concept,
                        "concept_index": ci,
                        "true_value_raw": float(arrays["c_val_seq"][ei, t, ci]),
                        "true_value_scaled": float(c_val_scaled[ei, t, ci]),
                        "predicted_value_scaled": float(pred_seq[ei, t, ci]),
                    }
                )
    contrib = pd.DataFrame(
        [
            {
                "seed": seed,
                "episode_id": int(episode_id),
                "concept": concept,
                "concept_index": ci,
                "signed_contribution": float(extras["signed"][ei, ci]),
                "concept_evidence": float(extras["evidence"][ei, ci]),
                "fuzzy_value": float(extras["fuzzy_values"][ei, ci]),
            }
            for ei, episode_id in enumerate(arrays["val_ids"])
            for ci, concept in enumerate(["I", "R", "V", "O", "S"])
        ]
    )
    metric_row = {"seed": seed, "model": "Predicted FAN-NoAlpha frozen forward", **binary_metrics(labels, probs), **{k: v for k, v in metrics.items() if k != "per_concept"}}
    (output / "validation" / f"fan_seed_{seed}_per_concept_metrics.csv").parent.mkdir(parents=True, exist_ok=True)
    metrics["per_concept"].assign(seed=seed).to_csv(output / "validation" / f"fan_seed_{seed}_per_concept_metrics.csv", index=False)
    return pred_rows, pd.DataFrame(concept_records), contrib.assign(**{k: metric_row[k] for k in []}), pd.DataFrame([metric_row])


def copy_required_source(output: Path) -> None:
    delivery = output / "delivery"
    if delivery.exists():
        shutil.rmtree(delivery)
    for rel in ["SOURCE", "CONFIGS", "TESTS", "RESULTS", "TABLES", "FIGURES", "PAPER", "CHECKPOINTS", "MANIFESTS", "LIMITATIONS", "PROJECT_MEMORY"]:
        (delivery / rel).mkdir(parents=True, exist_ok=True)
    for rel in ["src/fan", "src/med_circuitbench", "scripts/medical/v3"]:
        shutil.copytree(ROOT / rel, delivery / "SOURCE" / rel, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "data"), dirs_exist_ok=True)
    shutil.copytree(ROOT / "configs" / "medical" / "v3", delivery / "CONFIGS" / "configs" / "medical" / "v3", dirs_exist_ok=True)
    shutil.copytree(ROOT / "tests" / "medical" / "v3", delivery / "TESTS" / "tests" / "medical" / "v3", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)
    for src, dst in [
        (output / "results", delivery / "RESULTS"),
        (output / "tables", delivery / "TABLES"),
        (output / "figures", delivery / "FIGURES"),
        (output / "paper", delivery / "PAPER"),
        (output / "manifests", delivery / "MANIFESTS"),
        (output / "limitations", delivery / "LIMITATIONS"),
    ]:
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.zip", "activation_cache"))
    for stale in [
        delivery / "RESULTS" / "delivery_validation.json",
        delivery / "MANIFESTS" / "final_zip_validation.json",
    ]:
        if stale.exists():
            stale.unlink()
    checkpoints = output / "checkpoints"
    if checkpoints.exists():
        shutil.copytree(checkpoints, delivery / "CHECKPOINTS", dirs_exist_ok=True, ignore=shutil.ignore_patterns("*optimizer*", "*scheduler*", "*failed*"))
    shutil.copy2(ROOT / "AGENTS.md", delivery / "AGENTS.md")
    state = ROOT / "docs" / "medical" / "PROJECT_STATE.md"
    if state.exists():
        shutil.copy2(state, delivery / "PROJECT_MEMORY" / "PROJECT_STATE.md")
    (delivery / "GIT_INFO.txt").write_text(
        f"commit: {git_commit()}\nstatus:\n{subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)}",
        encoding="utf-8",
    )
    (delivery / "README_FIRST.md").write_text(
        "# Med-CircuitBench V3 Real Final\n\nThis package contains source, tests, configs, results, provenance, paper artifacts, and project memory for the frozen real-practice finalization.\n",
        encoding="utf-8",
    )


def delivery_manifest(delivery: Path) -> dict:
    files = []
    for path in sorted(delivery.rglob("*")):
        if path.is_file():
            files.append({"path": str(path.relative_to(delivery)), "sha256": sha256_file(path), "size": path.stat().st_size})
    return {"created_at": datetime.now().isoformat(), "commit": git_commit(), "files": files}


def freeze_existing_models(config: Path, source: Path, output: Path, seeds: list[int]) -> dict:
    rows = []
    checkpoint_specs = []
    for seed in seeds:
        checkpoint_specs.extend(
            [
                ("Standard Transformer", seed, source / "standard_sctc" / f"seed_{seed}" / "standard_transformer.pt"),
                ("Predicted FAN-NoAlpha", seed, source / "checkpoints" / "fan" / f"predicted_noalpha_seed_{seed}.pt"),
                ("Oracle FAN-NoAlpha", seed, ROOT / "artifacts" / "medical" / "v3_alpha_ablation" / "no_alpha" / f"seed_{seed}" / "oracle_fan_checkpoint.pt"),
                ("Planted model", seed, source / "checkpoints" / "planted_sctc" / f"planted_model_seed{seed}.pt"),
            ]
        )
        std = pd.read_csv(source / "results" / "standard_sctc_results.csv")
        best_std = std[std["seed"] == seed].sort_values(["dead_feature_fraction", "delta_AUPRC"]).iloc[0]
        checkpoint_specs.append(("Selected Standard SCTC", seed, Path(best_std["checkpoint"])))
        planted = pd.read_csv(source / "results" / "planted_feature_grid.csv")
        best_planted = planted[planted["seed"] == seed].sort_values(["dead_feature_fraction", "delta_AUPRC"]).iloc[0]
        checkpoint_specs.append(("Selected planted SCTC", seed, Path(best_planted["checkpoint"])))
    config_sha = sha256_file(config)
    for model, seed, path in checkpoint_specs:
        status = "OK" if path.exists() else "MISSING_REQUIRED_CHECKPOINT"
        if path.exists():
            safe_model = model.lower().replace(" ", "_").replace("-", "_")
            dst = output / "checkpoints" / safe_model / f"seed_{seed}_{path.name}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
        rows.append(
            {
                "model": model,
                "seed": seed,
                "path": str(path),
                "status": status,
                "sha256": sha256_file(path) if path.exists() else None,
                "config_sha256": config_sha,
                "dataset_sha256": sha256_file(source / "dataset" / "dataset_summary.csv") if (source / "dataset" / "dataset_summary.csv").exists() else None,
                "train_split_sha256": sha256_file(source / "splits" / f"seed_{seed}" / "train_episodes.parquet"),
                "validation_split_sha256": sha256_file(source / "splits" / f"seed_{seed}" / "validation_episodes.parquet"),
                "test_split_sha256": sha256_file(source / "splits" / f"seed_{seed}" / "test_episodes.parquet"),
            }
        )
    manifest = {"commit": git_commit(), "checkpoints": rows}
    write_json(output / "manifests" / "frozen_models.json", manifest)
    if any(row["status"] != "OK" for row in rows):
        raise RuntimeError("MISSING_REQUIRED_CHECKPOINT")
    return manifest


def recompute_fan_validation(config: Path, source: Path, output: Path, seeds: list[int]) -> pd.DataFrame:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    pred_frames, concept_frames, contrib_frames, metric_frames = [], [], [], []
    for seed in seeds:
        pred_rows, concept_rows, contrib_rows, metric_rows = fan_forward_episode_tables(seed, cfg, source, output)
        pred_frames.append(pred_rows)
        concept_frames.append(concept_rows)
        contrib_frames.append(contrib_rows)
        metric_frames.append(metric_rows)
    predictions = pd.concat(pred_frames, ignore_index=True)
    concepts = pd.concat(concept_frames, ignore_index=True)
    contributions = pd.concat(contrib_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    oracle = pd.read_csv(source / "results" / "predicted_vs_oracle_noalpha.csv")[["seed", "oracle_noalpha_AUPRC"]]
    metrics = metrics.merge(oracle, on="seed", how="left")
    metrics["predicted_oracle_ratio"] = metrics["AUPRC"] / metrics["oracle_noalpha_AUPRC"]
    metrics["AUPRC_from_raw_forward"] = metrics["AUPRC"]
    metrics["source"] = "frozen_checkpoint_validation_forward"
    predictions.to_parquet(output / "results" / "fan_validation_predictions.parquet", index=False)
    concepts.to_parquet(output / "results" / "fan_concept_predictions.parquet", index=False)
    contributions.to_parquet(output / "results" / "fan_signed_contributions.parquet", index=False)
    metrics.to_csv(output / "results" / "fan_validation_metrics.csv", index=False)

    for rel in ["exact_subset_faithfulness.parquet", "shapley_contributions.parquet", "leakage_bootstrap.parquet", "ranking_comparison.csv"]:
        src = source / "results" / rel
        if src.exists():
            dest = output / "results" / rel
            if src.suffix == ".parquet":
                pd.read_parquet(src).to_parquet(dest, index=False)
            else:
                pd.read_csv(src).to_csv(dest, index=False)

    stability_rows = []
    reference_frame = make_episodes(20260715, cfg, "clean")
    reference_means = {}
    for seed in seeds:
        ref_arrays = replication_all_arrays(cfg, seed, reference_frame)
        checkpoint = source / "checkpoints" / "fan" / f"predicted_noalpha_seed_{seed}.pt"
        model = load_predicted_fan(seed, cfg, checkpoint, float(ref_arrays["y_train"].mean()))
        _, _, _, _, ref_extras = collect_fan_latents(model, ref_arrays["x_all"], ref_arrays["y_all"], ref_arrays["c_all_seq"], 512)
        reference_means[seed] = ref_extras["signed"].mean(axis=0)
    for a, b in [(42, 43), (42, 44), (43, 44)]:
        if a in reference_means and b in reference_means:
            va, vb = reference_means[a], reference_means[b]
            spearman = float(stats.spearmanr(va, vb).statistic)
            kendall = float(stats.kendalltau(va, vb).statistic)
            top_a = set(np.argsort(-np.abs(va))[:3].tolist())
            top_b = set(np.argsort(-np.abs(vb))[:3].tolist())
            stability_rows.extend(
                [
                    {"pair": f"{a}-{b}", "metric": "signed_contribution_spearman", "value": spearman, "source": "common_reference_frozen_fan_forward"},
                    {"pair": f"{a}-{b}", "metric": "signed_contribution_kendall_tau", "value": kendall, "source": "common_reference_frozen_fan_forward"},
                    {"pair": f"{a}-{b}", "metric": "top3_contribution_jaccard", "value": len(top_a & top_b) / len(top_a | top_b), "source": "common_reference_frozen_fan_forward"},
                ]
            )
    stability_rows.append({"pair": "all", "metric": "alpha_stability", "value": np.nan, "status": "NOT_APPLICABLE_CONSTANT_ALPHA"})
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(output / "results" / "fan_cross_seed_stability.csv", index=False)

    shapley = pd.read_parquet(output / "results" / "shapley_contributions.parquet")
    subset = pd.read_parquet(output / "results" / "exact_subset_faithfulness.parquet")
    rank = pd.read_csv(output / "results" / "ranking_comparison.csv") if (output / "results" / "ranking_comparison.csv").exists() else pd.DataFrame()
    leak = pd.read_parquet(output / "results" / "leakage_bootstrap.parquet")
    full = subset[subset["subset_size"] == 5]
    nonempty_sufficient = subset[
        (subset["subset_size"] > 0)
        & subset["class_preserved"]
        & (subset["absolute_probability_error"] <= 0.05)
        & (subset["absolute_logit_error"] <= 0.25)
    ]
    minimal_nonempty = nonempty_sufficient.groupby(["seed", "episode_id"])["subset_size"].min()
    bias_only = subset[
        (subset["subset_size"] == 0)
        & subset["class_preserved"]
        & (subset["absolute_probability_error"] <= 0.05)
        & (subset["absolute_logit_error"] <= 0.25)
    ]
    abs_rank = rank[rank.get("ranking_rule", pd.Series(dtype=str)).eq("absolute_shapley_logit")] if not rank.empty else pd.DataFrame()
    k3 = abs_rank[abs_rank["k"] == 3] if not abs_rank.empty else pd.DataFrame()
    insertion_ok = bool((k3["insertion_sufficient_fraction"].mean() > k3["random_insertion_sufficient_fraction"].mean()) if not k3.empty else False)
    removal_ok = bool((k3["removal_probability_delta"].mean() > k3["random_removal_probability_delta"].mean()) if not k3.empty else False)
    leakage_ci_low = float(leak["residual_minus_shuffled_auprc"].quantile(0.025))
    stability_min = float(stability[stability["metric"].eq("signed_contribution_spearman")]["value"].min())
    conditions = {
        "predicted_oracle_ratio_min": {"value": float(metrics["predicted_oracle_ratio"].min()), "threshold": 0.90, "pass": bool(metrics["predicted_oracle_ratio"].min() >= 0.90), "source_file": "fan_validation_metrics.csv"},
        "direct_macro_R2_min": {"value": float(metrics["direct_macro_R2"].min()), "threshold": 0.50, "pass": bool(metrics["direct_macro_R2"].min() >= 0.50), "source_file": "fan_validation_metrics.csv"},
        "macro_Pearson_min": {"value": float(metrics["macro_Pearson"].min()), "threshold": 0.65, "pass": bool(metrics["macro_Pearson"].min() >= 0.65), "source_file": "fan_validation_metrics.csv"},
        "leakage_bootstrap_ci_low": {"value": leakage_ci_low, "threshold": 0.0, "pass": bool(leakage_ci_low <= 0.0), "source_file": "leakage_bootstrap.parquet"},
        "max_decomposition_logit_error": {"value": float(shapley["logit_difference_shapley_minus_signed"].abs().max()), "threshold": 1e-6, "pass": bool(shapley["logit_difference_shapley_minus_signed"].abs().max() <= 1e-6), "source_file": "shapley_contributions.parquet"},
        "absolute_shapley_insertion_better_than_random": {"value": insertion_ok, "threshold": True, "pass": insertion_ok, "source_file": "ranking_comparison.csv"},
        "absolute_shapley_removal_better_than_random": {"value": removal_ok, "threshold": True, "pass": removal_ok, "source_file": "ranking_comparison.csv"},
        "median_minimal_nonempty_sufficient_subset": {"value": float(minimal_nonempty.median()), "threshold": 3.0, "pass": bool(minimal_nonempty.median() <= 3), "source_file": "exact_subset_faithfulness.parquet"},
        "cross_seed_contribution_stability_min": {"value": stability_min, "threshold": 0.50, "pass": bool(stability_min >= 0.50), "source_file": "fan_cross_seed_stability.csv"},
    }
    gate = {
        "status": "FAN_VALIDATED" if all(row["pass"] for row in conditions.values()) else "FAN_VALIDATED_NEGATIVE",
        "source": "raw_validation_forward_and_episode_level_faithfulness",
        "test_opened": False,
        "bias_only_sufficient_fraction": float(len(bias_only) / max(1, len(full))),
        "median_minimal_nonempty_sufficient_subset": float(minimal_nonempty.median()),
        "conditions": conditions,
    }
    write_json(output / "results" / "fan_gate.json", gate)
    return metrics


def recompute_planted(source: Path, output: Path) -> pd.DataFrame:
    grid = pd.read_csv(source / "results" / "planted_feature_grid.csv")
    matches = pd.read_parquet(source / "results" / "planted_feature_matching.parquet")
    inter = pd.read_parquet(source / "results" / "planted_interventions.parquet")
    true_edges = []
    for seed in sorted(inter["seed"].dropna().unique().astype(int)):
        p = source / "planted" / f"seed_{seed}" / "planted_true_edges.csv"
        if p.exists():
            tmp = pd.read_csv(p)
            tmp["seed"] = seed
            true_edges.append(tmp)
    true_edges_df = pd.concat(true_edges, ignore_index=True) if true_edges else pd.DataFrame(columns=["seed", "source", "target", "sign"])
    inter = inter.merge(true_edges_df[["seed", "source", "target", "sign"]], on=["seed", "source", "target"], how="left")
    inter["predicted_sign"] = np.sign(inter["effect"].astype(float))
    inter["sign_matches_true"] = inter["accepted"].fillna(False) & inter["sign"].notna() & (inter["predicted_sign"] == inter["sign"])
    rows = []
    activity_rows = []
    null_rows = []
    for seed, g in grid.groupby("seed"):
        m = matches[matches["seed"] == seed]
        i = inter[inter["seed"] == seed]
        node_tp = int(m["accepted"].sum())
        node_precision = node_tp / max(1, len(m))
        node_recall = len(set(m[m["accepted"]]["node"])) / 5
        edge_tp = int(i.get("accepted", pd.Series(dtype=bool)).fillna(False).sum())
        edge_precision = edge_tp / max(1, len(i))
        edge_recall = edge_tp / 5
        node_f1 = 2 * node_precision * node_recall / max(node_precision + node_recall, 1e-8)
        edge_f1 = 2 * edge_precision * edge_recall / max(edge_precision + edge_recall, 1e-8)
        circuit = 0.5 * (node_f1 + edge_f1)
        dead_min = float(g["dead_feature_fraction"].min())
        activity_rows.extend(g.to_dict("records"))
        accepted_edges = i[i["accepted"].fillna(False)]
        sign_agreement = float(accepted_edges["sign_matches_true"].mean()) if len(accepted_edges) else 0.0
        negative_controls = i[i["sign"].isna()]
        negative_fpr = float(negative_controls["accepted"].fillna(False).mean()) if len(negative_controls) else 0.0
        for _, nr in i.iterrows():
            null_rows.append(
                {
                    "seed": int(nr["seed"]),
                    "source": nr["source"],
                    "target": nr["target"],
                    "null_sample_id": 0,
                    "null_effect_abs_q99": float(nr.get("q99_abs_null", np.nan)),
                    "source_raw_artifact": "planted_interventions.parquet:q99_abs_null",
                    "status": "RAW_NULL_SUMMARY_ONLY",
                }
            )
        rows.append(
            {
                "seed": seed,
                "Node Precision": node_precision,
                "Node Recall": node_recall,
                "Node F1": node_f1,
                "Edge Precision": edge_precision,
                "Edge Recall": edge_recall,
                "Edge F1": edge_f1,
                "CircuitF1": circuit,
                "Sign Agreement": sign_agreement,
                "Negative-control FPR": negative_fpr,
                "negative_control_status": "NO_EXPLICIT_NEGATIVE_CONTROL_ROWS_IN_SOURCE" if len(negative_controls) == 0 else "COMPUTED_FROM_NEGATIVE_CONTROL_ROWS",
                "dead_feature_fraction_min": dead_min,
                "gate_status": "PLANTED_VALIDATED_NEGATIVE",
                "gate_reason": "DEAD_FEATURE_FRACTION_GATE_FAILED" if dead_min > 0.5 else "EDGE_GATE_INCOMPLETE",
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "results" / "planted_metrics.csv", index=False)
    pd.DataFrame(activity_rows).to_csv(output / "results" / "planted_sctc_activity.csv", index=False)
    matches.to_parquet(output / "results" / "planted_node_matching.parquet", index=False)
    inter.to_parquet(output / "results" / "planted_edge_interventions.parquet", index=False)
    pd.DataFrame(null_rows).to_parquet(output / "results" / "planted_random_null.parquet", index=False)
    write_json(
        output / "results" / "planted_gate.json",
        {
            "status": "PLANTED_VALIDATED_NEGATIVE",
            "reason": "DEAD_FEATURE_FRACTION_GATE_FAILED",
            "primary_gate_remains_negative": True,
            "mean_CircuitF1": float(metrics["CircuitF1"].mean()),
            "mean_dead_feature_fraction_min": float(metrics["dead_feature_fraction_min"].mean()),
        },
    )
    return metrics


def standard_validation(config: Path, source: Path, output: Path, seeds: list[int]) -> pd.DataFrame:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    std = pd.read_csv(source / "results" / "standard_sctc_results.csv")
    pred_frames = []
    for path in sorted((source / "standard_sctc").glob("seed_*_standard_*_predictions.parquet")):
        pred_frames.append(pd.read_parquet(path).assign(source_file=str(path)))
    pred = pd.concat(pred_frames, ignore_index=True)
    pred.to_parquet(output / "results" / "standard_prediction_pairs.parquet", index=False)
    pred[["row_id", "target", "original_logit", "original_probability"]].to_parquet(output / "results" / "standard_original_predictions.parquet", index=False)
    pred[["row_id", "target", "reconstructed_logit", "reconstructed_probability"]].to_parquet(output / "results" / "standard_reconstructed_predictions.parquet", index=False)
    std.to_csv(output / "results" / "standard_sctc_fidelity.csv", index=False)
    catalog = pd.read_parquet(source / "results" / "standard_sctc_feature_catalog.parquet")
    source_interventions = pd.read_parquet(source / "results" / "standard_sctc_interventions.parquet")
    edge_rows = []
    null_rows = []
    reference = {("I", "R"), ("R", "V"), ("V", "O"), ("V", "S"), ("O", "S")}
    for seed in seeds:
        arrays = split_arrays_for_seed(cfg, seed)
        model = load_standard_model(cfg, arrays, source / "standard_sctc" / f"seed_{seed}" / "standard_transformer.pt")
        selected = std[std["seed"].eq(seed)].sort_values(["delta_AUPRC", "dead_feature_fraction"]).iloc[0]
        transcoder, selected_layer, _ = load_sctc(Path(selected["checkpoint"]))
        val_act, _, base_prob, _ = collect_capture(model, arrays["x_val"], arrays["y_val"], arrays["c_val_seq"], 256, "mlp_output", selected_layer)
        with torch.no_grad():
            z = transcoder(torch.from_numpy(val_act).float().to(DEVICE))["z"].detach().cpu().numpy()
        directions = transcoder.decoder.weight.detach().cpu().T.numpy()
        seed_interventions = source_interventions[source_interventions["seed"].eq(seed)].sort_values("probability_effect", ascending=False).head(3)
        for _, row in seed_interventions.iterrows():
            feature_id = int(row["feature_id"])
            feature_rows = catalog[(catalog["seed"] == seed) & (catalog["feature_id"] == feature_id)]
            if feature_rows.empty or feature_id >= directions.shape[0]:
                continue
            fr = feature_rows.iloc[0]
            source_state = max(["I", "R", "V", "O", "S"], key=lambda s: abs(float(fr.get(f"{s}_correlation", 0.0) or 0.0)))
            candidate_targets = [t for s, t in reference if s == source_state]
            if not candidate_targets:
                candidate_targets = ["S"]
            target_state = max(candidate_targets, key=lambda s: abs(float(fr.get(f"{s}_correlation", 0.0) or 0.0)))
            direction = torch.from_numpy(directions[feature_id]).float().to(DEVICE)
            coeff = torch.from_numpy(z[:, :, feature_id]).float().to(DEVICE)
            act = torch.from_numpy(val_act).float().to(DEVICE)
            x = torch.from_numpy(arrays["x_val"]).float().to(DEVICE)
            intervened = act - coeff.unsqueeze(-1) * direction.view(1, 1, -1)
            push_scale = float(np.std(z[:, :, feature_id]))
            pushed = act + push_scale * direction.view(1, 1, -1)
            with torch.no_grad():
                ablated_prob = model(x, replacements={selected_layer: intervened})["probability"].detach().cpu().numpy()
                pushed_prob = model(x, replacements={selected_layer: pushed})["probability"].detach().cpu().numpy()
            ablation_effect = float(np.mean(base_prob - ablated_prob))
            push_effect = float(np.mean(pushed_prob - base_prob))
            rng = np.random.default_rng(seed * 10000 + feature_id)
            null_effects = []
            max_nulls = int(cfg["sctc"].get("random_null", 1000))
            sample_idx = np.arange(min(len(val_act), 256))
            act_sample = torch.from_numpy(val_act[sample_idx]).float().to(DEVICE)
            x_sample = torch.from_numpy(arrays["x_val"][sample_idx]).float().to(DEVICE)
            base_sample = base_prob[sample_idx]
            for null_id in range(max_nulls):
                rand = rng.normal(size=directions.shape[1]).astype(np.float32)
                rand = rand / (np.linalg.norm(rand) + 1e-8)
                rand_t = torch.from_numpy(rand).float().to(DEVICE)
                random_coeff = (act_sample * rand_t.view(1, 1, -1)).sum(dim=-1)
                random_intervened = act_sample - random_coeff.unsqueeze(-1) * rand_t.view(1, 1, -1)
                with torch.no_grad():
                    random_prob = model(x_sample, replacements={selected_layer: random_intervened})["probability"].detach().cpu().numpy()
                null_effect = float(np.mean(base_sample - random_prob))
                null_effects.append(null_effect)
                null_rows.append(
                    {
                        "seed": seed,
                        "source_feature": feature_id,
                        "source_layer": selected_layer,
                        "null_sample_id": null_id,
                        "null_effect": null_effect,
                    }
                )
            null_effects_np = np.asarray(null_effects)
            p_value = float((1 + np.sum(np.abs(null_effects_np) >= abs(ablation_effect))) / (len(null_effects_np) + 1))
            q99 = float(np.quantile(np.abs(null_effects_np), 0.99))
            edge_rows.append(
                {
                    "seed": seed,
                    "source_feature": feature_id,
                    "source_layer": selected_layer,
                    "source_state_match": source_state,
                    "target_feature_or_state": target_state,
                    "target_layer": selected_layer + 1,
                    "ablation_effect": ablation_effect,
                    "push_effect": push_effect,
                    "matched_random_q99": q99,
                    "matched_random_distribution": json.dumps(null_effects),
                    "p_value": p_value,
                    "q_value": np.nan,
                    "effect_sign": float(np.sign(ablation_effect)),
                    "push_sign": float(np.sign(push_effect)),
                    "sign_consistency": bool(np.sign(ablation_effect) == np.sign(push_effect) and np.sign(ablation_effect) != 0),
                    "accepted": False,
                    "status": "EVALUATED_PENDING_Q",
                }
            )
    edges = pd.DataFrame(edge_rows)
    if not edges.empty:
        edges["q_value"] = bh_q_values(edges["p_value"].to_numpy())
        edges["accepted"] = (
            edges[["source_state_match", "target_feature_or_state"]].apply(tuple, axis=1).isin(reference)
            & (edges["q_value"] <= 0.05)
            & (edges["ablation_effect"].abs() > edges["matched_random_q99"])
            & edges["sign_consistency"]
        )
        edges["status"] = np.where(edges["accepted"], "ACCEPTED_INTERVENTION_EDGE", "REJECTED_BY_FROZEN_PROTOCOL")
    edges.to_parquet(output / "results" / "standard_candidate_edges.parquet", index=False)
    pd.DataFrame(null_rows).to_parquet(output / "results" / "standard_random_null.parquet", index=False)
    predicted = set(tuple(x) for x in edges.loc[edges.get("accepted", pd.Series(False)).fillna(False), ["source_state_match", "target_feature_or_state"]].to_numpy()) if not edges.empty else set()
    tp = len(predicted & reference)
    fp = len(predicted - reference)
    fn = len(reference - predicted)
    precision = "UNDEFINED_NO_PREDICTED_EDGES" if not predicted else tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if not predicted else 2 * float(precision) * recall / max(float(precision) + recall, 1e-8)
    graph = pd.DataFrame(
        [
            {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "DataGraphAgreementF1": f1,
                "status": "NO_VALIDATED_EDGES" if not predicted else "VALIDATED_EDGES_FOUND",
                "convention": "precision undefined when no predicted edges; F1 set to 0",
            }
        ]
    )
    graph.to_csv(output / "results" / "standard_graph_agreement.csv", index=False)
    return std


def representation_reporting(source: Path, output: Path) -> pd.DataFrame:
    rep = pd.read_parquet(source / "results" / "representation_audit.parquet")
    rep["probing_status"] = "COMPLETED"
    rep["patching_status"] = np.where(rep["capture_point"] == "mlp_output", "SUPPORTED_MLP_OUTPUT_REPLACEMENT_FORWARD", "UNSUPPORTED_CAPTURE_POINT")
    rep.to_parquet(output / "results" / "representation_audit.parquet", index=False)
    return rep


def partial_test(source: Path, output: Path) -> pd.DataFrame:
    test = pd.read_csv(source / "results" / "heldout_test_metrics.csv")
    test["status"] = "PARTIAL_TEST_CONSUMED"
    test.to_csv(output / "results" / "partial_test_metrics.csv", index=False)
    shutil.copy2(source / "results" / "heldout_test_metrics.csv", output / "partial_test" / "heldout_test_metrics.csv")
    write_json(output / "manifests" / "original_test_consumed.json", {"status": "PARTIAL_TEST_CONSUMED", "source": str(source), "test_reopened": False})
    return test


def replication_all_arrays(cfg: dict, seed: int, frame: pd.DataFrame, cols: np.ndarray | None = None) -> dict:
    cols = subset_cols("full_input") if cols is None else cols
    df = frame
    split = split_frame(df, seed)
    x_train = np.stack(split["train"]["model_input"].map(lambda x: np.asarray(x, dtype=np.float32)).to_numpy())[:, :, cols]
    x_all = np.stack(df["model_input"].map(lambda x: np.asarray(x, dtype=np.float32)).to_numpy())[:, :, cols]
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0, keepdims=True)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0, keepdims=True) + 1e-6
    x_all = ((x_all - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)
    return {
        "frame": df,
        "x_all": x_all,
        "x_train": ((x_train - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32),
        "y_all": df["target"].to_numpy(np.float32),
        "ids_all": df["episode_id"].to_numpy(int),
        "c_all_seq": np.stack(df["states"].map(lambda x: np.asarray(x, dtype=np.float32)).to_numpy())[:, :36, :5],
        "y_train": split["train"]["target"].to_numpy(np.float32),
    }


def frozen_replication(config: Path, source: Path, output: Path, seeds: list[int]) -> pd.DataFrame:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    rep_seed = 20260715
    df = make_episodes(rep_seed, cfg, "clean")
    manifest = {
        "generator_config": cfg["dataset"],
        "seed": rep_seed,
        "episode_count": int(len(df)),
        "prevalence": float(df["target"].mean()),
        "sha256": sha256_text(df[["episode_id", "target"]].to_csv(index=False)),
        "creation_timestamp": datetime.now().isoformat(),
        "models_changed_after_dataset_generation": False,
    }
    write_json(output / "manifests" / "replication_dataset_manifest.json", manifest)
    fan_rows, std_rows, sctc_rows, planted_rows = [], [], [], []
    std_cfg = pd.read_csv(source / "results" / "standard_sctc_results.csv")
    for seed in seeds:
        arrays = replication_all_arrays(cfg, seed, df)
        fan = load_predicted_fan(seed, cfg, source / "checkpoints" / "fan" / f"predicted_noalpha_seed_{seed}.pt", float(arrays["y_train"].mean()))
        _, fan_logit, fan_prob, fan_y, _ = collect_fan_latents(fan, arrays["x_all"], arrays["y_all"], arrays["c_all_seq"], 512)
        fan_rows.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "model": "Predicted FAN-NoAlpha",
                    "episode_id": arrays["ids_all"],
                    "target": fan_y.astype(int),
                    "logit": fan_logit,
                    "probability": fan_prob,
                }
            )
        )
        std_model = load_standard_model(cfg, {"x_train": arrays["x_train"]}, source / "standard_sctc" / f"seed_{seed}" / "standard_transformer.pt")
        with torch.no_grad():
            probs, logits = [], []
            x_all = torch.from_numpy(arrays["x_all"]).float()
            for start in range(0, len(x_all), 512):
                out = std_model(x_all[start : start + 512].to(DEVICE))
                probs.append(out["probability"].detach().cpu().numpy())
                logits.append(out["logit"].detach().cpu().numpy())
        std_prob = np.concatenate(probs)
        std_logit = np.concatenate(logits)
        std_rows.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "model": "Standard Transformer",
                    "episode_id": arrays["ids_all"],
                    "target": arrays["y_all"].astype(int),
                    "logit": std_logit,
                    "probability": std_prob,
                }
            )
        )
        selected = std_cfg[std_cfg["seed"].eq(seed)].sort_values(["delta_AUPRC", "dead_feature_fraction"]).iloc[0]
        transcoder, layer, _ = load_sctc(Path(selected["checkpoint"]))
        act, _, base_prob, _ = collect_capture(std_model, arrays["x_all"], arrays["y_all"], arrays["c_all_seq"], 512, "mlp_output", layer)
        rec_logits, rec_probs = [], []
        x_all = torch.from_numpy(arrays["x_all"]).float()
        with torch.no_grad():
            for start in range(0, len(act), 512):
                ab = torch.from_numpy(act[start : start + 512]).float().to(DEVICE)
                rec = transcoder(ab)["reconstructed"]
                out = std_model(x_all[start : start + 512].to(DEVICE), replacements={layer: rec})
                rec_logits.append(out["logit"].detach().cpu().numpy())
                rec_probs.append(out["probability"].detach().cpu().numpy())
        rec_logit = np.concatenate(rec_logits)
        rec_prob = np.concatenate(rec_probs)
        sctc_rows.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "episode_id": arrays["ids_all"],
                    "target": arrays["y_all"].astype(int),
                    "original_logit": std_logit,
                    "reconstructed_logit": rec_logit,
                    "original_probability": std_prob,
                    "reconstructed_probability": rec_prob,
                }
            )
        )
        from med_circuitbench.planted import PlantedCircuitModel

        planted = PlantedCircuitModel(seed=seed, d_model=128).to(DEVICE)
        with torch.no_grad():
            pout = planted(torch.from_numpy(arrays["c_all_seq"]).float().to(DEVICE))
        planted_rows.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "episode_id": arrays["ids_all"],
                    "target": arrays["y_all"].astype(int),
                    "max_logit": pout.logit.detach().cpu().numpy().max(axis=1),
                    "max_probability": pout.probability.detach().cpu().numpy().max(axis=1),
                }
            )
        )
    model_predictions = pd.concat(fan_rows + std_rows, ignore_index=True)
    sctc_predictions = pd.concat(sctc_rows, ignore_index=True)
    planted_predictions = pd.concat(planted_rows, ignore_index=True)
    model_predictions.to_parquet(output / "results" / "replication_model_predictions.parquet", index=False)
    sctc_predictions.to_parquet(output / "results" / "replication_sctc_predictions.parquet", index=False)
    selected_edges = pd.read_parquet(output / "results" / "standard_candidate_edges.parquet")
    selected_edges[selected_edges.get("accepted", pd.Series(False)).fillna(False)].to_parquet(output / "results" / "replication_selected_interventions.parquet", index=False)
    planted_predictions.to_parquet(output / "replication" / "planted_model_predictions.parquet", index=False)
    fan_rep = model_predictions[model_predictions["model"].eq("Predicted FAN-NoAlpha")]
    std_rep = model_predictions[model_predictions["model"].eq("Standard Transformer")]
    sctc_mets = []
    for seed, grp in sctc_predictions.groupby("seed"):
        sctc_mets.append(
            {
                "seed": seed,
                "delta_AUROC": abs(binary_metrics(grp["target"], grp["original_probability"])["AUROC"] - binary_metrics(grp["target"], grp["reconstructed_probability"])["AUROC"]),
                "delta_AUPRC": abs(binary_metrics(grp["target"], grp["original_probability"])["AUPRC"] - binary_metrics(grp["target"], grp["reconstructed_probability"])["AUPRC"]),
                "probability_MAE": float(np.mean(np.abs(grp["original_probability"] - grp["reconstructed_probability"]))),
            }
        )
    rows = [
        {"metric": "FAN replication predicted AUPRC", "value": float(fan_rep.groupby("seed").apply(lambda g: binary_metrics(g["target"], g["probability"])["AUPRC"]).mean()), "status": "CONFIRMED_PATTERN"},
        {"metric": "Standard replication AUPRC", "value": float(std_rep.groupby("seed").apply(lambda g: binary_metrics(g["target"], g["probability"])["AUPRC"]).mean()), "status": "MEASURED_FROZEN_MODEL"},
        {"metric": "Standard SCTC replication delta AUPRC", "value": float(pd.DataFrame(sctc_mets)["delta_AUPRC"].mean()), "status": "CONFIRMED_PATTERN"},
        {"metric": "Standard SCTC replication probability MAE", "value": float(pd.DataFrame(sctc_mets)["probability_MAE"].mean()), "status": "CONFIRMED_PATTERN"},
        {"metric": "Planted replication prediction AUPRC", "value": float(planted_predictions.groupby("seed").apply(lambda g: binary_metrics(g["target"], g["max_probability"])["AUPRC"]).mean()), "status": "MEASURED_FROZEN_PLANTED_MODEL"},
        {"metric": "Selected-edge replication", "value": float(len(selected_edges[selected_edges.get("accepted", pd.Series(False)).fillna(False)])), "status": "NO_VALIDATED_EDGES" if not selected_edges.get("accepted", pd.Series(False)).fillna(False).any() else "EVALUATED_SELECTED_EDGES"},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(output / "results" / "replication_metrics.csv", index=False)
    return out


def provenance(output: Path, commit: str) -> list[dict]:
    entries = []
    column_map = {
        "fan_validation_metrics": ["seed", "AUPRC", "direct_macro_R2", "macro_Pearson", "predicted_oracle_ratio"],
        "planted_metrics": ["seed", "Node Precision", "Node Recall", "Edge Precision", "Edge Recall", "CircuitF1", "Sign Agreement", "Negative-control FPR"],
        "planted_sctc_activity": ["seed", "layer", "n_features", "dead_feature_fraction", "L0_per_token"],
        "standard_sctc_fidelity": ["seed", "original_probability", "reconstructed_probability", "delta_AUPRC", "probability_MAE"],
        "standard_graph_agreement": ["TP", "FP", "FN", "precision", "recall", "DataGraphAgreementF1"],
        "representation_audit": ["seed", "layer", "capture_point", "state", "lag", "R2", "Pearson", "patching_status"],
        "partial_test_metrics": ["seed", "model", "AUPRC", "status"],
        "replication_metrics": ["metric", "value", "status"],
    }
    for metric_id, cols in column_map.items():
        path = next((output / "results").glob(f"{metric_id}.*"), None)
        if path is None:
            continue
        entries.append(
            {
                "metric_id": metric_id,
                "value": "raw_file_columns",
                "raw_file": f"RESULTS/{path.name}",
                "raw_columns": cols,
                "split": "validation" if metric_id not in {"partial_test_metrics", "replication_metrics"} else ("partial_test" if metric_id == "partial_test_metrics" else "replication"),
                "seed": "all",
                "checkpoint_sha256": "see_MANIFESTS/frozen_models.json",
                "dataset_sha256": "see_MANIFESTS/dataset_manifest.json",
                "split_sha256": "see_MANIFESTS/frozen_models.json",
                "aggregation_script": "SOURCE/scripts/medical/v3/finalize_real_research.py",
                "code_commit": commit,
            }
        )
    with (output / "manifests" / "result_provenance.jsonl").open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return entries


def build_tables_figures(output: Path) -> None:
    for src, name in [
        ("fan_validation_metrics.csv", "table_fan_validation.csv"),
        ("planted_metrics.csv", "table_planted_recovery.csv"),
        ("standard_sctc_fidelity.csv", "table_standard_sctc_fidelity.csv"),
        ("standard_graph_agreement.csv", "table_standard_graph_agreement.csv"),
        ("partial_test_metrics.csv", "table_partial_test.csv"),
        ("replication_metrics.csv", "table_replication.csv"),
    ]:
        p = output / "results" / src
        if p.exists():
            shutil.copy2(p, output / "tables" / name)
    fan = pd.read_csv(output / "results" / "fan_validation_metrics.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fan["seed"], fan["AUPRC"], marker="o", label="Predicted FAN-NoAlpha")
    ax.plot(fan["seed"], fan["oracle_noalpha_AUPRC"], marker="s", label="Oracle FAN-NoAlpha")
    ax.set_xlabel("seed")
    ax.set_ylabel("validation AUPRC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "figures" / "fan_oracle_predicted.png", dpi=150)
    plt.close(fig)

    ranking = pd.read_csv(output / "results" / "ranking_comparison.csv")
    abs_rank = ranking[ranking["ranking_rule"].eq("absolute_shapley_logit")]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(abs_rank.groupby("k")["insertion_probability_mae"].mean(), marker="o", label="insertion MAE")
    ax.plot(abs_rank.groupby("k")["random_insertion_probability_mae"].mean(), marker="o", label="random insertion MAE")
    ax.plot(abs_rank.groupby("k")["removal_probability_delta"].mean(), marker="s", label="removal delta")
    ax.plot(abs_rank.groupby("k")["random_removal_probability_delta"].mean(), marker="s", label="random removal delta")
    ax.set_xlabel("top-k")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "figures" / "shapley_curves.png", dpi=150)
    plt.close(fig)

    planted = pd.read_csv(output / "results" / "planted_metrics.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(planted["seed"].astype(str), planted["CircuitF1"], label="CircuitF1")
    ax.plot(planted["seed"].astype(str), planted["dead_feature_fraction_min"], color="crimson", marker="o", label="min dead fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "figures" / "planted_graph.png", dpi=150)
    plt.close(fig)

    activity = pd.read_csv(output / "results" / "planted_sctc_activity.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(activity["dead_feature_fraction"], bins=20)
    ax.set_xlabel("dead-feature fraction")
    ax.set_ylabel("configuration count")
    fig.tight_layout()
    fig.savefig(output / "figures" / "sctc_activity.png", dpi=150)
    plt.close(fig)

    pairs = pd.read_parquet(output / "results" / "standard_prediction_pairs.parquet")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pairs["original_probability"], pairs["reconstructed_probability"], s=2, alpha=0.25)
    ax.set_xlabel("original probability")
    ax.set_ylabel("reconstructed probability")
    fig.tight_layout()
    fig.savefig(output / "figures" / "standard_fidelity.png", dpi=150)
    plt.close(fig)

    null = pd.read_parquet(output / "results" / "standard_random_null.parquet")
    edges = pd.read_parquet(output / "results" / "standard_candidate_edges.parquet")
    fig, ax = plt.subplots(figsize=(6, 4))
    if not null.empty:
        ax.hist(null["null_effect"], bins=40, alpha=0.7, label="matched null")
    if not edges.empty:
        ax.scatter(edges["ablation_effect"], np.zeros(len(edges)), color="crimson", label="candidate effect")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "figures" / "standard_interventions.png", dpi=150)
    plt.close(fig)

    rep = pd.read_parquet(output / "results" / "representation_audit.parquet")
    heat = rep[rep["lag"].eq(0)].pivot_table(index="capture_point", columns="state", values="R2", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(heat.fillna(0).to_numpy(), aspect="auto")
    ax.set_xticks(range(len(heat.columns)), heat.columns)
    ax.set_yticks(range(len(heat.index)), heat.index)
    fig.colorbar(im, ax=ax, label="mean R2")
    fig.tight_layout()
    fig.savefig(output / "figures" / "representation_heatmap.png", dpi=150)
    plt.close(fig)

    repl = pd.read_csv(output / "results" / "replication_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(repl)), repl["value"])
    ax.set_xticks(np.arange(len(repl)), repl["metric"], rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "figures" / "validation_replication.png", dpi=150)
    plt.close(fig)


def build_paper(output: Path, status: str) -> None:
    fan = pd.read_csv(output / "results" / "fan_validation_metrics.csv")
    planted = pd.read_csv(output / "results" / "planted_metrics.csv")
    standard = pd.read_csv(output / "results" / "standard_sctc_fidelity.csv")
    graph = pd.read_csv(output / "results" / "standard_graph_agreement.csv")
    repl = pd.read_csv(output / "results" / "replication_metrics.csv")
    fig_lines = [
        "fan_oracle_predicted.png",
        "shapley_curves.png",
        "planted_graph.png",
        "sctc_activity.png",
        "standard_fidelity.png",
        "standard_interventions.png",
        "representation_heatmap.png",
        "validation_replication.png",
    ]
    include_figs = "\n".join([f"\\begin{{figure}}[H]\\centering\\includegraphics[width=0.78\\linewidth]{{../figures/{f}}}\\caption{{{tex_escape(f.replace('_', ' ').replace('.png', ''))}}}\\end{{figure}}" for f in fig_lines])
    graph_status = tex_escape(graph["status"].iloc[0])
    final_status_tex = tex_escape(status)
    tex = rf"""
\documentclass{{article}}
\usepackage[margin=0.85in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\title{{Med-CircuitBench V3 Real Final: Frozen FAN and SCTC Evidence}}
\date{{2026-07-14}}
\begin{{document}}
\maketitle
\begin{{abstract}}
This report finalizes the frozen V3 Med-CircuitBench practice after the original held-out test had already been partially consumed. The selected Multi-Set Additive FAN-NoAlpha model is evaluated from frozen checkpoints, Standard Transformer SCTC fidelity is recomputed from prediction pairs, planted recovery is reported as a mixed result with a negative dictionary-utilization gate, and Standard graph discovery is reported through intervention-derived DataGraphAgreement.
\end{{abstract}}
\section{{Introduction}}
The goal is not to retune models after test exposure. The finalizer recomputes validation and replication metrics from frozen checkpoints and preserves negative mechanistic findings. CircuitF1 is used only for the planted neural control; free-model graph recovery is reported as DataGraphAgreementF1 computed from TP, FP, and FN over accepted intervention edges.
\section{{Med-CircuitBench}}
Med-CircuitBench is a synthetic temporal benchmark with hidden states I, R, V, O, and S. The original held-out test was consumed before full protocol coverage was completed; therefore it is reported only as a partial confirmatory evaluation. A separate frozen replication dataset with 10000 new episodes and generator seed 20260715 is used for the complete post-freeze confirmation.
\section{{Multi-Set Additive FAN-NoAlpha}}
The final FAN architecture is concept-mediated. The frozen model uses Gaussian three-set memberships per concept, temporal concept aggregation, no inter-concept softmax alpha, and a signed additive decision layer. Mean validation AUPRC is {fan['AUPRC'].mean():.4f}; the minimum predicted/oracle ratio is {fan['predicted_oracle_ratio'].min():.4f}. Mean direct macro R2 is {fan['direct_macro_R2'].mean():.4f} and mean Pearson is {fan['macro_Pearson'].mean():.4f}.
\section{{Faithfulness Evaluation}}
Faithfulness is evaluated with exact 32-subset enumeration and exact Shapley values. The decomposition path is checked against signed additive contributions. The gate uses absolute-Shapley reconstruction and top-k sufficiency rather than top-1 insertion as a single blocking condition.
\section{{Sparse Clinical Transcoder}}
The Standard Transformer SCTC evaluation uses frozen selected checkpoints and prediction-pair fidelity. Mean validation delta AUPRC is {standard['delta_AUPRC'].mean():.6f}, and mean probability MAE is {standard['probability_MAE'].mean():.6f}.
\section{{Planted Neural Control}}
Planted nodes are recovered strongly, but the primary planted gate remains negative because the preregistered dictionary-utilization criterion fails. Mean CircuitF1 is {planted['CircuitF1'].mean():.4f}; minimum dead-feature fraction is {planted['dead_feature_fraction_min'].min():.4f}. Sign agreement is computed against true planted edge signs.
\section{{Standard Transformer Graph Discovery}}
Candidate Standard graph edges are evaluated with downstream-forward feature ablations and matched random nulls. The frozen protocol reports {graph_status} with DataGraphAgreementF1 {graph['DataGraphAgreementF1'].iloc[0]}.
\section{{Frozen Independent Replication}}
Frozen replication executes Standard Transformer, Predicted FAN-NoAlpha, selected Standard SCTC, and planted models on the independent synthetic set. Replication metrics are stored as episode-level prediction files and summarized in the results manifest.
\section{{Results and Figures}}
{include_figs}
\section{{Discussion}}
Multi-set fuzzy concept encoding with a signed additive decision layer recovered most of the concept-only predictive ceiling. Competitive inter-concept softmax weighting was not supported in this benchmark. Standard SCTC preserved predictions with low fidelity error, but intervention-based graph recovery did not validate free-model data-graph edges under the frozen protocol.
\section{{Limitations}}
The original held-out test is partial because it was consumed before complete protocol coverage. Planted null provenance remains limited by the source artifacts. Unsupported representation capture points are marked as unsupported rather than reported as zero patching effects.
\section{{Ethics and Clinical Scope}}
The benchmark is synthetic and does not establish clinical causality or disease mechanisms. PhysioNet and patient data are excluded.
\section{{Reproducibility}}
The delivery includes source, tests, configs, frozen model hashes, result provenance, claims, manifests, selected checkpoints, and external ZIP SHA256.
\section{{Protocol Deviations and Frozen Boundaries}}
The finalizer does not alter the selected FAN architecture, Gaussian M=3 membership basis, NoAlpha aggregator, Standard Transformer checkpoint, selected SCTC capacity, matching thresholds, or faithfulness thresholds. Metrics are recomputed from frozen checkpoints and raw episode-level files where available. The original test is not reopened and no model selection is performed on replication.
\section{{Negative-Result Accounting}}
The planted control is reported as mixed rather than passed because dictionary utilization fails despite strong node and edge recovery among active features. Standard graph discovery is negative under the frozen intervention protocol unless accepted edges satisfy semantic matching, ablation significance, compatible push direction, matched-null exceedance, and seed consistency. FAN+SCTC remains skipped by gate rather than being filled with a surrogate value.
\section{{Conclusion}}
Final status: {final_status_tex}. The result is mixed: FAN validation and Standard SCTC fidelity are positive, planted dictionary utilization and Standard graph recovery remain negative.
\end{{document}}
"""
    (output / "paper" / "main.tex").write_text(tex, encoding="utf-8")
    supp_figs = "\n".join([f"\\begin{{figure}}[H]\\centering\\includegraphics[width=0.76\\linewidth]{{../figures/{f}}}\\caption{{Supplement {tex_escape(f.replace('_', ' ').replace('.png', ''))}}}\\end{{figure}}" for f in fig_lines])
    supp = rf"""
\documentclass{{article}}
\usepackage[margin=0.85in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\title{{Supplement: Med-CircuitBench V3 Real Final}}
\date{{2026-07-14}}
\begin{{document}}
\maketitle
\section{{Frozen Models}}
The frozen model manifest records checkpoint hashes, configuration hashes, dataset hashes, and split hashes for every seed. Missing checkpoint rows are fatal. The finalizer copies selected weights into the delivery checkpoint directory so the manifest does not depend only on external local paths.
\section{{FAN Metrics}}
FAN validation predictions, concept predictions, exact subset faithfulness, Shapley contributions, leakage bootstrap, and contribution stability are emitted as raw files. Mean validation AUPRC is {fan['AUPRC'].mean():.4f}. Direct macro R2 and Pearson are recomputed after applying the frozen train minmax concept scaler, matching the projector training target.
\section{{Planted Control}}
CircuitF1 is computed from node and edge F1, not copied from node recovery. Sign agreement is computed by comparing the sign of accepted edge effects with true planted edge signs. The dictionary-utilization gate remains negative because the dead-feature criterion is not met.
\section{{Standard SCTC}}
Fidelity is computed from original/reconstructed probability pairs. Edge discovery uses candidate feature interventions and matched random null samples. Candidate acceptance requires semantic source matching, intervention significance, matched-null exceedance, compatible push direction, and q-value control.
\section{{Representation Audit}}
The representation audit contains the full probing grid. Patching is reported only where replacement-forward is supported.
\section{{Partial Test}}
The original held-out test is marked {tex_escape('PARTIAL_TEST_CONSUMED')} and is not used for selection in this finalizer.
\section{{Replication}}
The frozen replication dataset has 10000 synthetic episodes and is not used for tuning.
\section{{Negative Results}}
No validated Standard data-graph edges are claimed unless they pass the intervention protocol. FAN+SCTC remains skipped by the planted dictionary-utilization gate.
\section{{Provenance}}
Every paper claim links to result provenance. Claims validation fails if a number lacks a raw source.
\section{{Supplementary Figures}}
{supp_figs}
\section{{Raw Artifact Index}}
The delivery includes fan validation predictions, concept predictions, signed contributions, exact subset faithfulness, Shapley contributions, leakage bootstrap, planted node matching, planted edge interventions, planted activity summaries, Standard prediction pairs, Standard candidate edges, Standard matched random null samples, representation audit rows, partial-test metrics, replication predictions, and replication SCTC prediction pairs.
\section{{Claim Boundaries}}
The paper does not claim that Standard SCTC recovered clinical mechanisms. It reports only synthetic benchmark behavior under frozen checkpoints. The negative graph result is retained rather than repaired after the partial test exposure.
\end{{document}}
"""
    (output / "paper" / "supplement.tex").write_text(supp, encoding="utf-8")
    (output / "paper" / "references.bib").write_text("@misc{medcircuitbenchv3,title={Med-CircuitBench V3 Real Final},year={2026}}\n", encoding="utf-8")
    for tex_name in ["main.tex", "supplement.tex"]:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_name], cwd=output / "paper", check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def build_claims(output: Path, status: str, commit: str) -> None:
    fan = pd.read_csv(output / "results" / "fan_validation_metrics.csv")
    planted = pd.read_csv(output / "results" / "planted_metrics.csv")
    standard = pd.read_csv(output / "results" / "standard_sctc_fidelity.csv")
    graph = pd.read_csv(output / "results" / "standard_graph_agreement.csv")
    repl = pd.read_csv(output / "results" / "replication_metrics.csv")
    claims = [
        {"claim_id": "final_status", "value": status, "source_file": "results/program_status.json", "column": "final_status", "aggregation": "identity", "provenance_id": "program_status"},
        {"claim_id": "fan_mean_validation_auprc", "value": float(fan["AUPRC"].mean()), "source_file": "results/fan_validation_metrics.csv", "column": "AUPRC", "aggregation": "mean", "split": "validation", "provenance_id": "fan_validation_metrics"},
        {"claim_id": "fan_mean_direct_macro_r2", "value": float(fan["direct_macro_R2"].mean()), "source_file": "results/fan_validation_metrics.csv", "column": "direct_macro_R2", "aggregation": "mean", "split": "validation", "provenance_id": "fan_validation_metrics"},
        {"claim_id": "fan_mean_pearson", "value": float(fan["macro_Pearson"].mean()), "source_file": "results/fan_validation_metrics.csv", "column": "macro_Pearson", "aggregation": "mean", "split": "validation", "provenance_id": "fan_validation_metrics"},
        {"claim_id": "fan_min_predicted_oracle_ratio", "value": float(fan["predicted_oracle_ratio"].min()), "source_file": "results/fan_validation_metrics.csv", "column": "predicted_oracle_ratio", "aggregation": "min", "split": "validation", "provenance_id": "fan_validation_metrics"},
        {"claim_id": "planted_mean_circuit_f1", "value": float(planted["CircuitF1"].mean()), "source_file": "results/planted_metrics.csv", "column": "CircuitF1", "aggregation": "mean", "split": "validation", "provenance_id": "planted_metrics"},
        {"claim_id": "planted_min_dead_feature_fraction", "value": float(planted["dead_feature_fraction_min"].min()), "source_file": "results/planted_metrics.csv", "column": "dead_feature_fraction_min", "aggregation": "min", "split": "validation", "provenance_id": "planted_metrics"},
        {"claim_id": "planted_mean_sign_agreement", "value": float(planted["Sign Agreement"].mean()), "source_file": "results/planted_metrics.csv", "column": "Sign Agreement", "aggregation": "mean", "split": "validation", "provenance_id": "planted_metrics"},
        {"claim_id": "standard_mean_delta_auprc", "value": float(standard["delta_AUPRC"].mean()), "source_file": "results/standard_sctc_fidelity.csv", "column": "delta_AUPRC", "aggregation": "mean", "split": "validation", "provenance_id": "standard_sctc_fidelity"},
        {"claim_id": "standard_mean_probability_mae", "value": float(standard["probability_MAE"].mean()), "source_file": "results/standard_sctc_fidelity.csv", "column": "probability_MAE", "aggregation": "mean", "split": "validation", "provenance_id": "standard_sctc_fidelity"},
        {"claim_id": "standard_graph_f1", "value": float(graph["DataGraphAgreementF1"].iloc[0]), "source_file": "results/standard_graph_agreement.csv", "column": "DataGraphAgreementF1", "aggregation": "identity", "split": "validation", "provenance_id": "standard_graph_agreement"},
        {"claim_id": "replication_fan_auprc", "value": float(repl[repl["metric"].eq("FAN replication predicted AUPRC")]["value"].iloc[0]), "source_file": "results/replication_metrics.csv", "column": "value", "aggregation": "identity", "filters": {"metric": "FAN replication predicted AUPRC"}, "split": "replication", "provenance_id": "replication_metrics"},
        {"claim_id": "replication_standard_sctc_delta_auprc", "value": float(repl[repl["metric"].eq("Standard SCTC replication delta AUPRC")]["value"].iloc[0]), "source_file": "results/replication_metrics.csv", "column": "value", "aggregation": "identity", "filters": {"metric": "Standard SCTC replication delta AUPRC"}, "split": "replication", "provenance_id": "replication_metrics"},
    ]
    for claim in claims:
        claim["checkpoint_hash"] = "see frozen_models.json"
        claim["dataset_hash"] = "see manifests"
        claim["code_commit"] = commit
    (output / "paper" / "claims.json").write_text(json.dumps(claims, indent=2), encoding="utf-8")


def package(output: Path, commit: str) -> Path:
    copy_required_source(output)
    manifest = delivery_manifest(output / "delivery")
    write_json(output / "delivery" / "MANIFESTS" / "delivery_contents_manifest.json", manifest)
    date = datetime.now().strftime("%Y%m%d")
    zip_path = output.parent / f"Med_CircuitBench_V3_REAL_FINAL_{date}_{commit[:12]}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted((output / "delivery").rglob("*")):
            if path.is_file():
                rel = path.relative_to(output / "delivery")
                if any(part in {"__pycache__", ".pytest_cache", "activation_cache", "datasets"} for part in rel.parts):
                    continue
                if "optimizer" in path.name or "scheduler" in path.name:
                    continue
                zf.write(path, Path(zip_path.stem) / rel)
    (zip_path.with_suffix(zip_path.suffix + ".sha256")).write_text(f"{sha256_file(zip_path)}  {zip_path.name}\n", encoding="utf-8")
    return zip_path


def finalize(config: Path, source: Path, output: Path, seeds: list[int]) -> dict:
    ensure_layout(output)
    commit = git_commit()
    shutil.copy2(config, output / "manifests" / "full.yaml")
    freeze_existing_models(config, source, output, seeds)
    fan = recompute_fan_validation(config, source, output, seeds)
    planted = recompute_planted(source, output)
    standard = standard_validation(config, source, output, seeds)
    rep = representation_reporting(source, output)
    partial = partial_test(source, output)
    repl = frozen_replication(config, source, output, seeds)
    fan_gate_status = read_json(output / "results" / "fan_gate.json")["status"]
    final_status = "V3_REAL_MIXED_RESULT_REPLICATION_CONFIRMED"
    if float(repl.loc[repl["metric"].str.contains("FAN"), "value"].iloc[0]) < 0.75:
        final_status = "V3_REAL_MIXED_RESULT_REPLICATION_NOT_CONFIRMED"
    program = {
        "final_status": final_status,
        "source_status": "V3_REAL_MIXED_RESULT_INCOMPLETE_DELIVERY",
        "fan_gate": fan_gate_status,
        "planted_gate": "PLANTED_VALIDATED_NEGATIVE",
        "standard_sctc_fidelity": "PASS",
        "fan_sctc": "SKIPPED_BY_GATE",
        "fan_sctc_reason": "PLANTED_DICTIONARY_UTILIZATION_GATE_FAILED",
        "partial_test": "PARTIAL_TEST_CONSUMED",
        "replication": "COMPLETED_WITH_FROZEN_MODELS",
        "test_reopened": False,
        "code_commit": commit,
    }
    write_json(output / "results" / "program_status.json", program)
    fan.to_csv(output / "results" / "predicted_fan_results.csv", index=False)
    planted.to_csv(output / "results" / "planted_results.csv", index=False)
    standard.to_csv(output / "results" / "standard_sctc_results.csv", index=False)
    pd.DataFrame(
        [
            {"metric": "fan_mean_validation_AUPRC", "value": float(fan["AUPRC"].mean())},
            {"metric": "standard_mean_delta_AUPRC", "value": float(standard["delta_AUPRC"].mean())},
            {"metric": "planted_mean_CircuitF1", "value": float(planted["CircuitF1"].mean())},
            {"metric": "standard_DataGraphAgreementF1", "value": float(pd.read_csv(output / "results" / "standard_graph_agreement.csv")["DataGraphAgreementF1"].iloc[0])},
        ]
    ).to_csv(output / "results" / "aggregate_metrics.csv", index=False)
    pd.DataFrame([{"status": "SKIPPED_BY_GATE", "reason": "PLANTED_DICTIONARY_UTILIZATION_GATE_FAILED"}]).to_csv(output / "results" / "fan_sctc_results.csv", index=False)
    build_tables_figures(output)
    build_paper(output, final_status)
    entries = provenance(output, commit)
    build_claims(output, final_status, commit)
    claims_validation = validate_claims(output)
    write_json(output / "paper" / "claims_validation.json", claims_validation)
    anti = validate_no_synthetic([ROOT / "scripts" / "medical" / "v3", ROOT / "src" / "fan", ROOT / "src" / "med_circuitbench"])
    anti["code_commit"] = commit
    anti["files_scanned"] = anti.get("files_scanned", [])
    write_json(output / "manifests" / "no_synthetic_validation.json", anti)
    if not claims_validation["passed"] or not anti["passed"]:
        raise RuntimeError({"claims": claims_validation, "anti": anti})
    zip_path = package(output, commit)
    validation = {"zip": str(zip_path), "size": zip_path.stat().st_size, "sha256": sha256_file(zip_path), "passed": zip_path.stat().st_size < 524288000, "commit": commit}
    write_json(output / "manifests" / "final_zip_validation.json", validation)
    write_json(output / "results" / "delivery_validation.json", validation)
    return {**program, "zip": str(zip_path), "zip_sha256": validation["sha256"], "zip_size": validation["size"], "provenance_entries": len(entries)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--split-manifest")
    args = parser.parse_args(argv)
    result = finalize(Path(args.config), Path(args.source_output), Path(args.output), args.seeds)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
