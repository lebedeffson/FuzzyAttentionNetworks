#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy import stats
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fan.concept.stability import center_anchor_loss, contribution_consistency_loss, make_masked_view  # noqa: E402
from fan.evaluation.predictive import binary_classification_metrics  # noqa: E402
from scripts.medical.v2.run_v2_1_program import make_sequence_loaders  # noqa: E402
from scripts.medical.v2.run_v2_program import make_episodes, split_frame, subset_cols  # noqa: E402
from scripts.medical.v3.run_fan_iteration import ConceptScaler, build_loaders, set_all_seeds  # noqa: E402
from scripts.medical.v3.run_predicted_fan_strict import (  # noqa: E402
    collect_concept_predictions,
    exact_faithfulness,
    make_predicted_model,
    trajectory_metrics,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_baseline_model(cfg: dict, checkpoint: Path, prevalence: float):
    model = make_predicted_model(cfg, "no_alpha", prevalence)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    return model.to(DEVICE)


def evaluate_model(model, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, p, contrib = [], [], []
    model.eval()
    with torch.no_grad():
        for xb, yb, _ in loader:
            out = model(xb.to(DEVICE))
            y.append(yb.numpy())
            p.append(out.probability.detach().cpu().numpy())
            contrib.append(out.signed_decision_contributions.detach().cpu().numpy())
    return np.concatenate(y), np.concatenate(p), np.concatenate(contrib)


def train_stable(model, train_loader, val_loader, method_cfg: dict, lambda_stability: float, seed: int) -> tuple[list[dict], float]:
    model.freeze_concept_path()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params,
        lr=float(method_cfg["learning_rate"]),
        weight_decay=float(method_cfg["weight_decay"]),
    )
    initial_centers = model.membership.centers.detach().clone()
    best_state = None
    best_auprc = -1.0
    rows = []
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed * 1000 + 909)
    for epoch in range(int(method_cfg["max_epochs"])):
        losses = []
        model.train()
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out_a = model(xb)
            xb_view = make_masked_view(
                xb,
                mask_probability=float(method_cfg["mask_probability"]),
                noise_std=float(method_cfg["noise_std"]),
                generator=generator,
            )
            out_b = model(xb_view)
            task = F.binary_cross_entropy_with_logits(out_a.logit, yb)
            consistency = contribution_consistency_loss(out_a.signed_decision_contributions, out_b.signed_decision_contributions)
            anchor = center_anchor_loss(model.membership.centers, initial_centers)
            loss = task + float(lambda_stability) * consistency + float(method_cfg["lambda_center_anchor"]) * anchor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(
                {
                    "task_loss": float(task.detach().cpu()),
                    "contribution_consistency_loss": float(consistency.detach().cpu()),
                    "center_anchor_loss": float(anchor.detach().cpu()),
                    "total_loss": float(loss.detach().cpu()),
                }
            )
        y, p, _ = evaluate_model(model, val_loader)
        auprc = float(average_precision_score(y, p))
        row = {"epoch": epoch + 1, "lambda_stability": lambda_stability, "validation_AUPRC": auprc}
        for key in losses[0]:
            row[key] = float(np.mean([x[key] for x in losses]))
        rows.append(row)
        if auprc > best_auprc:
            best_auprc = auprc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return rows, best_auprc


def contribution_stability(models: dict[int, torch.nn.Module], ref_loader) -> tuple[pd.DataFrame, dict]:
    per_seed = {}
    for seed, model in models.items():
        _, _, contrib = evaluate_model(model, ref_loader)
        per_seed[seed] = contrib.mean(axis=0)
    rows = []
    seeds = sorted(per_seed)
    for i, a in enumerate(seeds):
        for b in seeds[i + 1 :]:
            rho = float(stats.spearmanr(per_seed[a], per_seed[b]).statistic)
            tau = float(stats.kendalltau(per_seed[a], per_seed[b]).statistic)
            top_a = set(np.argsort(-np.abs(per_seed[a]))[:3].tolist())
            top_b = set(np.argsort(-np.abs(per_seed[b]))[:3].tolist())
            rows.append(
                {
                    "seed_a": a,
                    "seed_b": b,
                    "signed_contribution_spearman": rho,
                    "signed_contribution_kendall_tau": tau,
                    "top3_jaccard": len(top_a & top_b) / max(1, len(top_a | top_b)),
                }
            )
    frame = pd.DataFrame(rows)
    return frame, {
        "mean_pairwise_contribution_spearman": float(frame["signed_contribution_spearman"].mean()) if len(frame) else float("nan"),
        "mean_pairwise_contribution_kendall_tau": float(frame["signed_contribution_kendall_tau"].mean()) if len(frame) else float("nan"),
        "mean_top3_jaccard": float(frame["top3_jaccard"].mean()) if len(frame) else float("nan"),
    }


def write_bundle(bundle_root: Path, seed: int, model_name: str, checkpoint: Path, cfg: dict, metrics: dict, commit: str) -> None:
    bundle_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, bundle_root / "checkpoint.pt")
    manifest = {
        "bundle_type": "fan",
        "model_name": model_name,
        "seed": seed,
        "checkpoint": "checkpoint.pt",
        "checkpoint_sha256": sha256_file(bundle_root / "checkpoint.pt"),
        "code_commit": commit,
        "saved_metrics": metrics,
        "mechanistic_recovery_default": False,
        "model_config": {
            "input_dim": int(cfg["model"]["input_dim"]),
            "sequence_length": int(cfg["dataset"]["observed_window"]),
            "latent_dim": int(cfg["model"]["latent_dim"]),
            "n_concepts": 5,
            "n_memberships": 3,
            "membership": "gaussian",
            "temporal_mode": "attention",
            "dropout": float(cfg["model"].get("dropout", 0.1)),
            "encoder_layers": int(cfg["model"].get("layers", 4)),
            "encoder_heads": int(cfg["model"].get("heads", 4)),
            "encoder_ffn": int(cfg["model"].get("d_ffn", 512)),
            "alpha_mode": "no_alpha",
            "positive_decision_weights": False,
        },
    }
    (bundle_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_seed(seed: int, cfg: dict, method_cfg: dict, output: Path, baseline_metrics: pd.DataFrame) -> tuple[dict, torch.nn.Module]:
    set_all_seeds(seed)
    clean = make_episodes(seed, cfg, "clean")
    split = split_frame(clean, seed)
    _, _, arrays = make_sequence_loaders(split, subset_cols("full_input"), int(method_cfg["batch_size"]), 5)
    scaler = ConceptScaler.fit(arrays["c_train_seq"], "minmax_train")
    train_loader, val_loader, _ = build_loaders(arrays, scaler, 5, int(method_cfg["batch_size"]))
    ckpt = Path(method_cfg["baseline_checkpoint_root"]) / f"seed_{seed}" / "Predicted_Temporal_FAN_NoAlpha_Strict" / "checkpoint.pt"
    model = load_baseline_model(cfg, ckpt, float(arrays["y_train"].mean()))
    baseline_row = baseline_metrics[baseline_metrics["seed"].astype(int).eq(seed)].iloc[0].to_dict()
    selected_lambda = float(method_cfg["lambda_stability"])
    histories, best_models = {}, {}
    for lam in [selected_lambda]:
        candidate = load_baseline_model(cfg, ckpt, float(arrays["y_train"].mean()))
        history, _ = train_stable(candidate, train_loader, val_loader, method_cfg, lam, seed)
        histories[lam] = history
        best_models[lam] = candidate
    y, p, _ = evaluate_model(best_models[selected_lambda], val_loader)
    selected_metrics = binary_classification_metrics(y, p)
    if float(baseline_row["AUPRC"]) - selected_metrics["AUPRC"] > float(method_cfg["fallback_if_auprc_drop_gt"]):
        fallback = float(method_cfg["safety_fallback_lambda_stability"])
        candidate = load_baseline_model(cfg, ckpt, float(arrays["y_train"].mean()))
        history, _ = train_stable(candidate, train_loader, val_loader, method_cfg, fallback, seed)
        histories[fallback] = history
        y_fb, p_fb, _ = evaluate_model(candidate, val_loader)
        fallback_metrics = binary_classification_metrics(y_fb, p_fb)
        if fallback_metrics["AUPRC"] >= selected_metrics["AUPRC"]:
            selected_lambda = fallback
            selected_metrics = fallback_metrics
            model = candidate
            y, p = y_fb, p_fb
        else:
            model = best_models[float(method_cfg["lambda_stability"])]
    else:
        model = best_models[selected_lambda]
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    for lam, history in histories.items():
        pd.DataFrame(history).to_csv(seed_dir / f"stable_training_lambda_{lam:.3f}.csv", index=False)
    pred = pd.DataFrame({"seed": seed, "target": y.astype(int), "probability": p.astype(float)})
    pred.to_parquet(seed_dir / "fan_stable_predictions.parquet", index=False)
    torch.save(model.state_dict(), seed_dir / "fan_noalpha_stable.pt")
    _, train_true, train_pred = collect_concept_predictions(model, train_loader)
    _, val_true, val_pred = collect_concept_predictions(model, val_loader)
    concept_met = trajectory_metrics(val_true, val_pred, train_true, train_pred)
    sub, shp, rnk, faith = exact_faithfulness(model, val_loader, seed)
    sub.to_parquet(seed_dir / "exact_subset_faithfulness.parquet", index=False)
    shp.to_parquet(seed_dir / "shapley_contributions.parquet", index=False)
    rnk.to_csv(seed_dir / "ranking_comparison.csv", index=False)
    row = {
        "seed": seed,
        "model": "FAN-NoAlpha-Stable",
        "selected_lambda_stability": selected_lambda,
        "baseline_AUPRC": float(baseline_row["AUPRC"]),
        **selected_metrics,
        "AUPRC_drop_vs_baseline": float(baseline_row["AUPRC"]) - selected_metrics["AUPRC"],
        "direct_macro_R2": concept_met["macro_direct_trajectory_r2"],
        "macro_Pearson": concept_met["mean_trajectory_pearson"],
        **faith,
    }
    return row, model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method-config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw_method = yaml.safe_load(Path(args.method_config).read_text(encoding="utf-8"))
    method_cfg = raw_method["fan_noalpha_stable"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.method_config, output / "resolved_config.yaml")
    baseline = pd.read_csv(method_cfg["baseline_results"])
    rows, models = [], {}
    for seed in args.seeds:
        row, model = run_seed(seed, cfg, method_cfg, output, baseline)
        rows.append(row)
        models[seed] = model
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "fan_stable_metrics.csv", index=False)
    ref_seed = args.seeds[0]
    ref_frame = make_episodes(ref_seed, cfg, "clean")
    ref_split = split_frame(ref_frame, ref_seed)
    _, _, ref_arrays = make_sequence_loaders(ref_split, subset_cols("full_input"), int(method_cfg["batch_size"]), 5)
    scaler = ConceptScaler.fit(ref_arrays["c_train_seq"], "minmax_train")
    _, ref_loader, _ = build_loaders(ref_arrays, scaler, 5, int(method_cfg["batch_size"]))
    stability_df, stability_summary = contribution_stability(models, ref_loader)
    stability_df.to_csv(output / "fan_cross_seed_stability.csv", index=False)
    gates_cfg = method_cfg["gates"]
    auprc_pass = metrics["AUPRC_drop_vs_baseline"].le(float(gates_cfg["auprc_drop_max"]))
    r2_pass = metrics["direct_macro_R2"].ge(float(gates_cfg["direct_macro_r2_min"]))
    pearson_pass = metrics["macro_Pearson"].ge(float(gates_cfg["pearson_min"]))
    faith_pass = metrics["max_decomposition_logit_error"].le(float(gates_cfg["decomposition_error_max"])) & metrics[
        "median_minimal_nonempty_sufficient_subset_size"
    ].le(float(gates_cfg["median_minimal_subset_max"]))
    contribution_stability_pass = stability_summary["mean_pairwise_contribution_spearman"] >= float(gates_cfg["contribution_spearman_min"])
    gate = {
        "status": "FAN_NOALPHA_STABLE_SELECTED"
        if int((auprc_pass & r2_pass & pearson_pass & faith_pass).sum()) >= 2 and contribution_stability_pass
        else "FAN_NOALPHA_BASELINE_RETAINED",
        "stable_seed_pass_count": int((auprc_pass & r2_pass & pearson_pass & faith_pass).sum()),
        "contribution_stability_pass": bool(contribution_stability_pass),
        **stability_summary,
        "production_model": "FAN-NoAlpha-Stable" if contribution_stability_pass else "FAN-NoAlpha",
        "sctc_development_status": "TERMINAL_NEGATIVE_CLOSED",
    }
    (output / "fan_stable_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    commit = "UNKNOWN"
    try:
        import subprocess

        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        pass
    bundle_root = output / "bundles" / ("fan_noalpha_stable" if gate["production_model"] == "FAN-NoAlpha-Stable" else "fan_noalpha")
    for row in rows:
        seed = int(row["seed"])
        ckpt = output / f"seed_{seed}" / "fan_noalpha_stable.pt"
        if gate["production_model"] == "FAN-NoAlpha":
            ckpt = Path(method_cfg["baseline_checkpoint_root"]) / f"seed_{seed}" / "Predicted_Temporal_FAN_NoAlpha_Strict" / "checkpoint.pt"
        write_bundle(bundle_root / f"seed{seed}", seed, str(gate["production_model"]), ckpt, cfg, row, commit)
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
