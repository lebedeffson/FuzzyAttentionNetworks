#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fan.concept import MultiSetAdditiveTemporalConceptFANModel
from scripts.medical.v2.run_v2_1_program import make_sequence_loaders
from scripts.medical.v2.run_v2_program import make_episodes, split_frame, subset_cols


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_NAMES = ["I", "R", "V", "O", "S"]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "AUROC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "AUPRC": float(average_precision_score(y, p)),
    }


@dataclass(frozen=True)
class ConceptScaler:
    kind: str
    center: np.ndarray
    scale: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None

    @classmethod
    def fit(cls, x: np.ndarray, kind: str) -> "ConceptScaler":
        flat = x.reshape(-1, x.shape[-1])
        if kind == "minmax_train":
            lower = flat.min(axis=0)
            upper = flat.max(axis=0)
            return cls(kind, lower, np.maximum(upper - lower, 1e-6), lower, upper)
        if kind == "robust_train":
            q25, q50, q75 = np.quantile(flat, [0.25, 0.5, 0.75], axis=0)
            return cls(kind, q50, np.maximum(q75 - q25, 1e-6), q25, q75)
        raise ValueError(f"Unknown scaler {kind}")

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "minmax_train":
            return np.clip((x - self.center) / self.scale, 0.0, 1.0).astype(np.float32)
        return (1.0 / (1.0 + np.exp(-((x - self.center) / self.scale)))).astype(np.float32)

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "lower": None if self.lower is None else self.lower.tolist(),
            "upper": None if self.upper is None else self.upper.tolist(),
        }


def build_loaders(arrays: dict, scaler: ConceptScaler, n_concepts: int, batch_size: int) -> tuple[DataLoader, DataLoader, dict]:
    c_train = scaler.transform(arrays["c_train_seq"][:, :, :n_concepts])
    c_val = scaler.transform(arrays["c_val_seq"][:, :, :n_concepts])
    train = TensorDataset(torch.from_numpy(arrays["x_train"]), torch.from_numpy(arrays["y_train"]), torch.from_numpy(c_train))
    val = TensorDataset(torch.from_numpy(arrays["x_val"]), torch.from_numpy(arrays["y_val"]), torch.from_numpy(c_val))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, {"c_train": c_train, "c_val": c_val}


def make_model(cfg: dict, n_concepts: int, family: str, n_memberships: int, temperature: float, prevalence: float) -> MultiSetAdditiveTemporalConceptFANModel:
    fan_cfg = cfg.get("fan", {})
    model = MultiSetAdditiveTemporalConceptFANModel(
        input_dim=int(cfg["model"]["input_dim"]),
        sequence_length=int(cfg["dataset"]["observed_window"]),
        latent_dim=int(cfg["model"]["latent_dim"]),
        n_concepts=n_concepts,
        n_memberships=n_memberships,
        membership=family,
        oracle=True,
        temporal_mode="attention",
        dropout=float(cfg["model"].get("dropout", 0.1)),
        encoder_layers=int(cfg["model"].get("layers", 4)),
        encoder_heads=int(cfg["model"].get("heads", 4)),
        encoder_ffn=int(cfg["model"].get("d_ffn", 512)),
        temperature=temperature,
        positive_decision_weights=bool(fan_cfg.get("positive_decision_weights", False)),
        alpha_mode=str(fan_cfg.get("alpha_mode", "softmax_alpha")),
        gamma_init=float(fan_cfg.get("gamma_init", 0.5)),
        gamma_max=float(fan_cfg.get("gamma_max", 0.9)),
    ).to(DEVICE)
    model.decision_head.initialize_bias_from_prevalence(prevalence)
    return model


def initialize_memberships(model: MultiSetAdditiveTemporalConceptFANModel, c_train: np.ndarray) -> None:
    summaries = torch.from_numpy(c_train.mean(axis=1)).to(DEVICE)
    model.membership.initialize_from_quantiles(summaries)


def train_oracle(model: MultiSetAdditiveTemporalConceptFANModel, cfg: dict, train_loader: DataLoader, val_loader: DataLoader, seed: int) -> list[dict]:
    lrs = cfg["fan"].get("oracle_lrs", {})
    max_epochs = int(cfg["training"].get("max_epochs", 50))
    freeze_membership_epochs = int(cfg["fan"].get("freeze_membership_epochs", 10))
    freeze_alpha_epochs = int(cfg["fan"].get("freeze_alpha_epochs", 5))
    alpha_params = [p for name, p in model.aggregator.named_parameters() if name != "raw_temperature"]
    params = [
        {"params": model.decision_head.parameters(), "lr": float(lrs.get("decision_head_lr", 3e-4))},
        {"params": alpha_params, "lr": float(lrs.get("alpha_lr", 1e-4))},
        {"params": model.membership.parameters(), "lr": float(lrs.get("membership_lr", 3e-5))},
        {"params": [model.aggregator.raw_temperature], "lr": float(lrs.get("temperature_lr", 1e-5))},
    ]
    opt = torch.optim.AdamW(params, weight_decay=float(cfg["training"].get("weight_decay", 0.0)))
    best, best_state, history = -1.0, None, []
    for epoch in range(max_epochs):
        model.freeze_alpha(epoch < freeze_alpha_epochs)
        model.freeze_membership(epoch < freeze_membership_epochs)
        model.train()
        losses = []
        for xb, yb, cb in train_loader:
            xb, yb, cb = xb.to(DEVICE), yb.to(DEVICE), cb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(xb, cb)
            entropy = -(out.concept_weights * torch.log(out.concept_weights + 1e-8)).sum(dim=1).mean()
            loss = F.binary_cross_entropy_with_logits(out.logit, yb) + float(cfg["fan"].get("lambda_sparse", 0.0)) * entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["fan"].get("gradient_clip", 1.0)))
            opt.step()
            losses.append(float(loss.item()))
        y, p, _ = evaluate_model(model, val_loader)
        auprc = average_precision_score(y, p)
        history.append({"seed": seed, "epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_AUPRC": float(auprc)})
        if auprc > best:
            best = float(auprc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def evaluate_model(model: MultiSetAdditiveTemporalConceptFANModel, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, dict]:
    model.eval()
    labels, probs = [], []
    extras = {k: [] for k in ["summaries", "memberships", "local_weights", "fuzzy_values", "alpha", "evidence", "signed", "beta"]}
    with torch.no_grad():
        for xb, yb, cb in loader:
            xb, cb = xb.to(DEVICE), cb.to(DEVICE)
            out = model(xb, cb)
            labels.append(yb.numpy())
            probs.append(out.probability.cpu().numpy())
            extras["summaries"].append(out.concept_summaries.cpu().numpy())
            extras["memberships"].append(out.memberships.cpu().numpy())
            extras["local_weights"].append(out.membership_local_weights.cpu().numpy())
            extras["fuzzy_values"].append(out.fuzzy_values.cpu().numpy())
            extras["alpha"].append(out.concept_weights.cpu().numpy())
            extras["evidence"].append(out.concept_evidence.cpu().numpy())
            extras["signed"].append(out.signed_decision_contributions.cpu().numpy())
            extras["beta"].append(out.temporal_concept_weights.cpu().numpy())
    return np.concatenate(labels), np.concatenate(probs), {k: np.concatenate(v, axis=0) for k, v in extras.items()}


def fit_prob(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, kind: str = "logistic") -> np.ndarray:
    if kind == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=0)
    else:
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(xtr, ytr)
    return model.predict_proba(xva)[:, 1]


def concept_ceiling(arrays: dict, scaler: ConceptScaler, n_concepts: int) -> tuple[float, pd.DataFrame]:
    c_train = scaler.transform(arrays["c_train_seq"][:, :, :n_concepts])
    c_val = scaler.transform(arrays["c_val_seq"][:, :, :n_concepts])
    feats_train = concept_features(c_train)
    feats_val = concept_features(c_val)
    ytr = arrays["y_train"].astype(int)
    yva = arrays["y_val"].astype(int)
    rows = []
    for name, kind in [("direct_logistic", "logistic"), ("direct_mlp", "mlp")]:
        p = fit_prob(feats_train, ytr, feats_val, kind)
        rows.append({"model": name, **binary_metrics(yva, p)})
    df = pd.DataFrame(rows)
    return float(df["AUPRC"].max()), df


def concept_features(seq: np.ndarray) -> np.ndarray:
    last = seq[:, -1, :]
    mean = seq.mean(axis=1)
    maxv = seq.max(axis=1)
    last6 = seq[:, -6:, :]
    mean6 = last6.mean(axis=1)
    slope6 = last6[:, -1, :] - last6[:, 0, :]
    return np.concatenate([last, mean, maxv, mean6, slope6], axis=1)


def info_path(seed: int, family: str, n_memberships: int, extras: dict, y: np.ndarray, direct_prob: np.ndarray) -> pd.DataFrame:
    stages = {
        "q_additive_classifier": extras["summaries"],
        "memberships_additive_classifier": extras["memberships"].reshape(len(y), -1),
        "fuzzy_values_additive_classifier": extras["fuzzy_values"],
        "alpha_times_fuzzy_values": extras["evidence"],
        "signed_contributions": extras["signed"],
    }
    rows = []
    for stage, x in stages.items():
        p = fit_prob(x, y.astype(int), x)
        rows.append(
            {
                "seed": seed,
                "membership_family": family,
                "n_memberships": n_memberships,
                "representation_stage": stage,
                **binary_metrics(y, p),
                "probability_MAE_vs_direct": float(np.mean(np.abs(p - direct_prob))),
            }
        )
    return pd.DataFrame(rows)


def faithfulness(seed: int, model: MultiSetAdditiveTemporalConceptFANModel, loader: DataLoader, label: dict) -> pd.DataFrame:
    model.eval()
    rng = torch.Generator(device=DEVICE).manual_seed(seed)
    rows = []
    episode_id = 0
    with torch.no_grad():
        for xb, yb, cb in loader:
            xb, cb = xb.to(DEVICE), cb.to(DEVICE)
            out = model(xb, cb)
            base = out.probability
            evidence = out.concept_evidence
            signed = out.signed_decision_contributions
            predicted_positive = base >= 0.5
            rank_score = torch.where(predicted_positive.unsqueeze(1), signed, -signed)
            top = torch.argsort(rank_score, dim=1, descending=True)[:, :1]
            bottom = torch.argsort(rank_score, dim=1, descending=False)[:, :1]
            random_idx = torch.stack([torch.randperm(evidence.shape[1], generator=rng, device=DEVICE)[:1] for _ in range(evidence.shape[0])])
            for intervention, idx in [("top1_removal", top), ("random_removal", random_idx), ("bottom1_removal", bottom), ("top1_insertion", top), ("random_insertion", random_idx)]:
                if "insertion" in intervention:
                    altered = torch.zeros_like(evidence)
                    altered.scatter_(1, idx, evidence.gather(1, idx))
                else:
                    altered = evidence.clone()
                    altered.scatter_(1, idx, 0.0)
                p = torch.sigmoid(model.decision_from_evidence(altered))
                for yy, b, pp in zip(yb.numpy(), base.cpu().numpy(), p.cpu().numpy()):
                    rows.append(
                        {
                            **label,
                            "episode_id": episode_id,
                            "intervention": intervention,
                            "target": int(yy),
                            "base_probability": float(b),
                            "intervened_probability": float(pp),
                            "probability_delta": float(pp - b),
                            "abs_probability_delta": float(abs(pp - b)),
                        }
                    )
                    episode_id += 1
    return pd.DataFrame(rows)


def ci_lower(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(d.mean() - 1.96 * d.std(ddof=0) / math.sqrt(max(1, len(d))))


def run_oracle_iteration(seed: int, cfg: dict, spec: dict, output: Path) -> dict:
    set_all_seeds(seed)
    clean = make_episodes(seed, cfg, "clean")
    split = split_frame(clean, seed)
    train_loader_raw, val_loader_raw, arrays = make_sequence_loaders(split, subset_cols("full_input"), int(cfg["training"]["batch_size"]), 5)
    scaler = ConceptScaler.fit(arrays["c_train_seq"][:, :, :5], spec["concept_scaling"])
    train_loader, val_loader, scaled = build_loaders(arrays, scaler, 5, int(cfg["training"]["batch_size"]))
    ceiling, ceiling_df = concept_ceiling(arrays, scaler, 5)
    model = make_model(cfg, 5, spec["membership_family"], int(spec["n_memberships"]), float(spec.get("temperature_init", 1.0)), float(arrays["y_train"].mean()))
    initialize_memberships(model, scaled["c_train"])
    history = train_oracle(model, cfg, train_loader, val_loader, seed)
    y, p, extras = evaluate_model(model, val_loader)
    direct_p = fit_prob(concept_features(scaled["c_train"]), arrays["y_train"].astype(int), concept_features(scaled["c_val"]))
    label = {"seed": seed, **spec}
    info = info_path(seed, spec["membership_family"], int(spec["n_memberships"]), extras, y, direct_p)
    faith = faithfulness(seed, model, val_loader, label)
    rem = ci_lower(
        faith[faith.intervention == "top1_removal"]["abs_probability_delta"].to_numpy(),
        faith[faith.intervention == "random_removal"]["abs_probability_delta"].to_numpy(),
    )
    ins = ci_lower(
        faith[faith.intervention == "top1_insertion"]["abs_probability_delta"].to_numpy(),
        faith[faith.intervention == "random_insertion"]["abs_probability_delta"].to_numpy(),
    )
    metrics = {
        **label,
        **binary_metrics(y, p),
        "oracle_ceiling_AUPRC": ceiling,
        "ceiling_ratio": float(average_precision_score(y, p) / max(ceiling, 1e-8)),
        "removal_margin_ci_lower": rem,
        "insertion_margin_ci_lower": ins,
        "alpha_sum_error": float(np.max(np.abs(extras["alpha"].sum(axis=1) - 1.0))),
        "beta_sum_error": float(np.max(np.abs(extras["beta"].sum(axis=1) - 1.0))),
        "membership_finite": bool(np.isfinite(extras["memberships"]).all()),
        "membership_saturation_fraction": float(np.mean((extras["memberships"] < 0.01) | (extras["memberships"] > 0.99))),
        "decision_weight_min": float(model.decision_head.weight.detach().cpu().min()),
        "decision_weight_max": float(model.decision_head.weight.detach().cpu().max()),
        "positive_decision_weights": bool(model.decision_head.positive_weights),
    }
    seed_dir = output / "runs" / f"iteration_{spec['iteration']:02d}" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(seed_dir / "training_log.csv", index=False)
    ceiling_df.assign(seed=seed, iteration=spec["iteration"]).to_csv(seed_dir / "concept_ceiling.csv", index=False)
    info.to_csv(seed_dir / "fan_information_path.csv", index=False)
    faith.to_csv(seed_dir / "faithfulness_results.csv", index=False)
    (seed_dir / "concept_scaler.json").write_text(json.dumps(scaler.to_json(), indent=2), encoding="utf-8")
    torch.save(model.state_dict(), seed_dir / "oracle_fan_checkpoint.pt")
    return {"metrics": metrics, "info": info, "faith": faith}


def registered_specs(cfg: dict) -> list[dict]:
    base = [
        ("gaussian", 3, "minmax_train"),
        ("bell", 3, "minmax_train"),
        ("gaussian", 5, "minmax_train"),
        ("bell", 5, "minmax_train"),
        ("sigmoid", 3, "minmax_train"),
        ("mixed", 3, "minmax_train"),
        ("gaussian", 3, "robust_train"),
        ("bell", 3, "robust_train"),
        ("gaussian", 5, "robust_train"),
        ("bell", 5, "robust_train"),
        ("sigmoid", 3, "robust_train"),
        ("mixed", 5, "robust_train"),
    ]
    specs = []
    for idx, (family, n_memberships, scaling) in enumerate(base[: int(cfg["fan"].get("max_validation_iterations", 12))], start=1):
        specs.append(
            {
                "iteration": idx,
                "membership_family": family,
                "n_memberships": n_memberships,
                "concept_scaling": scaling,
                "temperature_init": 1.0,
            }
        )
    return specs


def diagnose(iter_df: pd.DataFrame, info_df: pd.DataFrame) -> dict:
    best = iter_df.sort_values("AUPRC", ascending=False).iloc[0].to_dict()
    passed = iter_df.groupby("seed").apply(
        lambda x: bool(
            (x["ceiling_ratio"].iloc[0] >= 0.95)
            and (x["removal_margin_ci_lower"].iloc[0] > 0)
            and (x["insertion_margin_ci_lower"].iloc[0] > 0)
        ),
        include_groups=False,
    )
    info_mean = info_df.groupby("representation_stage")["AUPRC"].mean().sort_values(ascending=False)
    stages = info_mean.to_dict()
    if stages.get("q_additive_classifier", 0) > stages.get("memberships_additive_classifier", 0) + 0.05:
        loss_stage = "membership_basis"
    elif stages.get("memberships_additive_classifier", 0) > stages.get("fuzzy_values_additive_classifier", 0) + 0.05:
        loss_stage = "local_fuzzy_aggregation"
    elif stages.get("fuzzy_values_additive_classifier", 0) > stages.get("alpha_times_fuzzy_values", 0) + 0.05:
        loss_stage = "concept_alpha"
    elif stages.get("alpha_times_fuzzy_values", 0) > stages.get("signed_contributions", 0) + 0.05:
        loss_stage = "additive_decision_head"
    else:
        loss_stage = "no_single_large_information_drop"
    return {
        "oracle_pass_count": int(passed.sum()),
        "oracle_status": "FAN_VALIDATED" if int(passed.sum()) >= 2 else "FAN_VALIDATED_NEGATIVE",
        "best_config": {k: best[k] for k in ["membership_family", "n_memberships", "concept_scaling", "temperature_init"]},
        "best_oracle_auprc": float(best["AUPRC"]),
        "best_ceiling_ratio": float(best["ceiling_ratio"]),
        "information_loss_stage": loss_stage,
        "stage_mean_auprc": stages,
    }


def run_loop(cfg: dict, seeds: list[int], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    all_metrics, all_info, all_faith = [], [], []
    terminal = None
    for spec in registered_specs(cfg):
        iter_metrics, iter_info, iter_faith = [], [], []
        for seed in seeds:
            res = run_oracle_iteration(seed, cfg, spec, output)
            iter_metrics.append(res["metrics"])
            iter_info.append(res["info"])
            iter_faith.append(res["faith"])
        iter_df = pd.DataFrame(iter_metrics)
        info_df = pd.concat(iter_info, ignore_index=True)
        faith_df = pd.concat(iter_faith, ignore_index=True)
        iter_dir = output / "runs" / f"iteration_{spec['iteration']:02d}"
        iter_df.to_csv(iter_dir / "oracle_fan_results.csv", index=False)
        info_df.to_csv(iter_dir / "fan_information_path.csv", index=False)
        faith_df.to_csv(iter_dir / "faithfulness_results.csv", index=False)
        all_metrics.append(iter_df)
        all_info.append(info_df)
        all_faith.append(faith_df)
        diagnosis = diagnose(iter_df, info_df)
        (iter_dir / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
        next_config = {"next": registered_specs(cfg)[spec["iteration"]]} if spec["iteration"] < len(registered_specs(cfg)) else {"next": None}
        (iter_dir / "next_config.yaml").write_text(yaml.safe_dump(next_config, sort_keys=False), encoding="utf-8")
        if diagnosis["oracle_status"] == "FAN_VALIDATED":
            terminal = diagnosis
            break
        terminal = diagnosis
    metrics = pd.concat(all_metrics, ignore_index=True)
    info = pd.concat(all_info, ignore_index=True)
    faith = pd.concat(all_faith, ignore_index=True)
    metrics.to_csv(output / "iteration_comparison.csv", index=False)
    info.to_csv(output / "fan_information_path.csv", index=False)
    faith.to_csv(output / "faithfulness_results.csv", index=False)
    terminal = terminal or diagnose(metrics, info)
    (output / "diagnosis.json").write_text(json.dumps(terminal, indent=2), encoding="utf-8")
    (output / "next_config.yaml").write_text(yaml.safe_dump({"next": None, "reason": "terminal_or_max_iterations"}, sort_keys=False), encoding="utf-8")
    return terminal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run_loop(cfg, args.seeds, Path(args.output))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
