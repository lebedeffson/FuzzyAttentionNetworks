from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from .data import PreparedData, compute_delta_time
from .metrics import binary_metrics
from .training import evaluate_model, load_model_from_checkpoint, make_loader


ROBUSTNESS_ARMS = ["ConceptFAN-NoAlpha", "PureNoFuzzy"]


def _perturb(data: PreparedData, scenario: str, level: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = data.indices("test")
    v = data.v[indices].copy()
    m = data.m[indices].copy()
    d = data.d[indices].copy()
    observed = m > 0
    if scenario == "gaussian_noise":
        noise = rng.normal(0.0, level, size=v.shape).astype(np.float32)
        v[observed] += noise[observed]
    elif scenario == "outliers":
        selected = observed & (rng.random(v.shape) < level)
        high = rng.random(v.shape) >= 0.5
        v[selected & high] = 4.0
        v[selected & ~high] = -4.0
    elif scenario == "mcar":
        selected = observed & (rng.random(v.shape) < level)
        v[selected] = 0.0
        m[selected] = 0.0
        d = np.stack([compute_delta_time(patient, 48) for patient in m], axis=0)
    elif scenario == "block_missing":
        block_hours = int(level)
        variable_count = max(1, v.shape[-1] // 5)
        for patient in range(len(v)):
            variables = rng.choice(v.shape[-1], size=variable_count, replace=False)
            start = int(rng.integers(0, 48 - block_hours + 1))
            v[patient, start : start + block_hours, variables] = 0.0
            m[patient, start : start + block_hours, variables] = 0.0
            d[patient] = compute_delta_time(m[patient], 48)
    elif scenario == "sensor_bias":
        vital_names = ["HR", "MAP", "NIMAP", "RespRate", "Temp", "SaO2"]
        variable_index = {name: index for index, name in enumerate(data.variables)}
        positions = [variable_index[name] for name in vital_names if name in variable_index]
        for position in positions:
            v[:, -12:, position] += float(level)
    else:
        raise ValueError(scenario)
    return v, m, d


def _inputs(data: PreparedData, v: np.ndarray, m: np.ndarray, d: np.ndarray) -> np.ndarray:
    static = np.repeat(data.static[data.indices("test"), None, :], 48, axis=1)
    return np.concatenate([v, m, d, static], axis=-1).astype(np.float32)


def _explanation_metrics(clean: np.ndarray, perturbed: np.ndarray) -> dict[str, float]:
    spearman = []
    for left, right in zip(clean, perturbed):
        if np.std(left) < 1e-12 or np.std(right) < 1e-12:
            spearman.append(0.0)
        else:
            spearman.append(float(stats.spearmanr(left, right).statistic))
    return {
        "contribution_spearman_vs_clean": float(np.nanmean(spearman)),
        "top1_agreement_vs_clean": float(np.mean(np.argmax(np.abs(clean), axis=1) == np.argmax(np.abs(perturbed), axis=1))),
        "sign_agreement_vs_clean": float(np.mean(np.sign(clean) == np.sign(perturbed))),
        "mean_absolute_contribution_change": float(np.mean(np.abs(clean - perturbed))),
    }


def run_robustness(data: PreparedData, config: dict, runs_root: Path, output_path: Path, device: torch.device, batch_size: int) -> pd.DataFrame:
    rows: list[dict] = []
    scenario_levels = config["robustness"]
    test_indices = data.indices("test")
    for arm in ROBUSTNESS_ARMS:
        run_dirs = sorted(path for path in (runs_root / arm).glob("run_*") if (path / "checkpoint_best.pt").exists())
        if len(run_dirs) != 30:
            raise ValueError(f"{arm}: expected 30 checkpoints, found {len(run_dirs)}")
        for run_dir in run_dirs:
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            model, _ = load_model_from_checkpoint(run_dir / "checkpoint_best.pt", config, device)
            clean_logits = pd.read_parquet(run_dir / "logits_test.parquet")
            clean_contributions_frame = pd.read_parquet(run_dir / "contributions_test.parquet")
            clean_contributions = clean_contributions_frame[[f"contribution_{index}" for index in range(5)]].to_numpy()
            clean_probability = clean_logits["probability_raw"].to_numpy()
            clean_metrics = binary_metrics(data.y[test_indices], clean_probability, clean_logits["logit_raw"].to_numpy())
            rows.append(
                {
                    "model_arm": arm,
                    "run_id": run_dir.name,
                    "init_seed": manifest["init_seed"],
                    "data_order_seed": manifest["data_order_seed"],
                    "scenario": "clean",
                    "level": 0.0,
                    **clean_metrics,
                    "AUPRC_degradation": 0.0,
                    "AUROC_degradation": 0.0,
                    "concept_MAE": float("nan"),
                    "contribution_spearman_vs_clean": 1.0,
                    "top1_agreement_vs_clean": 1.0,
                    "sign_agreement_vs_clean": 1.0,
                    "mean_absolute_contribution_change": 0.0,
                }
            )
            for scenario, levels in scenario_levels.items():
                for level_position, level in enumerate(levels):
                    seed = int(manifest["init_seed"]) * 100_000 + int(manifest["data_order_seed"]) + level_position
                    v, m, d = _perturb(data, scenario, float(level), seed)
                    x = _inputs(data, v, m, d)
                    loader = make_loader(
                        x,
                        data.y[test_indices],
                        data.concepts[test_indices],
                        data.concept_mask[test_indices],
                        data.record_ids[test_indices],
                        np.arange(len(test_indices)),
                        batch_size,
                        False,
                        seed,
                    )
                    evaluation = evaluate_model(model, loader, device)
                    metrics = binary_metrics(evaluation.target, evaluation.probability, evaluation.logits)
                    target = data.concepts[test_indices]
                    target_mask = data.concept_mask[test_indices]
                    concept_mae = float(
                        (np.abs(evaluation.concept_trajectories - target) * target_mask).sum() / target_mask.sum().clip(min=1)
                    )
                    rows.append(
                        {
                            "model_arm": arm,
                            "run_id": run_dir.name,
                            "init_seed": manifest["init_seed"],
                            "data_order_seed": manifest["data_order_seed"],
                            "scenario": scenario,
                            "level": float(level),
                            **metrics,
                            "AUPRC_degradation": float(clean_metrics["AUPRC"] - metrics["AUPRC"]),
                            "AUROC_degradation": float(clean_metrics["AUROC"] - metrics["AUROC"]),
                            "concept_MAE": concept_mae,
                            **_explanation_metrics(clean_contributions, evaluation.contributions),
                        }
                    )
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False, compression="zstd")
    return frame


def aggregate_robustness(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["model_arm", "scenario", "level"], as_index=False).agg(
        runs=("run_id", "nunique"),
        AUPRC_mean=("AUPRC", "mean"),
        AUPRC_std=("AUPRC", "std"),
        AUPRC_degradation_mean=("AUPRC_degradation", "mean"),
        AUROC_mean=("AUROC", "mean"),
        AUROC_degradation_mean=("AUROC_degradation", "mean"),
        concept_MAE_mean=("concept_MAE", "mean"),
        contribution_spearman_mean=("contribution_spearman_vs_clean", "mean"),
        top1_agreement_mean=("top1_agreement_vs_clean", "mean"),
        sign_agreement_mean=("sign_agreement_vs_clean", "mean"),
    )
