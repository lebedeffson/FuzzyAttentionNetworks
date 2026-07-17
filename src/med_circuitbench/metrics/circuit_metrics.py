from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from src.med_circuitbench.metrics import safe_pearson


def cie(base_prob: np.ndarray, intervened_prob: np.ndarray) -> dict[str, float]:
    delta = np.asarray(base_prob, dtype=float) - np.asarray(intervened_prob, dtype=float)
    return {"CIE_abs": float(np.mean(np.abs(delta))), "CIE_signed": float(np.mean(delta))}


def intervention_predictability(chain_strength: np.ndarray, probability_effect: np.ndarray) -> dict[str, float]:
    strength = np.asarray(chain_strength, dtype=float)
    effect = np.asarray(probability_effect, dtype=float)
    pearson = safe_pearson(strength, np.abs(effect))
    if np.std(strength) == 0 or np.std(effect) == 0:
        spearman = float("nan")
    else:
        spearman = float(spearmanr(strength, np.abs(effect)).statistic)
    return {"IP_pearson": pearson, "IP_spearman": spearman}


def completeness(top_effect: float, all_effect: float) -> float:
    if all_effect < 1e-8:
        return float("nan")
    return float(np.clip(top_effect / (all_effect + 1e-8), 0.0, 1.0))


def off_target_effect(base: np.ndarray, intervened: np.ndarray, std: np.ndarray) -> float:
    base = np.asarray(base, dtype=float)
    intervened = np.asarray(intervened, dtype=float)
    std = np.asarray(std, dtype=float)
    return float(np.mean(np.abs(intervened - base) / (std + 1e-8)))


def error_coverage_at3(base_prob: np.ndarray, changed_prob: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    base_prob = np.asarray(base_prob, dtype=float)
    changed_prob = np.asarray(changed_prob, dtype=float)
    target = np.asarray(target, dtype=int)
    pred = (base_prob >= threshold).astype(int)
    fp = (pred == 1) & (target == 0)
    fn = (pred == 0) & (target == 1)
    fp_cov = (base_prob - changed_prob >= 0.05) & fp
    fn_cov = (changed_prob - base_prob >= 0.05) & fn
    total_errors = int(fp.sum() + fn.sum())
    return {
        "ErrorCoverageAt3": float((fp_cov.sum() + fn_cov.sum()) / total_errors) if total_errors else float("nan"),
        "ErrorCoverageAt3_FP": float(fp_cov.sum() / fp.sum()) if fp.sum() else float("nan"),
        "ErrorCoverageAt3_FN": float(fn_cov.sum() / fn.sum()) if fn.sum() else float("nan"),
    }
