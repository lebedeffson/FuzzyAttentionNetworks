from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


WEIGHTS = {"A": 0.35, "S": 0.20, "G": 0.30, "R": 0.15}


@dataclass(frozen=True)
class CLSResult:
    cls: np.ndarray
    ranking_report: List[int]
    selected_report: List[int]


def _positive_decode_score(auc: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, 2.0 * auc - 1.0).mean(axis=1)


def _minmax(x: np.ndarray) -> tuple[np.ndarray | None, bool]:
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.min(x)), float(np.max(x))
    if abs(hi - lo) < 1e-12:
        return None, False
    return (x - lo) / (hi - lo), True


def compute_cls(
    auc_by_layer: np.ndarray,
    sparsity: np.ndarray,
    sensitivity: np.ndarray,
    robustness: np.ndarray,
    threshold_ratio: float = 0.4,
    max_layers: int = 3,
) -> CLSResult:
    raw = {
        "A": _positive_decode_score(np.asarray(auc_by_layer, dtype=float)),
        "S": np.asarray(sparsity, dtype=float),
        "G": np.asarray(sensitivity, dtype=float),
        "R": np.asarray(robustness, dtype=float),
    }
    normalized: Dict[str, np.ndarray] = {}
    active_weights: Dict[str, float] = {}
    for key, values in raw.items():
        norm, active = _minmax(values)
        if active:
            normalized[key] = norm
            active_weights[key] = WEIGHTS[key]
    total = sum(active_weights.values())
    if total == 0:
        cls = np.zeros(len(sparsity), dtype=float)
    else:
        cls = sum(normalized[k] * (w / total) for k, w in active_weights.items())
    ranking = list(np.argsort(-cls))
    selected = [idx for idx in ranking if cls[idx] >= threshold_ratio * float(cls.max())][:max_layers]
    return CLSResult(
        cls=cls,
        ranking_report=[idx + 1 for idx in ranking],
        selected_report=[idx + 1 for idx in selected],
    )


def sparsity_score(activations: np.ndarray) -> float:
    threshold = 0.01 * np.quantile(np.abs(activations), 0.99)
    return float(1.0 - np.mean(np.abs(activations) > threshold))


def trim_gradient_norms(norms: Iterable[float]) -> float:
    values = np.asarray(list(norms), dtype=float)
    if values.size == 0:
        return 0.0
    cutoff = np.quantile(values, 0.99)
    return float(np.median(values[values <= cutoff]))
