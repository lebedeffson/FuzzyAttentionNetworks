"""Metrics for medical circuit evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr


@dataclass(frozen=True)
class CircuitF1Result:
    precision: float
    recall: float
    circuit_f1: float
    tp: int
    fp: int
    fn: int
    assignment: Dict[int, str]


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(pearsonr(x, y).statistic)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty_like(ranked)
    n = len(p)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, ranked[i] * n / (i + 1))
        adjusted[i] = running
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def assign_features_to_states(
    feature_activations: np.ndarray,
    states: np.ndarray,
    state_names: Sequence[str],
    threshold: float = 0.50,
) -> Tuple[np.ndarray, Dict[int, str]]:
    matrix = np.zeros((feature_activations.shape[1], states.shape[1]), dtype=float)
    for i in range(matrix.shape[0]):
        for k in range(matrix.shape[1]):
            matrix[i, k] = abs(safe_pearson(feature_activations[:, i], states[:, k]))
    rows, cols = linear_sum_assignment(-matrix)
    assignment: Dict[int, str] = {}
    for r, c in zip(rows, cols):
        if matrix[r, c] > threshold:
            assignment[int(r)] = state_names[int(c)]
    return matrix, assignment


def circuit_f1(
    feature_activations: np.ndarray,
    states: np.ndarray,
    recovered_edges: Iterable[Mapping[str, object]],
    true_graph: Mapping[str, Mapping[str, float]],
    state_names: Sequence[str] = ("I", "R", "V", "O", "S"),
    assignment_threshold: float = 0.50,
    edge_threshold: float = 0.50,
) -> CircuitF1Result:
    _, assignment = assign_features_to_states(feature_activations, states, state_names, assignment_threshold)
    true_edges = {
        (src, dst): float(w)
        for src, targets in true_graph.items()
        for dst, w in targets.items()
        if dst != "Y" and abs(float(w)) >= edge_threshold
    }
    found = []
    tp = fp = 0
    for edge in recovered_edges:
        src_feature = int(edge["source_feature"])
        dst_feature = int(edge["target_feature"])
        dr = float(edge.get("DR", edge.get("dr", 0.0)))
        src_state = assignment.get(src_feature)
        dst_state = assignment.get(dst_feature)
        if src_state is None or dst_state is None:
            fp += 1
            continue
        key = (src_state, dst_state)
        found.append(key)
        if key in true_edges and np.sign(dr) == np.sign(true_edges[key]):
            tp += 1
        else:
            fp += 1
    fn = sum(1 for key in true_edges if key not in found)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return CircuitF1Result(precision, recall, f1, tp, fp, fn, assignment)


def intervention_predictability(strength: Sequence[float], effect: Sequence[float]) -> float:
    return safe_pearson(np.asarray(strength, dtype=float), np.asarray(effect, dtype=float))
