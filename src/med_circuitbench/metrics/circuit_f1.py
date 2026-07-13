from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.med_circuitbench.metrics import safe_pearson

NodeId = Tuple[int, int]


@dataclass(frozen=True)
class LayerAwareCircuitF1:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    circuit_f1: float
    fp_unmapped: int
    fp_duplicate: int
    fp_wrong: int


def assign_nodes_to_states(
    node_activations: Mapping[NodeId, np.ndarray],
    states: np.ndarray,
    state_names: Sequence[str] = ("I", "R", "V", "O", "S"),
    threshold: float = 0.50,
) -> tuple[pd.DataFrame, dict[NodeId, str]]:
    nodes = list(node_activations)
    matrix = np.zeros((len(nodes), len(state_names)), dtype=float)
    signs = np.zeros_like(matrix)
    for i, node in enumerate(nodes):
        z = np.asarray(node_activations[node], dtype=float).reshape(-1)
        for k in range(len(state_names)):
            s = np.asarray(states[..., k], dtype=float).reshape(-1)
            corr = safe_pearson(z, s)
            matrix[i, k] = abs(corr)
            signs[i, k] = np.sign(corr)
    rows, cols = linear_sum_assignment(-matrix)
    assignment: dict[NodeId, str] = {}
    records = []
    accepted_pairs = set(zip(rows, cols))
    for i, node in enumerate(nodes):
        for k, state in enumerate(state_names):
            accepted = (i, k) in accepted_pairs and matrix[i, k] > threshold
            if accepted:
                assignment[node] = state
            records.append(
                {
                    "layer": node[0],
                    "feature_id": node[1],
                    "state": state,
                    "correlation": float(matrix[i, k]),
                    "sign_raw": float(signs[i, k]),
                    "accepted": bool(accepted),
                }
            )
    return pd.DataFrame(records), assignment


def layer_aware_circuit_f1(
    node_activations: Mapping[NodeId, np.ndarray],
    states: np.ndarray,
    recovered_edges: Iterable[Mapping[str, object]],
    true_graph: Mapping[str, Mapping[str, float]],
    edge_threshold: float = 0.50,
) -> tuple[LayerAwareCircuitF1, pd.DataFrame, pd.DataFrame]:
    assignment_frame, assignment = assign_nodes_to_states(node_activations, states)
    true_edges = {
        (src, dst): float(w)
        for src, targets in true_graph.items()
        for dst, w in targets.items()
        if dst != "Y" and abs(float(w)) >= edge_threshold
    }
    represented: set[tuple[str, str]] = set()
    tp = fp_unmapped = fp_duplicate = fp_wrong = 0
    edge_rows = []
    for edge in recovered_edges:
        src = (int(edge["source_layer"]), int(edge["source_feature"]))
        dst = (int(edge["target_layer"]), int(edge["target_feature"]))
        dr = float(edge.get("DR", edge.get("dr", 0.0)))
        src_state = assignment.get(src)
        dst_state = assignment.get(dst)
        status = "FP_unmapped"
        if src_state is None or dst_state is None:
            fp_unmapped += 1
        else:
            state_edge = (src_state, dst_state)
            if state_edge in true_edges and np.sign(dr) == np.sign(true_edges[state_edge]):
                if state_edge in represented:
                    fp_duplicate += 1
                    status = "FP_duplicate"
                else:
                    tp += 1
                    represented.add(state_edge)
                    status = "TP"
            else:
                fp_wrong += 1
                status = "FP_wrong"
        edge_rows.append({**dict(edge), "source_state": src_state, "target_state": dst_state, "status": status})
    fn = sum(1 for edge in true_edges if edge not in represented)
    fp = fp_unmapped + fp_duplicate + fp_wrong
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (
        LayerAwareCircuitF1(tp, fp, fn, precision, recall, f1, fp_unmapped, fp_duplicate, fp_wrong),
        pd.DataFrame(edge_rows),
        assignment_frame,
    )
