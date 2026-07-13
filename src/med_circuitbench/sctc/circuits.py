from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np

from src.med_circuitbench.metrics import benjamini_hochberg


@dataclass(frozen=True)
class EdgeCandidate:
    source_layer: int
    source_feature: int
    target_layer: int
    target_feature: int
    association: float
    dr: float
    p_value: float
    adjusted_p_value: float = 1.0
    accepted: bool = False


def association_score(z_i: np.ndarray, z_j: np.ndarray, decoder_direction: np.ndarray, encoder_direction: np.ndarray) -> float:
    corr = 0.0 if np.std(z_i) == 0 or np.std(z_j) == 0 else abs(float(np.corrcoef(z_i, z_j)[0, 1]))
    enc = encoder_direction / (np.linalg.norm(encoder_direction) + 1e-12)
    dec = decoder_direction / (np.linalg.norm(decoder_direction) + 1e-12)
    return float(max(corr, abs(np.dot(dec, enc))))


def empirical_p_value(dr_true: float, dr_random: Sequence[float]) -> float:
    random = np.asarray(dr_random, dtype=float)
    return float((1 + np.sum(np.abs(random) >= abs(dr_true))) / (len(random) + 1))


def accept_edges(edges: Sequence[EdgeCandidate], minimum_dr: float = 0.10, fdr_alpha: float = 0.05) -> List[EdgeCandidate]:
    adjusted = benjamini_hochberg([e.p_value for e in edges])
    out = []
    for edge, adj in zip(edges, adjusted):
        out.append(
            EdgeCandidate(
                edge.source_layer,
                edge.source_feature,
                edge.target_layer,
                edge.target_feature,
                edge.association,
                edge.dr,
                edge.p_value,
                float(adj),
                bool(adj < fdr_alpha and abs(edge.dr) >= minimum_dr),
            )
        )
    return out


def build_graph(edges: Iterable[EdgeCandidate]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for edge in edges:
        if not edge.accepted:
            continue
        src = (edge.source_layer, edge.source_feature)
        dst = (edge.target_layer, edge.target_feature)
        graph.add_edge(src, dst, DR=edge.dr, adjusted_p_value=edge.adjusted_p_value)
    return graph


def enumerate_layer_increasing_paths(graph: nx.DiGraph, min_edges: int = 2) -> List[List[Tuple[int, int]]]:
    paths: List[List[Tuple[int, int]]] = []
    for source in graph.nodes:
        for target in graph.nodes:
            if source == target:
                continue
            for path in nx.all_simple_paths(graph, source, target, cutoff=len(graph.nodes)):
                if len(path) - 1 < min_edges:
                    continue
                layers = [node[0] for node in path]
                if all(a < b for a, b in zip(layers, layers[1:])):
                    paths.append(path)
    return paths
