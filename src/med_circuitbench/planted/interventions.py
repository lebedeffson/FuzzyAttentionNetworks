from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from med_circuitbench.planted.model import EDGES, STATE_NAMES


STATE_INDEX = {name: idx for idx, name in enumerate(STATE_NAMES)}
EDGE_WEIGHT = {(source, target): float(weight) for source, target, weight in EDGES}
DIRECT_EDGES = frozenset(EDGE_WEIGHT)
DESCENDANTS = {
    "I": frozenset({"R", "V", "O", "S"}),
    "R": frozenset({"V", "O", "S"}),
    "V": frozenset({"O", "S"}),
    "O": frozenset({"S"}),
    "S": frozenset(),
}
TOPOLOGICAL_ORDER = ("I", "R", "V", "O", "S")


@dataclass(frozen=True)
class CounterfactualResult:
    nodes: torch.Tensor
    changed: torch.Tensor
    mode: Literal["do", "direct", "total"]


def _as_value(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _set_node(nodes: torch.Tensor, node: str, value: torch.Tensor | float) -> torch.Tensor:
    out = nodes.clone()
    idx = STATE_INDEX[node]
    out[..., idx] = _as_value(value, nodes[..., idx])
    return out


def structural_equation(nodes: torch.Tensor, target: str) -> torch.Tensor:
    """Evaluate one planted structural equation from currently supplied parent values."""
    i = nodes[..., STATE_INDEX["I"]]
    r = nodes[..., STATE_INDEX["R"]]
    v = nodes[..., STATE_INDEX["V"]]
    o = nodes[..., STATE_INDEX["O"]]
    if target == "I":
        return i
    if target == "R":
        return EDGE_WEIGHT[("I", "R")] * i
    if target == "V":
        return EDGE_WEIGHT[("R", "V")] * r
    if target == "O":
        return EDGE_WEIGHT[("V", "O")] * v
    if target == "S":
        return EDGE_WEIGHT[("V", "S")] * v + EDGE_WEIGHT[("O", "S")] * o
    raise ValueError(f"unknown target node {target}")


def do_node_intervention(baseline_nodes: torch.Tensor, source: str, value: torch.Tensor | float) -> CounterfactualResult:
    """Pearl-style do(source=value): keep nondescendants fixed and recompute descendants only."""
    if source not in STATE_INDEX:
        raise ValueError(f"unknown source node {source}")
    out = _set_node(baseline_nodes, source, value)
    descendants = DESCENDANTS[source]
    for node in TOPOLOGICAL_ORDER:
        if node in descendants:
            out[..., STATE_INDEX[node]] = structural_equation(out, node)
    return CounterfactualResult(nodes=out, changed=out - baseline_nodes, mode="do")


def total_effect_counterfactual(baseline_nodes: torch.Tensor, source: str, value: torch.Tensor | float) -> CounterfactualResult:
    """Alias for do intervention when evaluating reachability over all descendants."""
    result = do_node_intervention(baseline_nodes, source, value)
    return CounterfactualResult(nodes=result.nodes, changed=result.changed, mode="total")


def direct_edge_counterfactual(baseline_nodes: torch.Tensor, source: str, target: str, value: torch.Tensor | float) -> CounterfactualResult:
    """Change source and recompute only the target equation when source is a direct parent.

    Non-edge pairs are kept as direct-effect negative controls: target remains at baseline even
    though the source column records the intervention.
    """
    if source not in STATE_INDEX or target not in STATE_INDEX:
        raise ValueError(f"unknown source/target pair {source}->{target}")
    out = _set_node(baseline_nodes, source, value)
    if (source, target) in DIRECT_EDGES:
        out[..., STATE_INDEX[target]] = structural_equation(out, target)
    return CounterfactualResult(nodes=out, changed=out - baseline_nodes, mode="direct")


def reverse_effect_counterfactual(baseline_nodes: torch.Tensor, source: str, target: str, value: torch.Tensor | float) -> CounterfactualResult:
    """Evaluate an upstream/reverse control by direct-edge semantics."""
    return direct_edge_counterfactual(baseline_nodes, source, target, value)


def standardized_effect(after: torch.Tensor, before: torch.Tensor, target: str, train_std: torch.Tensor | float) -> torch.Tensor:
    idx = STATE_INDEX[target]
    std = _as_value(train_std, before[..., idx]).clamp_min(1e-8)
    return (after[..., idx] - before[..., idx]) / std


def target_train_std(train_nodes: torch.Tensor, target: str) -> torch.Tensor:
    idx = STATE_INDEX[target]
    return train_nodes[..., idx].std().clamp_min(1e-8)


def descendant_pairs() -> set[tuple[str, str]]:
    return {(source, target) for source in STATE_NAMES for target in DESCENDANTS[source]}

