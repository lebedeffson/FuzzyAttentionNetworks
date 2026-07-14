from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import torch


STATE_NAMES = ["I", "R", "V", "O", "S"]
EDGES = [("I", "R", 0.95), ("R", "V", 0.90), ("V", "O", 0.85), ("V", "S", 0.80), ("O", "S", 0.75)]
NODE_LAYERS = {"I": 0, "R": 1, "V": 2, "O": 3, "S": 3}


@dataclass(frozen=True)
class PlantedForward:
    nodes: torch.Tensor
    layers: Dict[int, torch.Tensor]
    logit: torch.Tensor
    probability: torch.Tensor


class PlantedCircuitModel(torch.nn.Module):
    """Layered planted circuit with no residual or direct output bypass.

    The model takes an observed infection trajectory and deterministically
    computes downstream nodes through the registered graph. Hidden activations
    are node coefficients embedded in orthonormal directions.
    """

    def __init__(self, seed: int = 42, d_model: int = 128, output_scale: float = 4.0):
        super().__init__()
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.normal(size=(d_model, len(STATE_NAMES))))
        directions = torch.from_numpy(q.T.astype(np.float32))
        self.register_buffer("directions", directions)
        self.register_buffer("edge_weights", torch.tensor([w for _, _, w in EDGES], dtype=torch.float32))
        self.output_scale = float(output_scale)
        self.seed = int(seed)
        self.d_model = int(d_model)

    def compute_nodes_from_i(self, infection: torch.Tensor) -> torch.Tensor:
        i = infection
        r = self.edge_weights[0] * i
        v = self.edge_weights[1] * r
        o = self.edge_weights[2] * v
        s = self.edge_weights[3] * v + self.edge_weights[4] * o
        return torch.stack([i, r, v, o, s], dim=-1)

    def layers_from_nodes(self, nodes: torch.Tensor) -> Dict[int, torch.Tensor]:
        d = self.directions
        return {
            0: nodes[..., [0]] @ d[[0]],
            1: nodes[..., [1]] @ d[[1]],
            2: nodes[..., [2]] @ d[[2]],
            3: nodes[..., 3:5] @ d[3:5],
        }

    def forward_from_nodes(self, nodes: torch.Tensor) -> PlantedForward:
        layers = self.layers_from_nodes(nodes)
        logit = self.output_scale * nodes[..., 4]
        return PlantedForward(nodes=nodes, layers=layers, logit=logit, probability=torch.sigmoid(logit))

    def forward(self, true_states: torch.Tensor) -> PlantedForward:
        infection = true_states[..., 0]
        return self.forward_from_nodes(self.compute_nodes_from_i(infection))

    def recover_layer_nodes(self, layer: int, activation: torch.Tensor) -> Dict[str, torch.Tensor]:
        if layer == 0:
            return {"I": activation @ self.directions[0]}
        if layer == 1:
            return {"R": activation @ self.directions[1]}
        if layer == 2:
            return {"V": activation @ self.directions[2]}
        if layer == 3:
            return {"O": activation @ self.directions[3], "S": activation @ self.directions[4]}
        raise ValueError(f"unknown planted layer {layer}")

    def downstream_from_layer(self, layer: int, activation: torch.Tensor) -> PlantedForward:
        recovered = self.recover_layer_nodes(layer, activation)
        if layer == 0:
            nodes = self.compute_nodes_from_i(recovered["I"])
        elif layer == 1:
            r = recovered["R"]
            v = self.edge_weights[1] * r
            o = self.edge_weights[2] * v
            s = self.edge_weights[3] * v + self.edge_weights[4] * o
            i = r / self.edge_weights[0].clamp_min(1e-8)
            nodes = torch.stack([i, r, v, o, s], dim=-1)
        elif layer == 2:
            v = recovered["V"]
            o = self.edge_weights[2] * v
            s = self.edge_weights[3] * v + self.edge_weights[4] * o
            r = v / self.edge_weights[1].clamp_min(1e-8)
            i = r / self.edge_weights[0].clamp_min(1e-8)
            nodes = torch.stack([i, r, v, o, s], dim=-1)
        elif layer == 3:
            o = recovered["O"]
            s_direct = recovered["S"]
            v = o / self.edge_weights[2].clamp_min(1e-8)
            r = v / self.edge_weights[1].clamp_min(1e-8)
            i = r / self.edge_weights[0].clamp_min(1e-8)
            s = s_direct
            nodes = torch.stack([i, r, v, o, s], dim=-1)
        else:
            raise ValueError(f"unknown planted layer {layer}")
        return self.forward_from_nodes(nodes)

    def ablate_source_node(self, true_states: torch.Tensor, source: str) -> PlantedForward:
        nodes = self.forward(true_states).nodes.clone()
        idx = STATE_NAMES.index(source)
        nodes[..., idx] = 0.0
        if source == "I":
            nodes = self.compute_nodes_from_i(nodes[..., 0])
        elif source == "R":
            v = self.edge_weights[1] * nodes[..., 1]
            o = self.edge_weights[2] * v
            s = self.edge_weights[3] * v + self.edge_weights[4] * o
            nodes = torch.stack([nodes[..., 0], nodes[..., 1], v, o, s], dim=-1)
        elif source == "V":
            o = self.edge_weights[2] * nodes[..., 2]
            s = self.edge_weights[3] * nodes[..., 2] + self.edge_weights[4] * o
            nodes = torch.stack([nodes[..., 0], nodes[..., 1], nodes[..., 2], o, s], dim=-1)
        elif source == "O":
            s = self.edge_weights[3] * nodes[..., 2] + self.edge_weights[4] * nodes[..., 3]
            nodes = torch.stack([nodes[..., 0], nodes[..., 1], nodes[..., 2], nodes[..., 3], s], dim=-1)
        return self.forward_from_nodes(nodes)

    def spec(self) -> dict:
        return {
            "seed": self.seed,
            "d_model": self.d_model,
            "layers": {str(layer): [node for node, node_layer in NODE_LAYERS.items() if node_layer == layer] for layer in range(4)},
            "edges": [{"source": s, "target": t, "weight": w, "sign": float(np.sign(w))} for s, t, w in EDGES],
            "residual_bypass": False,
            "direct_input_to_output": False,
        }

    def bypass_audit(self, true_states: torch.Tensor) -> dict:
        base = self.forward(true_states)
        all_zero = torch.zeros_like(base.layers[0])
        bias_only = self.downstream_from_layer(0, all_zero)
        audit = {"all_paths_removed_probability_mae": float((bias_only.probability - 0.5).abs().mean().item())}
        for source in ["I", "R", "V", "O"]:
            ablated = self.ablate_source_node(true_states, source)
            for target in STATE_NAMES[STATE_NAMES.index(source) + 1 :]:
                delta = (base.nodes[..., STATE_NAMES.index(target)] - ablated.nodes[..., STATE_NAMES.index(target)]).abs().mean()
                audit[f"ablate_{source}_changes_{target}"] = float(delta.item())
        return audit
