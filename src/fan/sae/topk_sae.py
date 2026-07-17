from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, n_features: int, top_k: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_features = int(n_features)
        self.top_k = int(top_k)
        self.encoder = nn.Linear(input_dim, n_features)
        self.decoder = nn.Linear(n_features, input_dim, bias=False)
        self.reconstruction_bias = nn.Parameter(torch.zeros(input_dim))
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        with torch.no_grad():
            self.decoder.weight.copy_(F.normalize(self.encoder.weight, dim=1).T)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.encoder(x))
        if self.top_k < z.shape[-1]:
            values, indices = torch.topk(z, self.top_k, dim=-1)
            sparse = torch.zeros_like(z)
            z = sparse.scatter(-1, indices, values)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.reconstruction_bias

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        return {"z": z, "reconstructed": self.decode(z)}

    def normalize_decoder_(self) -> None:
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))
