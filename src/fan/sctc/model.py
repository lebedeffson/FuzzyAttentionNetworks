from __future__ import annotations

import torch
from torch import nn


class SparseTranscoder(nn.Module):
    def __init__(self, d_model: int = 128, n_features: int = 512, top_k: int = 24):
        super().__init__()
        self.top_k = int(top_k)
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model, bias=False)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.constant_(self.encoder.bias, 0.05)
        self.normalize_decoder_()

    def forward(self, activation: torch.Tensor) -> dict[str, torch.Tensor]:
        z = torch.relu(self.encoder(activation))
        if self.top_k > 0 and self.top_k < z.shape[-1]:
            values, indices = torch.topk(z, k=self.top_k, dim=-1)
            sparse_z = torch.zeros_like(z)
            z = sparse_z.scatter(-1, indices, values)
        reconstructed = self.decoder(z)
        return {"z": z, "reconstructed": reconstructed}

    def normalize_decoder_(self) -> None:
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0).clamp_min(1e-8)
            self.decoder.weight.div_(norms.view(1, -1))
            self.encoder.weight.mul_(norms.view(-1, 1))
            self.encoder.bias.mul_(norms)
