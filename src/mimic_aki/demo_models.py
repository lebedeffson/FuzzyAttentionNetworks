from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fan.attention import FuzzyTemporalAttentionEncoder
from fan.concept import MultiSetAdditiveTemporalConceptFANModel
from med_circuitbench.models.transformer import ClinicalTransformer, TransformerConfig


@dataclass(frozen=True)
class DemoModelConfig:
    input_dim: int
    sequence_length: int = 4
    d_model: int = 32
    heads: int = 4
    layers: int = 1
    d_ffn: int = 64
    n_concepts: int = 5
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    seed: int = 42


class TransformerCBM(nn.Module):
    def __init__(self, cfg: DemoModelConfig):
        super().__init__()
        self.backbone = ClinicalTransformer(
            TransformerConfig(
                input_dim=cfg.input_dim,
                layers=cfg.layers,
                d_model=cfg.d_model,
                heads=cfg.heads,
                d_ffn=cfg.d_ffn,
                dropout=0.1,
                sequence_length=cfg.sequence_length,
            )
        )
        self.concept_head = nn.Linear(cfg.d_model, cfg.n_concepts)
        self.task_head = nn.Linear(cfg.n_concepts, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.backbone(x, return_activations=True)
        h = out["capture_points"]["residual_post"][-1].mean(dim=1)
        concepts = torch.sigmoid(self.concept_head(h))
        logit = self.task_head(concepts).squeeze(-1)
        return {"logit": logit, "probability": torch.sigmoid(logit), "concepts": concepts}


class TransformerConceptFAN(nn.Module):
    def __init__(self, cfg: DemoModelConfig):
        super().__init__()
        self.model = MultiSetAdditiveTemporalConceptFANModel(
            input_dim=cfg.input_dim,
            sequence_length=cfg.sequence_length,
            latent_dim=cfg.d_model,
            n_concepts=cfg.n_concepts,
            n_memberships=3,
            oracle=False,
            encoder_layers=cfg.layers,
            encoder_heads=cfg.heads,
            encoder_ffn=cfg.d_ffn,
            alpha_mode="no_alpha",
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.model(x)
        return {
            "logit": out.logit,
            "probability": out.probability,
            "concepts": out.concept_summaries,
            "contributions": out.signed_decision_contributions,
        }


class FuzzyEncoderClassifier(nn.Module):
    def __init__(self, cfg: DemoModelConfig):
        super().__init__()
        self.encoder = FuzzyTemporalAttentionEncoder(
            input_dim=cfg.input_dim,
            sequence_length=cfg.sequence_length,
            d_model=cfg.d_model,
            n_heads=cfg.heads,
            n_layers=cfg.layers,
            d_ffn=cfg.d_ffn,
            dropout=0.1,
        )
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        logit = self.head(h.mean(dim=1)).squeeze(-1)
        return {"logit": logit, "probability": torch.sigmoid(logit), "latent_sequence": h}


class FuzzyEncoderConceptFAN(nn.Module):
    def __init__(self, cfg: DemoModelConfig):
        super().__init__()
        self.fuzzy_encoder = FuzzyTemporalAttentionEncoder(
            input_dim=cfg.input_dim,
            sequence_length=cfg.sequence_length,
            d_model=cfg.d_model,
            n_heads=cfg.heads,
            n_layers=cfg.layers,
            d_ffn=cfg.d_ffn,
            dropout=0.1,
        )
        self.concept_fan = MultiSetAdditiveTemporalConceptFANModel(
            input_dim=cfg.input_dim,
            sequence_length=cfg.sequence_length,
            latent_dim=cfg.d_model,
            n_concepts=cfg.n_concepts,
            n_memberships=3,
            oracle=False,
            encoder_layers=cfg.layers,
            encoder_heads=cfg.heads,
            encoder_ffn=cfg.d_ffn,
            alpha_mode="no_alpha",
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.fuzzy_encoder(x)
        trajectories = self.concept_fan.projector(h)
        summaries, temporal_weights = self.concept_fan.temporal_aggregator(trajectories, self.concept_fan.temporal_mode)
        out = self.concept_fan.forward_from_summaries(h, trajectories, summaries, temporal_weights)
        return {
            "logit": out.logit,
            "probability": out.probability,
            "concepts": out.concept_summaries,
            "contributions": out.signed_decision_contributions,
            "latent_sequence": h,
        }


def feature_table_to_tensors(feature_table, feature_columns: list[str], concept_columns: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = feature_table[feature_columns].astype("float32").to_numpy()
    concepts = feature_table[concept_columns].astype("float32").to_numpy()
    y = feature_table["label"].astype("float32").to_numpy()
    # Demo has one aggregate feature row per window. The repeated scaled input
    # exercises temporal model wiring only; it is not an ICU hourly trajectory.
    seq = np.stack(
        [
            values,
            values * 0.75,
            values * 0.50,
            values * 0.25,
        ],
        axis=1,
    ).astype("float32")
    return torch.from_numpy(seq), torch.from_numpy(y), torch.from_numpy(concepts)


def make_model(name: str, cfg: DemoModelConfig) -> nn.Module:
    if name == "standard_transformer":
        return ClinicalTransformer(
            TransformerConfig(
                input_dim=cfg.input_dim,
                layers=cfg.layers,
                d_model=cfg.d_model,
                heads=cfg.heads,
                d_ffn=cfg.d_ffn,
                dropout=0.1,
                sequence_length=cfg.sequence_length,
            )
        )
    if name == "standard_cbm":
        return TransformerCBM(cfg)
    if name == "standard_conceptfan_noalpha":
        return TransformerConceptFAN(cfg)
    if name == "fuzzy_encoder":
        return FuzzyEncoderClassifier(cfg)
    if name == "fuzzy_encoder_conceptfan_noalpha":
        return FuzzyEncoderConceptFAN(cfg)
    raise ValueError(f"unknown demo model: {name}")


def _forward(model: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    out = model(x)
    if isinstance(out, dict):
        return out
    return {"logit": out.logit, "probability": torch.sigmoid(out.logit)}


def train_torch_demo_model(
    name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    concepts: torch.Tensor,
    split: np.ndarray,
    cfg: DemoModelConfig,
) -> tuple[nn.Module, dict, np.ndarray]:
    torch.manual_seed(cfg.seed)
    model = make_model(name, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    train_idx = np.where(split == "train")[0]
    ds = TensorDataset(x[train_idx], y[train_idx], concepts[train_idx])
    loader = DataLoader(ds, batch_size=min(cfg.batch_size, len(ds)), shuffle=True)
    losses = []
    for _ in range(cfg.epochs):
        model.train()
        for xb, yb, cb in loader:
            opt.zero_grad(set_to_none=True)
            out = _forward(model, xb)
            loss = F.binary_cross_entropy_with_logits(out["logit"], yb)
            if "concepts" in out:
                loss = loss + 0.2 * F.huber_loss(out["concepts"], cb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
    model.eval()
    with torch.inference_mode():
        out = _forward(model, x)
        prob = out["probability"].detach().cpu().numpy()
    finite = bool(np.isfinite(prob).all() and np.isfinite(losses).all())
    reloaded = copy.deepcopy(model)
    reloaded.load_state_dict(model.state_dict())
    with torch.inference_mode():
        reload_prob = _forward(reloaded.eval(), x)["probability"].detach().cpu().numpy()
    metrics = {
        "model": name,
        "status": "PASS" if finite and np.allclose(prob, reload_prob, atol=1e-6) else "FAIL",
        "finite_loss": bool(np.isfinite(losses).all()),
        "finite_predictions": bool(np.isfinite(prob).all()),
        "checkpoint_reload": bool(np.allclose(prob, reload_prob, atol=1e-6)),
        "final_loss": float(losses[-1]) if losses else float("nan"),
    }
    return model, metrics, prob
