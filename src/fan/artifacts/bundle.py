from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from fan.concept import MultiSetAdditiveTemporalConceptFANModel
from fan.evaluation.predictive import binary_classification_metrics


@dataclass(frozen=True)
class ModelBundle:
    root: Path
    manifest: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return cls(root=root, manifest=manifest)

    @property
    def checkpoint_path(self) -> Path:
        return self.root / str(self.manifest["checkpoint"])


class FANBundle(ModelBundle):
    def build_model(self, map_location: str | torch.device = "cpu") -> MultiSetAdditiveTemporalConceptFANModel:
        cfg = self.manifest["model_config"]
        model = MultiSetAdditiveTemporalConceptFANModel(
            input_dim=int(cfg["input_dim"]),
            sequence_length=int(cfg["sequence_length"]),
            latent_dim=int(cfg["latent_dim"]),
            n_concepts=int(cfg.get("n_concepts", 5)),
            n_memberships=int(cfg.get("n_memberships", 3)),
            membership=str(cfg.get("membership", "gaussian")),
            oracle=False,
            temporal_mode=str(cfg.get("temporal_mode", "attention")),
            dropout=float(cfg.get("dropout", 0.1)),
            encoder_layers=int(cfg.get("encoder_layers", 4)),
            encoder_heads=int(cfg.get("encoder_heads", 4)),
            encoder_ffn=int(cfg.get("encoder_ffn", 512)),
            alpha_mode=str(cfg.get("alpha_mode", "no_alpha")),
            positive_decision_weights=bool(cfg.get("positive_decision_weights", False)),
        )
        state = torch.load(self.checkpoint_path, map_location=map_location)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()
        return model


@dataclass(frozen=True)
class AuditConfig:
    predictive: bool = True
    concepts: bool = True
    faithfulness: bool = True
    stability: bool = True
    sparse_fidelity: bool = True
    mechanistic_recovery: bool = False


@dataclass(frozen=True)
class AuditReport:
    metrics: dict[str, Any]
    limitations: list[str]

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"metrics": self.metrics, "limitations": self.limitations}, indent=2), encoding="utf-8")


class AuditRunner:
    def __init__(self, config: AuditConfig):
        self.config = config

    def run(self, bundle: ModelBundle, dataset: pd.DataFrame | None = None) -> AuditReport:
        metrics: dict[str, Any] = {
            "bundle_type": bundle.manifest.get("bundle_type"),
            "model_name": bundle.manifest.get("model_name"),
            "seed": bundle.manifest.get("seed"),
            "mechanistic_recovery_requested": self.config.mechanistic_recovery,
        }
        limitations: list[str] = []
        saved = bundle.manifest.get("saved_metrics", {})
        if self.config.predictive:
            metrics.update({f"saved_{k}": v for k, v in saved.items() if k in {"AUROC", "AUPRC", "F1", "Brier", "ECE"}})
        if dataset is not None and {"target", "probability"}.issubset(dataset.columns):
            metrics.update(binary_classification_metrics(dataset["target"].to_numpy(), dataset["probability"].to_numpy()))
        if not self.config.mechanistic_recovery:
            limitations.append("mechanistic_recovery_disabled_by_default_requires_known_mechanism_benchmark")
        return AuditReport(metrics=metrics, limitations=limitations)


def save_prediction_pairs(path: str | Path, target: np.ndarray, probability: np.ndarray) -> None:
    pd.DataFrame({"target": target.astype(int), "probability": probability.astype(float)}).to_parquet(path, index=False)
