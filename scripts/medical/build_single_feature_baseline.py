#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.medical._common import add_run_context_args, git_commit
from scripts.medical.build_circuits import _encode, _load_array, _load_sctc, _load_transformer
from src.med_circuitbench.metrics.circuit_metrics import cie, error_coverage_at3, intervention_predictability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_run_context_args(parser)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    root = Path(cfg.get("artifacts", {}).get("root", "artifacts/medical"))
    seed = int(cfg["dataset"].get("seed", 42))
    transformer = _load_transformer(root, args.dataset, args.device)
    activation_dir = root / args.dataset / "activations"
    h_layers = _load_array(activation_dir / "h_ffn").astype(np.float32)
    a_layers = _load_array(activation_dir / "a_ffn").astype(np.float32)
    x = _load_array(activation_dir / "x").astype(np.float32)
    split = _load_array(activation_dir / "split")
    y = _load_array(activation_dir / "target").astype(np.int64)
    base_prob = _load_array(activation_dir / "probability").astype(np.float32)
    val_idx = np.where(split == 1)[0]
    catalog = pd.read_parquet(root / args.dataset / "sctc" / "feature_catalog.parquet")
    if "eligible" in catalog.columns:
        catalog = catalog[catalog["eligible"] == True]  # noqa: E712
    rows = []
    for layer in sorted(catalog["layer"].unique()):
        model = _load_sctc(root / args.dataset / "sctc" / "checkpoints" / f"layer_{int(layer)}.ckpt", args.device)
        z = _encode(model, h_layers[int(layer), val_idx], args.device)
        decoder = model.decoder.weight.detach().cpu().numpy().T
        for feature in catalog[catalog.layer == layer].sort_values("mean_activation", ascending=False).head(int(cfg["interventions"]["top_features"])).feature_id.astype(int):
            strength = z[:, :, feature].mean(axis=1)
            order = np.argsort(-strength)[: min(int(cfg["interventions"]["high_activation_windows"]), len(val_idx))]
            selected = val_idx[order]
            local_z = z[order, :, feature]
            replacement = a_layers[int(layer), selected].copy()
            replacement -= local_z[:, :, None] * decoder[feature]
            with torch.no_grad():
                out = transformer(
                    torch.tensor(x[selected], device=args.device, dtype=torch.float32),
                    replacements={int(layer): torch.tensor(replacement, device=args.device, dtype=torch.float32)},
                )
            int_prob = out["probability"].detach().cpu().numpy()
            c = cie(base_prob[selected], int_prob)
            ip = intervention_predictability(strength[order], int_prob - base_prob[selected])
            rows.append(
                {
                    "method": "single_features",
                    "seed": seed,
                    "source_layer": int(layer),
                    "source_feature": int(feature),
                    "target_layer": int(layer),
                    "target_feature": int(feature),
                    "association_corr": np.nan,
                    "association_geometry": np.nan,
                    "DR_ablation": np.nan,
                    "DR_push": np.nan,
                    "p_value": np.nan,
                    "adjusted_p_value": np.nan,
                    "accepted": False,
                    "n_windows": int(len(selected)),
                    "n_random_directions": 0,
                    "runtime_seconds": 0.0,
                    "CIE_abs": c["CIE_abs"],
                    "CIE_signed": c["CIE_signed"],
                    "IP_pearson": ip["IP_pearson"],
                    "IP_spearman": ip["IP_spearman"],
                    "Completeness": np.nan,
                    "OTE": np.nan,
                    **error_coverage_at3(base_prob[selected], int_prob, y[selected]),
                }
            )
    out_dir = root / args.dataset / "single_features_circuits"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_dir / "edge_catalog.parquet", index=False)
    (out_dir / "circuit_catalog.json").write_text("[]\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps({"dataset": args.dataset, "method": "single_features", "seed": seed, "commit": git_commit()}, indent=2), encoding="utf-8")
    print({"method": "single_features", "features": len(rows), "out": str(out_dir)})


if __name__ == "__main__":
    main()
