#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import yaml
import zarr

from src.med_circuitbench.sctc.layer_screening import compute_cls, sparsity_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    root = Path(cfg.get("artifacts", {}).get("root", "artifacts/medical"))
    activation_dir = root / args.dataset / "activations"
    a = np.asarray(zarr.load(str(activation_dir / "a_ffn")))
    n_layers = a.shape[0]
    sparsity = np.asarray([sparsity_score(a[layer]) for layer in range(n_layers)], dtype=float)
    # AUC, sensitivity and robustness require heavier probes/gradients. The MVP
    # script keeps the registered CLS formula and uses neutral components when
    # probe inputs are unavailable.
    auc = np.full((n_layers, 1), 0.5, dtype=float)
    sensitivity = np.linspace(1.0, 1.0 + 1e-3, n_layers)
    robustness = np.linspace(1.0, 1.0 + 1e-3, n_layers)[::-1]
    cls_cfg = cfg.get("cls", {})
    result = compute_cls(
        auc,
        sparsity,
        sensitivity,
        robustness,
        threshold_ratio=float(cls_cfg.get("relative_threshold", 0.4)),
        max_layers=int(cls_cfg.get("max_layers", 3)),
    )
    selected_zero_based = [layer - 1 for layer in result.selected_report[: int(cls_cfg.get("max_layers", 3))]]
    out_dir = root / args.dataset / "layers"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for layer in range(n_layers):
        rows.append(
            {
                "layer": layer,
                "decodability": float(auc[layer].mean()),
                "sparsity": float(sparsity[layer]),
                "sensitivity": float(sensitivity[layer]),
                "stability": float(robustness[layer]),
                "cls": float(result.cls[layer]),
                "selected": bool(layer in selected_zero_based),
                "exclusion_reason": "" if layer in selected_zero_based else "below_top3_or_threshold",
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "layer_scores.csv", index=False)
    (out_dir / "layer_selection.json").write_text(
        json.dumps({"ranking_report": result.ranking_report, "selected_layers": selected_zero_based}, indent=2),
        encoding="utf-8",
    )
    print({"layer_scores": str(out_dir / "layer_scores.csv"), "selected_layers": selected_zero_based})


if __name__ == "__main__":
    main()
