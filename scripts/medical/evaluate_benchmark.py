#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _dataset_root_from_manifest(manifest_path: Path, manifest: dict) -> Path:
    dataset = manifest.get("dataset")
    if dataset == "med_circuitbench":
        return Path("artifacts/medical/med_circuitbench")
    if dataset:
        return Path("artifacts/medical") / dataset
    if manifest_path.parent.name in {"transformer", "sctc", "circuits"}:
        return manifest_path.parent.parent
    return manifest_path.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    dataset_root = _dataset_root_from_manifest(args.manifest, manifest)
    transformer_metrics = pd.read_csv(dataset_root / "transformer" / "model_metrics.csv").iloc[0].to_dict()
    edge_catalog = pd.read_parquet(dataset_root / "circuits" / "edge_catalog.parquet")
    circuit_catalog = json.loads((dataset_root / "circuits" / "circuit_catalog.json").read_text())
    sctc_selection = pd.read_csv(dataset_root / "sctc" / "lambda_selection.csv")
    summary = {
        "dataset": manifest.get("dataset", dataset_root.name),
        "validation_auprc": float(transformer_metrics.get("validation_auprc", 0.0)),
        "validation_auroc": float(transformer_metrics.get("validation_auroc", 0.5)),
        "accepted_edges": int(edge_catalog["accepted"].sum()),
        "candidate_edges": int(len(edge_catalog)),
        "circuits": int(len(circuit_catalog)),
        "accepted_sctc_configs": int(sctc_selection["accepted"].sum()),
        "sctc_layers": int(sctc_selection.loc[sctc_selection["accepted"], "layer"].nunique()),
    }
    out = dataset_root / "evaluation_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
