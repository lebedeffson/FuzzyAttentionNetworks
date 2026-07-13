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
import zarr

from src.med_circuitbench.metrics.circuit_f1 import layer_aware_circuit_f1
from src.med_circuitbench.sctc.transcoder import SparseClinicalTranscoder


def _dataset_root_from_manifest(manifest_path: Path, manifest: dict) -> Path:
    if manifest_path.parent.name in {"transformer", "sctc", "circuits"}:
        return manifest_path.parent.parent
    dataset = manifest.get("dataset")
    if dataset:
        candidate = manifest_path.parent / dataset
        if candidate.exists():
            return candidate
        return Path("artifacts/medical") / dataset
    return manifest_path.parent


def _load_array(path: Path) -> np.ndarray:
    return np.asarray(zarr.load(str(path)))


def _load_sctc(path: Path) -> SparseClinicalTranscoder:
    ckpt = torch.load(path, map_location="cpu")
    model = SparseClinicalTranscoder(d_model=int(ckpt["d_model"]), n_features=int(ckpt["n_features"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _node_activations(dataset_root: Path, edge_catalog: pd.DataFrame) -> dict[tuple[int, int], np.ndarray]:
    activation_dir = dataset_root / "activations"
    h_layers = _load_array(activation_dir / "h_ffn").astype(np.float32)
    split = _load_array(activation_dir / "split")
    val_idx = np.where(split == 1)[0]
    nodes = set()
    accepted = edge_catalog[edge_catalog["accepted"] == True] if "accepted" in edge_catalog.columns else edge_catalog  # noqa: E712
    for row in accepted.to_dict("records"):
        nodes.add((int(row["source_layer"]), int(row["source_feature"])))
        nodes.add((int(row["target_layer"]), int(row["target_feature"])))
    by_layer: dict[int, SparseClinicalTranscoder] = {}
    out: dict[tuple[int, int], np.ndarray] = {}
    for layer, feature in nodes:
        if layer not in by_layer:
            by_layer[layer] = _load_sctc(dataset_root / "sctc" / "checkpoints" / f"layer_{layer}.ckpt")
        with torch.no_grad():
            z = by_layer[layer](torch.tensor(h_layers[layer, val_idx], dtype=torch.float32))["z"].numpy()
        out[(layer, feature)] = z[:, :, feature]
    return out


def _validation_states(dataset_root: Path) -> np.ndarray:
    episodes = pd.read_parquet(dataset_root.parent / "benchmark" / "episodes.parquet")
    splits = json.loads((dataset_root.parent / "benchmark" / "splits.json").read_text())
    selected_ids = np.asarray(splits["train"] + splits["validation"], dtype=np.int64)
    split = _load_array(dataset_root / "activations" / "split")
    val_idx = np.where(split == 1)[0]
    states = np.stack([np.asarray(v, dtype=np.float32) for v in episodes.iloc[selected_ids]["states"]])
    return states[val_idx, :36, :]


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

    accepted_edges = edge_catalog[edge_catalog["accepted"] == True].to_dict("records") if len(edge_catalog) else []  # noqa: E712
    node_acts = _node_activations(dataset_root, edge_catalog) if accepted_edges else {}
    true_graph = json.loads((dataset_root.parent / "benchmark" / "true_graph.json").read_text())
    states = _validation_states(dataset_root) if accepted_edges else np.zeros((0, 36, 5), dtype=np.float32)
    if accepted_edges:
        circuit_f1, classified_edges, assignments = layer_aware_circuit_f1(node_acts, states, accepted_edges, true_graph)
    else:
        circuit_f1 = type(
            "CircuitF1",
            (),
            {
                "tp": 0,
                "fp": 0,
                "fn": sum(1 for src, targets in true_graph.items() for dst in targets if dst != "Y"),
                "precision": 0.0,
                "recall": 0.0,
                "circuit_f1": 0.0,
                "fp_unmapped": 0,
                "fp_duplicate": 0,
                "fp_wrong": 0,
            },
        )()
        classified_edges = pd.DataFrame()
        assignments = pd.DataFrame()

    eval_dir = dataset_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    classified_edges.to_parquet(eval_dir / "edge_classification.parquet", index=False)
    assignments.to_parquet(eval_dir / "node_state_assignment.parquet", index=False)
    c_f1_payload = {
        "TP": int(circuit_f1.tp),
        "FP": int(circuit_f1.fp),
        "FN": int(circuit_f1.fn),
        "Precision": float(circuit_f1.precision),
        "Recall": float(circuit_f1.recall),
        "CircuitF1": float(circuit_f1.circuit_f1),
        "FP_unmapped": int(circuit_f1.fp_unmapped),
        "FP_duplicate": int(circuit_f1.fp_duplicate),
        "FP_wrong": int(circuit_f1.fp_wrong),
    }
    (eval_dir / "circuit_f1.json").write_text(json.dumps(c_f1_payload, indent=2), encoding="utf-8")

    summary = {
        "dataset": manifest.get("dataset", dataset_root.name),
        "seed": int(manifest.get("seed", -1)),
        "status": "PASS" if float(circuit_f1.circuit_f1) >= 0.70 else "NO_GO_CIRCUIT_F1",
        "validation_auprc": float(transformer_metrics.get("validation_auprc", 0.0)),
        "validation_auroc": float(transformer_metrics.get("validation_auroc", 0.5)),
        "accepted_edges": int(edge_catalog["accepted"].sum()) if "accepted" in edge_catalog.columns else 0,
        "candidate_edges": int(len(edge_catalog)),
        "circuits": int(len(circuit_catalog)),
        "accepted_sctc_configs": int(sctc_selection["accepted"].sum()),
        "sctc_layers": int(sctc_selection.loc[sctc_selection["accepted"], "layer"].nunique()) if len(sctc_selection) else 0,
        **c_f1_payload,
    }
    pd.DataFrame([summary]).to_csv(eval_dir / "graph_metrics.csv", index=False)
    (dataset_root / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
