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

from scripts.medical._common import add_run_context_args
from src.med_circuitbench.metrics.circuit_f1 import layer_aware_circuit_f1
from src.med_circuitbench.sctc.transcoder import SparseClinicalTranscoder


def _json_clean(value):
    if isinstance(value, dict):
        return {k: _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if pd.isna(value) else float(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _dataset_root_from_manifest(manifest_path: Path, manifest: dict) -> Path:
    if manifest_path.parent.name in {"transformer", "sctc", "circuits"} or manifest_path.parent.name.endswith("_circuits"):
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
    setattr(model, "input_kind", ckpt.get("input_kind", "h_ffn"))
    model.eval()
    return model


def _node_activations(dataset_root: Path, edge_catalog: pd.DataFrame, method: str) -> dict[tuple[int, int], np.ndarray]:
    activation_dir = dataset_root / "activations"
    h_layers = _load_array(activation_dir / "h_ffn").astype(np.float32)
    a_layers = _load_array(activation_dir / "a_ffn").astype(np.float32)
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
            by_layer[layer] = _load_sctc(dataset_root / method / "checkpoints" / f"layer_{layer}.ckpt")
        source = a_layers[layer, val_idx] if getattr(by_layer[layer], "input_kind", "h_ffn") == "a_ffn" else h_layers[layer, val_idx]
        with torch.no_grad():
            z = by_layer[layer](torch.tensor(source, dtype=torch.float32))["z"].numpy()
        out[(layer, feature)] = z[:, :, feature]
    return out


def _eval_states(dataset_root: Path, eval_split: str) -> np.ndarray:
    episodes = pd.read_parquet(dataset_root.parent / "benchmark" / "episodes.parquet")
    splits = json.loads((dataset_root.parent / "benchmark" / "splits.json").read_text())
    selected_ids = np.asarray(splits["train"] + splits[eval_split], dtype=np.int64)
    split = _load_array(dataset_root / "activations" / "split")
    val_idx = np.where(split == 1)[0]
    states = np.stack([np.stack(v).astype(np.float32) for v in episodes.iloc[selected_ids]["states"]])
    return states[val_idx, :36, :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["sctc", "sae", "linear_probes", "random_directions", "single_features"])
    add_run_context_args(parser)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    dataset_root = _dataset_root_from_manifest(args.manifest, manifest)
    transformer_metrics = pd.read_csv(dataset_root / "transformer" / "model_metrics.csv").iloc[0].to_dict()
    true_graph = json.loads((dataset_root.parent / "benchmark" / "true_graph.json").read_text())
    eval_dir = dataset_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    state_count = sum(1 for src, targets in true_graph.items() for dst in targets if dst != "Y")
    for method in args.methods:
        edge_path = dataset_root / f"{method}_circuits" / "edge_catalog.parquet"
        circuit_path = dataset_root / f"{method}_circuits" / "circuit_catalog.json"
        if not edge_path.exists() and method == "sctc":
            edge_path = dataset_root / "sctc_circuits" / "edge_catalog.parquet"
            circuit_path = dataset_root / "sctc_circuits" / "circuit_catalog.json"
        if not edge_path.exists():
            summaries.append({"method": method, "status": "NO_GO_MISSING_METHOD_OUTPUT", "CircuitF1": np.nan})
            continue
        edge_catalog = pd.read_parquet(edge_path)
        circuit_catalog = json.loads(circuit_path.read_text()) if circuit_path.exists() else []
        accepted_edges = edge_catalog[edge_catalog["accepted"] == True].to_dict("records") if len(edge_catalog) else []  # noqa: E712
        states = _eval_states(dataset_root, args.split or "validation") if accepted_edges else np.zeros((0, 36, 5), dtype=np.float32)
        if accepted_edges and method != "single_features":
            node_acts = _node_activations(dataset_root, edge_catalog, method)
            circuit_f1, classified_edges, assignments = layer_aware_circuit_f1(node_acts, states, accepted_edges, true_graph)
        else:
            circuit_f1 = type(
                "CircuitF1",
                (),
                {
                    "tp": 0,
                    "fp": 0,
                    "fn": state_count,
                    "precision": 0.0,
                    "recall": 0.0,
                    "circuit_f1": np.nan if method == "single_features" else 0.0,
                    "fp_unmapped": 0,
                    "fp_duplicate": 0,
                    "fp_wrong": 0,
                },
            )()
            classified_edges = pd.DataFrame()
            assignments = pd.DataFrame()
        method_eval = eval_dir / method
        method_eval.mkdir(parents=True, exist_ok=True)
        classified_edges.to_parquet(method_eval / "edge_classification.parquet", index=False)
        assignments.to_parquet(method_eval / "node_state_assignment.parquet", index=False)
        c_f1_payload = {
            "TP": int(circuit_f1.tp),
            "FP": int(circuit_f1.fp),
            "FN": int(circuit_f1.fn),
            "precision": float(circuit_f1.precision),
            "recall": float(circuit_f1.recall),
            "CircuitF1": None if pd.isna(circuit_f1.circuit_f1) else float(circuit_f1.circuit_f1),
        }
        (method_eval / "circuit_f1.json").write_text(json.dumps(c_f1_payload, indent=2), encoding="utf-8")
        metric_records = circuit_catalog if circuit_catalog else edge_catalog.to_dict("records")

        def _mean_metric(name: str) -> float:
            vals = pd.to_numeric(pd.Series([record.get(name, np.nan) for record in metric_records]), errors="coerce").dropna()
            return float(vals.mean()) if len(vals) else np.nan

        summaries.append(
            {
                "dataset": manifest.get("dataset", dataset_root.name),
                "seed": int(manifest.get("seed", -1)),
                "method": method,
                "status": "PASS" if (not pd.isna(circuit_f1.circuit_f1) and float(circuit_f1.circuit_f1) >= 0.70) else "NO_GO_CIRCUIT_F1",
                "validation_auprc": float(transformer_metrics.get("validation_auprc", 0.0)),
                "validation_auroc": float(transformer_metrics.get("validation_auroc", 0.5)),
                "accepted_edges": int(edge_catalog["accepted"].sum()) if "accepted" in edge_catalog.columns else 0,
                "candidate_edges": int(len(edge_catalog)),
                "circuits": int(len(circuit_catalog)),
                "CIE_abs": _mean_metric("CIE_abs"),
                "IP_pearson": _mean_metric("IP_pearson"),
                "Completeness": _mean_metric("Completeness"),
                "OTE": _mean_metric("OTE"),
                "ErrorCoverageAt3": _mean_metric("ErrorCoverageAt3"),
                "TP": int(circuit_f1.tp),
                "FP": int(circuit_f1.fp),
                "FN": int(circuit_f1.fn),
                "precision": float(circuit_f1.precision),
                "recall": float(circuit_f1.recall),
                "CircuitF1": np.nan if pd.isna(circuit_f1.circuit_f1) else float(circuit_f1.circuit_f1),
            }
        )
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(eval_dir / "graph_metrics.csv", index=False)
    summary = {"methods": summaries, "status": "PASS" if (summary_frame["method"].eq("sctc") & (summary_frame["CircuitF1"] >= 0.70)).any() else "NO_GO"}
    (dataset_root / "evaluation_summary.json").write_text(json.dumps(_json_clean(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
