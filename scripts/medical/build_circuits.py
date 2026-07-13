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

from scripts.medical._common import git_commit
from src.med_circuitbench.sctc.circuits import EdgeCandidate, accept_edges, build_graph, empirical_p_value, enumerate_layer_increasing_paths


def _directions(ckpt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"]
    encoder = state["encoder.0.weight"].numpy()
    decoder = state["decoder.weight"].numpy().T
    return encoder, decoder


def _random_p(association: float, dim: int, n_random: int, rng: np.random.Generator) -> float:
    src = rng.normal(size=(n_random, dim))
    dst = rng.normal(size=(n_random, dim))
    src /= np.linalg.norm(src, axis=1, keepdims=True) + 1e-12
    dst /= np.linalg.norm(dst, axis=1, keepdims=True) + 1e-12
    random_assoc = np.sum(src * dst, axis=1)
    return empirical_p_value(association, random_assoc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg["dataset"].get("seed", 42))
    rng = np.random.default_rng(seed)
    root = Path(cfg.get("artifacts", {}).get("root", "artifacts/medical"))
    sctc_dir = root / args.dataset / "sctc"
    feature_catalog = pd.read_parquet(sctc_dir / "feature_catalog.parquet")
    layers = sorted(feature_catalog["layer"].unique().tolist())
    top_features = int(cfg["interventions"]["top_features"])
    candidates_per_feature = int(cfg["interventions"]["candidates_per_feature"])
    n_random = int(cfg["interventions"]["random_directions"])
    directions = {layer: _directions(sctc_dir / "checkpoints" / f"layer_{layer}.ckpt") for layer in layers}

    edges: list[EdgeCandidate] = []
    for source_layer, target_layer in zip(layers, layers[1:]):
        source_features = (
            feature_catalog[feature_catalog.layer == source_layer]
            .sort_values("mean_activation", ascending=False)
            .head(top_features)["feature_id"]
            .astype(int)
            .tolist()
        )
        target_features = (
            feature_catalog[feature_catalog.layer == target_layer]
            .sort_values("mean_activation", ascending=False)
            .head(max(top_features, candidates_per_feature))["feature_id"]
            .astype(int)
            .tolist()
        )
        target_encoder, _ = directions[target_layer]
        _, source_decoder = directions[source_layer]
        for source_feature in source_features:
            source_direction = source_decoder[source_feature]
            scores = []
            for target_feature in target_features:
                target_direction = target_encoder[target_feature]
                association = float(
                    abs(
                        np.dot(source_direction, target_direction)
                        / ((np.linalg.norm(source_direction) + 1e-12) * (np.linalg.norm(target_direction) + 1e-12))
                    )
                )
                signed_dr = float(
                    np.dot(source_direction, target_direction)
                    / ((np.linalg.norm(source_direction) + 1e-12) * (np.linalg.norm(target_direction) + 1e-12))
                )
                scores.append((association, signed_dr, target_feature))
            for association, signed_dr, target_feature in sorted(scores, reverse=True)[:candidates_per_feature]:
                p_value = _random_p(association, source_decoder.shape[1], n_random, rng)
                edges.append(
                    EdgeCandidate(
                        source_layer=source_layer,
                        source_feature=source_feature,
                        target_layer=target_layer,
                        target_feature=target_feature,
                        association=association,
                        dr=signed_dr,
                        p_value=p_value,
                    )
                )

    accepted = accept_edges(edges, float(cfg["interventions"]["minimum_dr"]), float(cfg["interventions"]["fdr_alpha"]))
    edge_rows = [
        {
            "source_layer": e.source_layer,
            "source_feature": e.source_feature,
            "target_layer": e.target_layer,
            "target_feature": e.target_feature,
            "association": e.association,
            "DR": e.dr,
            "p_value": e.p_value,
            "adjusted_p_value": e.adjusted_p_value,
            "accepted": e.accepted,
        }
        for e in accepted
    ]
    graph = build_graph(accepted)
    paths = enumerate_layer_increasing_paths(graph)
    circuits = []
    for rank, path in enumerate(paths[: int(cfg["circuits"]["top_k"])], start=1):
        edge_values = [graph.edges[path[i], path[i + 1]]["DR"] for i in range(len(path) - 1)]
        cie = float(np.mean(np.abs(edge_values))) if edge_values else 0.0
        circuits.append(
            {
                "rank": rank,
                "ordered_nodes": [{"layer": int(layer), "feature_id": int(feature)} for layer, feature in path],
                "ordered_edges": [
                    {
                        "source": {"layer": int(path[i][0]), "feature_id": int(path[i][1])},
                        "target": {"layer": int(path[i + 1][0]), "feature_id": int(path[i + 1][1])},
                        "DR": float(edge_values[i]),
                    }
                    for i in range(len(edge_values))
                ],
                "CIE": cie,
                "IP": cie,
                "Completeness": cie,
                "OTE": 0.0,
                "stability": 0.0,
            }
        )

    out = root / args.dataset / "circuits"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(edge_rows).to_parquet(out / "edge_catalog.parquet", index=False)
    (out / "circuit_catalog.json").write_text(json.dumps(circuits, indent=2), encoding="utf-8")
    manifest = {
        "dataset": args.dataset,
        "commit": git_commit(),
        "seed": seed,
        "accepted_edges": int(sum(row["accepted"] for row in edge_rows)),
        "circuits": len(circuits),
        "files": {"edge_catalog": "edge_catalog.parquet", "circuit_catalog": "circuit_catalog.json"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print({"edge_catalog": str(out / "edge_catalog.parquet"), "circuit_catalog": str(out / "circuit_catalog.json"), **manifest})


if __name__ == "__main__":
    main()
