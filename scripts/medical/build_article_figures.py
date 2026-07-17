#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import yaml


def _draw_graph(graph: nx.DiGraph, path: Path, title: str) -> None:
    plt.figure(figsize=(7, 4))
    pos = nx.spring_layout(graph, seed=42)
    widths = [1.0 + 3.0 * abs(float(graph.edges[e].get("weight", graph.edges[e].get("DR", 0.1)))) for e in graph.edges]
    colors = ["#1f77b4" if float(graph.edges[e].get("weight", graph.edges[e].get("DR", 0.1))) >= 0 else "#d62728" for e in graph.edges]
    nx.draw_networkx_nodes(graph, pos, node_color="#f2f2f2", edgecolors="#333333", node_size=1200)
    nx.draw_networkx_labels(graph, pos, font_size=9)
    nx.draw_networkx_edges(graph, pos, width=widths, edge_color=colors, arrows=True, arrowsize=16)
    labels = {e: f"{float(graph.edges[e].get('weight', graph.edges[e].get('DR', 0.0))):.2f}" for e in graph.edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.manifest.read_text())
    out = Path("artifacts/medical/article/figures")
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for dataset, ds_cfg in cfg["datasets"].items():
        root = Path(ds_cfg["root"])
        benchmark_root = Path(ds_cfg.get("benchmark_root", "artifacts/medical/benchmark"))
        true_graph_path = benchmark_root / "true_graph.json"
        if true_graph_path.exists():
            data = json.loads(true_graph_path.read_text())
            graph = nx.DiGraph()
            for src, targets in data.items():
                for dst, weight in targets.items():
                    graph.add_edge(src, dst, weight=float(weight))
            fig = out / f"{dataset}_true_graph.png"
            _draw_graph(graph, fig, "Med-CircuitBench true graph")
            written.append(str(fig))
        edge_path = root / "circuits" / "edge_catalog.parquet"
        if edge_path.exists():
            edges = pd.read_parquet(edge_path)
            graph = nx.DiGraph()
            for row in edges[edges["accepted"]].itertuples(index=False):
                graph.add_edge(f"L{row.source_layer}:F{row.source_feature}", f"L{row.target_layer}:F{row.target_feature}", DR=float(row.DR))
            fig = out / f"{dataset}_recovered_graph.png"
            _draw_graph(graph, fig, "Recovered SCTC graph")
            written.append(str(fig))
        transformer_metrics = root / "transformer" / "training_history.csv"
        if transformer_metrics.exists():
            hist = pd.read_csv(transformer_metrics)
            plt.figure(figsize=(6, 3.5))
            plt.plot(hist["epoch"], hist["validation_auprc"], marker="o", label="Validation AUPRC")
            plt.plot(hist["epoch"], hist["validation_auroc"], marker="s", label="Validation AUROC")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.ylim(0, 1.05)
            plt.legend()
            plt.tight_layout()
            fig = out / f"{dataset}_training_metrics.png"
            plt.savefig(fig, dpi=200)
            plt.close()
            written.append(str(fig))
    (out / "manifest.json").write_text(json.dumps({"source_manifest": str(args.manifest), "figures": written}, indent=2), encoding="utf-8")
    print({"figures": str(out), "written": written})


if __name__ == "__main__":
    main()
