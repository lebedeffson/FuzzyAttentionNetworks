#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.manifest.read_text())
    out = Path("artifacts/medical/article/tables")
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for dataset, ds_cfg in cfg["datasets"].items():
        root = Path(ds_cfg["root"])
        summary_path = root / "evaluation_summary.json"
        if summary_path.exists():
            summary = pd.DataFrame([json.loads(summary_path.read_text())])
            summary.to_csv(out / f"{dataset}_summary.csv", index=False)
            summary.to_latex(out / f"{dataset}_summary.tex", index=False)
            written.append(f"{dataset}_summary")
        baseline_path = root / "baselines" / "baseline_metrics.parquet"
        if baseline_path.exists():
            baselines = pd.read_parquet(baseline_path)
            baselines.to_csv(out / f"{dataset}_baselines.csv", index=False)
            baselines.to_latex(out / f"{dataset}_baselines.tex", index=False)
            written.append(f"{dataset}_baselines")
        edge_path = root / "circuits" / "edge_catalog.parquet"
        if edge_path.exists():
            edges = pd.read_parquet(edge_path)
            edge_summary = (
                edges.groupby(["source_layer", "target_layer"], as_index=False)
                .agg(candidate_edges=("accepted", "size"), accepted_edges=("accepted", "sum"), mean_abs_dr=("DR", lambda s: float(s.abs().mean())))
            )
            edge_summary.to_csv(out / f"{dataset}_edge_summary.csv", index=False)
            edge_summary.to_latex(out / f"{dataset}_edge_summary.tex", index=False)
            written.append(f"{dataset}_edge_summary")
        circuit_path = root / "circuits" / "circuit_catalog.json"
        if circuit_path.exists():
            circuits = json.loads(circuit_path.read_text())
            circuit_rows = [
                {
                    "rank": c["rank"],
                    "nodes": " -> ".join(f"L{n['layer']}:F{n['feature_id']}" for n in c.get("nodes", c.get("ordered_nodes", []))),
                    "CIE_abs": c.get("CIE_abs", c.get("CIE", 0.0)),
                    "IP_pearson": c.get("IP_pearson", c.get("IP", 0.0)),
                    "Completeness": c["Completeness"],
                    "OTE": c["OTE"],
                }
                for c in circuits
            ]
            pd.DataFrame(circuit_rows).to_csv(out / f"{dataset}_circuits.csv", index=False)
            pd.DataFrame(circuit_rows).to_latex(out / f"{dataset}_circuits.tex", index=False)
            written.append(f"{dataset}_circuits")
    (out / "manifest.json").write_text(json.dumps({"source_manifest": str(args.manifest), "tables": written}, indent=2), encoding="utf-8")
    print({"tables": str(out), "written": written})


if __name__ == "__main__":
    main()
