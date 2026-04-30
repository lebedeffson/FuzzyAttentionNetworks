#!/usr/bin/env python3
"""Build a compact LaTeX table from FAN membership ablation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def cell(row, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.4f} $\\pm$ {row[f'{metric}_ci95']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/icp_membership_ablation"))
    parser.add_argument("--out-tex", type=Path, default=Path("paper/tables/membership_sensitivity.tex"))
    parser.add_argument("--out-csv", type=Path, default=Path("paper/results/membership_sensitivity.csv"))
    args = parser.parse_args()

    families = ["gaussian", "bell", "sigmoid", "mixed"]
    records = []
    for family in families:
        record = {"membership": family}
        for dataset in ["swat", "fd001"]:
            path = args.root / f"{dataset}_{family}" / f"{dataset}_summary.csv"
            row = pd.read_csv(path).iloc[0]
            record[f"{dataset}_f1_mean"] = row["f1_mean"]
            record[f"{dataset}_f1_ci95"] = row["f1_ci95"]
            record[f"{dataset}_auc_mean"] = row["roc_auc_mean"]
            record[f"{dataset}_auc_ci95"] = row["roc_auc_ci95"]
        records.append(record)

    labels = {
        "gaussian": "Gaussian",
        "bell": "Bell",
        "sigmoid": "Sigmoid",
        "mixed": "Mixed",
    }
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Membership & SWaT F1 & SWaT ROC-AUC & FD001 F1 & FD001 ROC-AUC \\\\",
        "\\midrule",
    ]
    for record in records:
        lines.append(
            f"{labels[record['membership']]} & "
            f"{cell(record, 'swat_f1')} & {cell(record, 'swat_auc')} & "
            f"{cell(record, 'fd001_f1')} & {cell(record, 'fd001_auc')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])

    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(records).to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
