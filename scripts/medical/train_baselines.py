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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.medical._common import git_commit


def _metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    pred = (prob >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return {
        "auprc": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "auroc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.5,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _load_benchmark(root: Path):
    df = pd.read_parquet(root / "benchmark" / "episodes.parquet")
    splits = json.loads((root / "benchmark" / "splits.json").read_text())
    x = np.stack([np.stack(v).astype(np.float32) for v in df["model_input"]])
    y = df["target"].to_numpy(dtype=np.float32)
    return x.reshape(len(x), -1), y, splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    root = Path(cfg.get("artifacts", {}).get("root", "artifacts/medical"))
    seed = int(cfg["dataset"].get("seed", 42))
    if args.dataset != "med_circuitbench":
        raise SystemExit("baseline MVP currently expects prepared med_circuitbench artifacts")
    x, y, splits = _load_benchmark(root)
    train_ids = np.asarray(splits["train"], dtype=np.int64)
    val_ids = np.asarray(splits["validation"], dtype=np.int64)
    rng = np.random.default_rng(seed)

    rows = []
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=seed),
    )
    clf.fit(x[train_ids], y[train_ids])
    rows.append({"method": "logistic_regression_window", **_metrics(y[val_ids], clf.predict_proba(x[val_ids])[:, 1])})

    random_prob = rng.random(len(val_ids))
    rows.append({"method": "random_scores", **_metrics(y[val_ids], random_prob)})

    majority_prob = np.repeat(float(y[train_ids].mean()), len(val_ids))
    rows.append({"method": "majority_rate", **_metrics(y[val_ids], majority_prob)})

    out = root / args.dataset / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "baseline_metrics.parquet", index=False)
    (out / "manifest.json").write_text(
        json.dumps({"dataset": args.dataset, "commit": git_commit(), "seed": seed, "files": {"metrics": "baseline_metrics.parquet"}}, indent=2),
        encoding="utf-8",
    )
    print({"baseline_metrics": str(out / "baseline_metrics.parquet"), "methods": [row["method"] for row in rows]})


if __name__ == "__main__":
    main()
