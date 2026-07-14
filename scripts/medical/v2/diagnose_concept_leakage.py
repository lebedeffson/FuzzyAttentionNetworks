#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score


def diagnose(train_pred: np.ndarray, train_true: np.ndarray, train_y: np.ndarray, val_pred: np.ndarray, val_true: np.ndarray, val_y: np.ndarray, seed: int = 0) -> dict:
    mapper = LinearRegression().fit(train_pred, train_true)
    train_residual = train_pred - mapper.predict(train_pred)
    val_residual = val_pred - mapper.predict(val_pred)
    rng = np.random.default_rng(seed)
    shuffled = val_residual.copy()
    rng.shuffle(shuffled, axis=0)

    def score(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> float:
        if len(np.unique(y_train)) < 2:
            return float(np.mean(y_val))
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(x_train, y_train)
        return float(average_precision_score(y_val, clf.predict_proba(x_val)[:, 1]))

    residual_auprc = score(train_residual, train_y, val_residual, val_y)
    shuffled_auprc = score(train_residual, train_y, shuffled, val_y)
    prevalence = float(np.mean(val_y))
    return {
        "prevalence": prevalence,
        "true_concepts_auprc": score(train_true, train_y, val_true, val_y),
        "predicted_concepts_auprc": score(train_pred, train_y, val_pred, val_y),
        "residual_auprc": residual_auprc,
        "shuffled_residual_auprc": shuffled_auprc,
        "leakage_gate_passed": bool(residual_auprc <= prevalence + 0.05 or residual_auprc <= shuffled_auprc),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = np.load(args.input)
    result = diagnose(data["train_pred"], data["train_true"], data["train_y"], data["val_pred"], data["val_true"], data["val_y"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

