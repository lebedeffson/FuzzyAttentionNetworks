from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

from .calibration import fit_calibrators, select_primary_calibrator
from .data import PreparedData
from .metrics import binary_metrics


CHANNELS = ["V", "V+M", "V+D", "V+M+D"]
CONTROLS = [
    "residuals",
    "shuffled_labels",
    "matched_gaussian_noise",
    "permuted_residual_vectors",
    "missingness_only",
    "delta_time_only",
    "raw_value_only",
]


def _patient_positions(data: PreparedData, record_ids: np.ndarray) -> np.ndarray:
    lookup = {int(record_id): position for position, record_id in enumerate(data.record_ids)}
    return np.asarray([lookup[int(record_id)] for record_id in record_ids], dtype=np.int64)


def _residual_summary(run_dir: Path, split: str, data: PreparedData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_parquet(run_dir / f"concepts_{split}.parquet")
    record_ids = frame["RecordID"].to_numpy(dtype=np.int64)
    positions = _patient_positions(data, record_ids)
    predicted = np.stack(
        [np.stack(frame[f"concept_{concept}_trajectory"].map(np.asarray).to_numpy()) for concept in range(5)],
        axis=-1,
    ).astype(np.float64)
    target = data.concepts[positions].astype(np.float64)
    observed = data.concept_mask[positions].astype(bool)
    residual = np.where(observed, target - predicted, 0.0)
    hours = np.arange(residual.shape[1], dtype=np.float64)
    features: list[np.ndarray] = []
    for concept in range(5):
        values = residual[:, :, concept]
        mask = observed[:, :, concept]
        count = mask.sum(axis=1).clip(min=1)
        mean = values.sum(axis=1) / count
        maximum = np.max(np.where(mask, np.abs(values), -np.inf), axis=1)
        maximum[~np.isfinite(maximum)] = 0.0
        centered_hour = hours[None, :] - np.divide((mask * hours).sum(axis=1), count)[:, None]
        centered_value = values - mean[:, None]
        slope = np.divide(
            (centered_hour * centered_value * mask).sum(axis=1),
            (centered_hour**2 * mask).sum(axis=1),
            out=np.zeros(len(values)),
            where=(centered_hour**2 * mask).sum(axis=1) > 1e-12,
        )
        last_index = np.maximum(mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1), 0)
        last = values[np.arange(len(values)), last_index]
        std = np.sqrt(((centered_value**2) * mask).sum(axis=1) / count)
        missing = mask.shape[1] - mask.sum(axis=1)
        features.extend([last, mean, maximum, std, slope, missing.astype(np.float64) / mask.shape[1]])
    return record_ids, np.column_stack(features), positions


def _control_features(control: str, residuals: np.ndarray, positions: np.ndarray, data: PreparedData, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if control in {"residuals", "shuffled_labels", "permuted_residual_vectors"}:
        if control == "permuted_residual_vectors":
            return residuals[rng.permutation(len(residuals))]
        return residuals
    if control == "matched_gaussian_noise":
        scale = np.std(residuals, axis=0, ddof=1)
        return rng.normal(0.0, np.maximum(scale, 1e-8), size=residuals.shape)
    if control == "missingness_only":
        return np.column_stack([data.m[positions].mean(axis=1), data.concept_mask[positions].mean(axis=1)])
    if control == "delta_time_only":
        return np.column_stack([data.d[positions].mean(axis=1), data.d[positions].max(axis=1)])
    if control == "raw_value_only":
        return np.column_stack([data.v[positions].mean(axis=1), data.v[positions].std(axis=1)])
    raise ValueError(control)


def _bootstrap_metric(y: np.ndarray, probability: np.ndarray, metric: str, seed: int, repetitions: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(repetitions):
        sample = rng.integers(0, len(y), size=len(y))
        if np.unique(y[sample]).size < 2:
            continue
        function = average_precision_score if metric == "AUPRC" else roc_auc_score
        estimates.append(float(function(y[sample], probability[sample])))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _fit_l2(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> LogisticRegression:
    best: tuple[float, LogisticRegression] | None = None
    for c_value in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = LogisticRegression(C=c_value, solver="liblinear", max_iter=2000)
        model.fit(x_train, y_train)
        score = float(average_precision_score(y_validation, model.predict_proba(x_validation)[:, 1]))
        if best is None or score > best[0]:
            best = (score, model)
    assert best is not None
    return best[1]


def _run_dirs(runs_root: Path, channels: str) -> list[Path]:
    if channels == "V+M+D":
        candidates = sorted((runs_root / "ConceptFAN-NoAlpha").glob("run_*"))
        candidates = [path for path in candidates if json.loads((path / "run_manifest.json").read_text())["data_order_seed"] == 1001]
    else:
        candidates = sorted((runs_root / "channel_ablation" / channels.replace("+", "_")).glob("run_*"))
    return [path for path in candidates if (path / "run_manifest.json").exists()]


def analyze_residual_signal(
    data: PreparedData,
    runs_root: Path,
    output_path: Path,
    importance_path: Path,
    bootstrap_repetitions: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    importance_rows: list[dict] = []
    for channel_position, channels in enumerate(CHANNELS):
        run_dirs = _run_dirs(runs_root, channels)
        if len(run_dirs) != 10:
            raise ValueError(f"{channels}: expected 10 leakage-localization runs, found {len(run_dirs)}")
        for run_position, run_dir in enumerate(run_dirs):
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            summaries: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
                split: _residual_summary(run_dir, split, data) for split in ["train", "validation", "calibration", "test"]
            }
            seed = 20260717 + channel_position * 1000 + run_position
            for control in CONTROLS:
                matrices: dict[str, np.ndarray] = {}
                targets: dict[str, np.ndarray] = {}
                for split_position, split in enumerate(["train", "validation", "calibration", "test"]):
                    _, residuals, positions = summaries[split]
                    matrices[split] = _control_features(control, residuals, positions, data, seed + split_position)
                    targets[split] = data.y[positions].copy()
                training_target = targets["train"].copy()
                if control == "shuffled_labels":
                    training_target = np.random.default_rng(seed).permutation(training_target)
                classifier = _fit_l2(matrices["train"], training_target, matrices["validation"], targets["validation"])
                calibration_logits = classifier.decision_function(matrices["calibration"])
                calibrators = fit_calibrators(calibration_logits, targets["calibration"])
                selected, _ = select_primary_calibrator(calibrators, calibration_logits, targets["calibration"])
                test_logits = classifier.decision_function(matrices["test"])
                test_probability = calibrators[selected].predict(test_logits)
                metrics = binary_metrics(targets["test"], test_probability, calibrators[selected].transform_logits(test_logits))
                auprc_low, auprc_high = _bootstrap_metric(
                    targets["test"], test_probability, "AUPRC", seed, bootstrap_repetitions
                )
                auroc_low, auroc_high = _bootstrap_metric(
                    targets["test"], test_probability, "AUROC", seed + 1, bootstrap_repetitions
                )
                row = {
                    "channels": channels,
                    "run_id": run_dir.name,
                    "init_seed": manifest["init_seed"],
                    "control": control,
                    "diagnostic_model": "l2_logistic_regression",
                    "selected_C": float(classifier.C),
                    "selected_calibrator": selected,
                    **metrics,
                    "AUPRC_ci95_low": auprc_low,
                    "AUPRC_ci95_high": auprc_high,
                    "AUROC_ci95_low": auroc_low,
                    "AUROC_ci95_high": auroc_high,
                }
                rows.append(row)
                if control == "residuals":
                    baseline = metrics["AUPRC"]
                    rng = np.random.default_rng(seed)
                    for concept in range(5):
                        permuted = matrices["test"].copy()
                        columns = np.arange(concept * 6, concept * 6 + 6)
                        permuted[:, columns] = permuted[rng.permutation(len(permuted))][:, columns]
                        probability = calibrators[selected].predict(classifier.decision_function(permuted))
                        importance_rows.append(
                            {
                                "channels": channels,
                                "run_id": run_dir.name,
                                "group": data.concept_names[concept],
                                "AUPRC_drop": float(baseline - average_precision_score(targets["test"], probability)),
                            }
                        )
                    mlp = MLPClassifier(
                        hidden_layer_sizes=(32,), alpha=1e-3, early_stopping=True, random_state=seed, max_iter=300
                    )
                    mlp.fit(matrices["train"], targets["train"])
                    mlp_probability = mlp.predict_proba(matrices["test"])[:, 1]
                    rows.append(
                        {
                            **{key: row[key] for key in ["channels", "run_id", "init_seed"]},
                            "control": "residuals",
                            "diagnostic_model": "mlp_sensitivity",
                            **binary_metrics(targets["test"], mlp_probability),
                        }
                    )
    frame = pd.DataFrame(rows)
    importance = pd.DataFrame(importance_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False, compression="zstd")
    importance.to_parquet(importance_path, index=False, compression="zstd")
    return frame, importance


def summarize_leakage(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["channels", "control", "diagnostic_model"], as_index=False).agg(
        runs=("run_id", "nunique"),
        AUPRC_mean=("AUPRC", "mean"),
        AUPRC_std=("AUPRC", "std"),
        AUROC_mean=("AUROC", "mean"),
        AUROC_std=("AUROC", "std"),
        Brier_mean=("Brier", "mean"),
    )
