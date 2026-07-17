from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

from .models import STABILITY_ANALYSIS_ARMS


METRICS = ["spearman", "kendall", "top1_agreement", "top3_jaccard", "sign_agreement", "cosine_similarity", "absolute_difference"]


def _run_dirs(runs_root: Path, arm: str) -> list[Path]:
    return sorted(path for path in (runs_root / arm).glob("run_*") if (path / "run_manifest.json").exists())


def _contribution_matrix(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(run_dir / "contributions_test.parquet")
    columns = [f"contribution_{index}" for index in range(5)]
    return frame["RecordID"].to_numpy(), frame[columns].to_numpy(dtype=np.float64)


def _rank_correlation(left: np.ndarray, right: np.ndarray, method: str) -> np.ndarray:
    result = np.empty(len(left), dtype=np.float64)
    for index, (a, b) in enumerate(zip(left, right)):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            result[index] = 0.0
        elif method == "spearman":
            result[index] = float(stats.spearmanr(a, b).statistic)
        else:
            result[index] = float(stats.kendalltau(a, b).statistic)
    return np.nan_to_num(result)


def _pair_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, np.ndarray]:
    top_left = np.argsort(-np.abs(left), axis=1)
    top_right = np.argsort(-np.abs(right), axis=1)
    top3 = np.asarray(
        [len(set(a[:3]).intersection(b[:3])) / len(set(a[:3]).union(b[:3])) for a, b in zip(top_left, top_right)],
        dtype=np.float64,
    )
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.divide((left * right).sum(axis=1), denominator, out=np.zeros(len(left)), where=denominator > 1e-12)
    return {
        "spearman": _rank_correlation(left, right, "spearman"),
        "kendall": _rank_correlation(left, right, "kendall"),
        "top1_agreement": (top_left[:, 0] == top_right[:, 0]).astype(np.float64),
        "top3_jaccard": top3,
        "sign_agreement": (np.sign(left) == np.sign(right)).mean(axis=1),
        "cosine_similarity": cosine,
        "absolute_difference": np.abs(left - right).mean(axis=1),
    }


def build_episode_stability(runs_root: Path, output_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    pair_summaries: list[dict] = []
    counts: dict[str, int] = {}
    try:
        for arm in STABILITY_ANALYSIS_ARMS:
            runs = _run_dirs(runs_root, arm)
            if len(runs) != 30:
                raise ValueError(f"{arm}: expected 30 completed runs, found {len(runs)}")
            counts[arm] = math.comb(len(runs), 2)
            matrices = {run.name: _contribution_matrix(run) for run in runs}
            for run_a, run_b in itertools.combinations(runs, 2):
                ids_a, contribution_a = matrices[run_a.name]
                ids_b, contribution_b = matrices[run_b.name]
                if not np.array_equal(ids_a, ids_b):
                    raise ValueError(f"Patient order mismatch: {run_a} vs {run_b}")
                representations = {
                    "signed": (contribution_a, contribution_b),
                    "absolute": (np.abs(contribution_a), np.abs(contribution_b)),
                    "l1_normalized": (
                        contribution_a / (np.abs(contribution_a).sum(axis=1, keepdims=True) + 1e-8),
                        contribution_b / (np.abs(contribution_b).sum(axis=1, keepdims=True) + 1e-8),
                    ),
                }
                for representation, (left, right) in representations.items():
                    metrics = _pair_metrics(left, right)
                    chunk = pd.DataFrame(
                        {
                            "model_arm": arm,
                            "run_a": run_a.name,
                            "run_b": run_b.name,
                            "representation": representation,
                            "RecordID": ids_a,
                            **metrics,
                        }
                    )
                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                    writer.write_table(table)
                    pair_summaries.append(
                        {
                            "model_arm": arm,
                            "run_a": run_a.name,
                            "run_b": run_b.name,
                            "representation": representation,
                            **{metric: float(np.mean(values)) for metric, values in metrics.items()},
                        }
                    )
    finally:
        if writer is not None:
            writer.close()
    return pd.DataFrame(pair_summaries), counts


def _hierarchical_ci(
    values: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
    chunk_size: int = 16,
) -> tuple[float, float]:
    pairs, patients = values.shape
    estimates = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        size = stop - start
        sampled_pairs = rng.integers(0, pairs, size=(size, pairs))
        selected = values[sampled_pairs]
        patient_sample = rng.integers(0, patients, size=(size, pairs, patients))
        sampled = np.take_along_axis(selected, patient_sample, axis=2)
        estimates[start:stop] = sampled.mean(axis=(1, 2))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def aggregate_stability(episode_path: Path, pair_summary: pd.DataFrame, repetitions: int = 1000) -> pd.DataFrame:
    episode = pd.read_parquet(episode_path)
    rows: list[dict] = []
    rng = np.random.default_rng(20260717)
    for (arm, representation), group in episode.groupby(["model_arm", "representation"], sort=True):
        pair_keys = list(group[["run_a", "run_b"]].drop_duplicates().itertuples(index=False, name=None))
        patient_ids = np.sort(group["RecordID"].unique())
        for metric in METRICS:
            pivot = group.pivot(index=["run_a", "run_b"], columns="RecordID", values=metric).loc[pair_keys, patient_ids]
            values = pivot.to_numpy(dtype=np.float64)
            low, high = _hierarchical_ci(values, repetitions, rng)
            flattened = values.reshape(-1)
            rows.append(
                {
                    "model_arm": arm,
                    "representation": representation,
                    "metric": metric,
                    "model_pairs": values.shape[0],
                    "patients": values.shape[1],
                    "mean": float(np.mean(flattened)),
                    "median": float(np.median(flattened)),
                    "std": float(np.std(flattened, ddof=1)),
                    "ci95_low_hierarchical": low,
                    "ci95_high_hierarchical": high,
                }
            )
    return pd.DataFrame(rows)
