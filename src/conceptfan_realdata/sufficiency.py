from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .metrics import sigmoid
from .models import STABILITY_ANALYSIS_ARMS


def _mask_bits(mask_id: int, n_concepts: int = 5) -> np.ndarray:
    return np.asarray([(mask_id >> position) & 1 for position in range(n_concepts)], dtype=np.float64)


def build_sufficiency(runs_root: Path, output_path: Path, controls_path: Path) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    control_rows: list[dict] = []
    counts: dict[str, int] = {}
    rng = np.random.default_rng(20260717)
    try:
        for arm in STABILITY_ANALYSIS_ARMS:
            runs = sorted(path for path in (runs_root / arm).glob("run_*") if (path / "run_manifest.json").exists())
            if len(runs) != 30:
                raise ValueError(f"{arm}: expected 30 completed runs, found {len(runs)}")
            counts[arm] = len(runs)
            for run_dir in runs:
                contributions = pd.read_parquet(run_dir / "contributions_test.parquet")
                logits = pd.read_parquet(run_dir / "logits_test.parquet")
                matrix = contributions[[f"contribution_{index}" for index in range(5)]].to_numpy(dtype=np.float64)
                bias = contributions["bias"].to_numpy(dtype=np.float64)
                clean_probability = logits["probability_raw"].to_numpy(dtype=np.float64)
                chunks: list[pd.DataFrame] = []
                for mask_id in range(32):
                    bits = _mask_bits(mask_id)
                    masked_logit = bias + (matrix * bits).sum(axis=1)
                    probability = sigmoid(masked_logit)
                    chunks.append(
                        pd.DataFrame(
                            {
                                "model_arm": arm,
                                "run_id": run_dir.name,
                                "RecordID": contributions["RecordID"].to_numpy(),
                                "mask_id": mask_id,
                                "mask_size": int(bits.sum()),
                                "masked_logit": masked_logit,
                                "masked_probability": probability,
                                "probability_difference_from_clean": probability - clean_probability,
                            }
                        )
                    )
                chunk = pd.concat(chunks, ignore_index=True)
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table)
                ranking = np.argsort(-np.abs(matrix), axis=1)
                patient_positions = np.arange(len(matrix))
                record_ids = contributions["RecordID"].to_numpy(dtype=np.int64)
                full_sum = matrix.sum(axis=1)
                for size in [1, 2, 3, 4, 5]:
                    top_indices = ranking[:, :size]
                    top_sum = np.take_along_axis(matrix, top_indices, axis=1).sum(axis=1)
                    top_probability = sigmoid(bias + top_sum)
                    removed_probability = sigmoid(bias + full_sum - top_sum)
                    combinations = np.asarray(list(itertools.combinations(range(5), size)), dtype=np.int64)
                    sampled_combinations = combinations[rng.integers(0, len(combinations), size=(len(matrix), 100))]
                    sampled_values = matrix[
                        patient_positions[:, None, None],
                        sampled_combinations,
                    ].sum(axis=2)
                    random_probabilities = sigmoid(bias[:, None] + sampled_values)
                    control_rows.extend(
                        pd.DataFrame(
                            {
                                "model_arm": arm,
                                "run_id": run_dir.name,
                                "RecordID": record_ids,
                                "M": size,
                                "random_draws": 100,
                                "top_ranking_probability": top_probability,
                                "random_same_size_probability_mean_100": random_probabilities.mean(axis=1),
                                "random_same_size_probability_std_100": random_probabilities.std(axis=1),
                                "clean_probability": clean_probability,
                                "sufficiency_gap": clean_probability - top_probability,
                                "comprehensiveness": clean_probability - removed_probability,
                                "decision_agreement_0_3": (top_probability >= 0.3) == (clean_probability >= 0.3),
                                "decision_agreement_0_5": (top_probability >= 0.5) == (clean_probability >= 0.5),
                                "decision_agreement_0_7": (top_probability >= 0.7) == (clean_probability >= 0.7),
                            }
                        ).to_dict("records")
                    )
    finally:
        if writer is not None:
            writer.close()
    pd.DataFrame(control_rows).to_parquet(controls_path, index=False, compression="zstd")
    return counts


def summarize_sufficiency(exhaustive_path: Path, controls_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    exhaustive = pd.read_parquet(exhaustive_path)
    controls = pd.read_parquet(controls_path)
    exhaustive_summary = exhaustive.groupby(["model_arm", "mask_size"], as_index=False).agg(
        observations=("RecordID", "size"),
        mean_probability=("masked_probability", "mean"),
        median_probability=("masked_probability", "median"),
        mean_probability_difference=("probability_difference_from_clean", "mean"),
    )
    controls_summary = controls.groupby(["model_arm", "M"], as_index=False).agg(
        observations=("RecordID", "size"),
        random_draws=("random_draws", "min"),
        top_ranking_probability=("top_ranking_probability", "mean"),
        random_same_size_probability=("random_same_size_probability_mean_100", "mean"),
        sufficiency_gap=("sufficiency_gap", "mean"),
        comprehensiveness=("comprehensiveness", "mean"),
        decision_agreement_0_3=("decision_agreement_0_3", "mean"),
        decision_agreement_0_5=("decision_agreement_0_5", "mean"),
        decision_agreement_0_7=("decision_agreement_0_7", "mean"),
    )
    return exhaustive_summary, controls_summary
