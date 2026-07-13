from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


STATE_NAMES = ["I", "R", "V", "O", "S"]
TREATMENT_NAMES = ["antibiotic", "fluids", "vasopressor"]
OBSERVATION_NAMES = ["temperature", "heart_rate", "map", "lactate", "wbc", "creatinine", "urine", "spo2"]

A = np.asarray(
    [
        [0.90, 0.00, 0.00, 0.00, 0.00],
        [0.80, 0.75, 0.00, 0.00, 0.00],
        [0.00, 0.70, 0.70, 0.00, 0.00],
        [0.15, 0.00, 0.65, 0.75, 0.00],
        [0.10, 0.00, 0.60, 0.75, 0.50],
    ],
    dtype=np.float64,
)
B = np.asarray(
    [
        [-0.65, 0.00, 0.00],
        [-0.20, 0.00, 0.00],
        [0.00, -0.25, -0.70],
        [0.00, -0.20, -0.10],
        [0.00, -0.10, -0.25],
    ],
    dtype=np.float64,
)
DELAY = np.asarray([3, 1, 0], dtype=int)
BIAS = np.asarray([-2.2, -2.4, -2.4, -2.5, -2.8], dtype=np.float64)
MU_X = np.asarray([36.7, 75.0, 90.0, 1.2, 7.0, 0.9, 90.0, 97.0], dtype=np.float64)
G = np.asarray(
    [
        [0.8, 2.2, 0.0, 0.0, 0.0],
        [0.0, 25.0, 25.0, 5.0, 0.0],
        [0.0, 0.0, -35.0, -5.0, 0.0],
        [0.0, 0.0, 1.5, 4.5, 0.0],
        [4.0, 12.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2.8, 0.0],
        [0.0, 0.0, -10.0, -70.0, 0.0],
        [0.0, 0.0, 0.0, -10.0, 0.0],
    ],
    dtype=np.float64,
)
H = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 4.0, 10.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 15.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
CLINICAL_MIN = np.asarray([35.0, 35.0, 35.0, 0.3, 1.0, 0.2, 0.0, 65.0], dtype=np.float64)
CLINICAL_MAX = np.asarray([41.0, 180.0, 140.0, 12.0, 35.0, 8.0, 250.0, 100.0], dtype=np.float64)
OBS_NOISE = np.asarray([0.25, 5.0, 4.0, 0.30, 1.2, 0.15, 10.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int = 42
    n_samples: int = 10_000
    sequence_length: int = 42
    input_window: int = 36
    prediction_horizon: int = 6


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def true_graph(edge_threshold: float = 0.50) -> Dict[str, Dict[str, float]]:
    graph: Dict[str, Dict[str, float]] = {name: {} for name in STATE_NAMES}
    for dst_idx, dst_name in enumerate(STATE_NAMES):
        for src_idx, src_name in enumerate(STATE_NAMES):
            weight = float(A[dst_idx, src_idx])
            if src_idx != dst_idx and abs(weight) >= edge_threshold:
                graph[src_name][dst_name] = weight
    graph["S"]["Y"] = 1.0
    return graph


def _treatment_policy(prev_state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    probs = np.asarray(
        [
            0.15 + 0.70 * float(prev_state[0] > 0.55),
            0.10 + 0.70 * float(prev_state[2] + prev_state[3] > 0.75),
            0.05 + 0.80 * float(prev_state[2] > 0.60),
        ]
    )
    treatment = (rng.random(3) < probs).astype(np.float64)
    random_flip = rng.random(3) < 0.05
    treatment[random_flip] = 1.0 - treatment[random_flip]
    return treatment


def _missing_mask(obs: np.ndarray, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    base = np.asarray([0.85, 0.95, 0.95, 0.45, 0.55, 0.55, 0.55, 0.90])
    severity = np.clip(0.5 * state[3] + 0.5 * state[4], 0, 1)
    extra = np.asarray([0.05, 0.02, 0.02, 0.40, 0.25, 0.20, 0.20, 0.05]) * severity
    return (rng.random(8) < np.clip(base + extra, 0.0, 1.0)).astype(np.float64)


def _delta_times(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.float64)
    last_seen = np.full(mask.shape[1], -10_000, dtype=int)
    for t in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[t, j] == 1:
                last_seen[j] = t
                out[t, j] = 0.0
            else:
                delta = 36 if last_seen[j] < 0 else min(t - last_seen[j], 36)
                out[t, j] = np.log1p(delta)
    return out


def generate_episode(episode_id: int, cfg: BenchmarkConfig, rng: np.random.Generator) -> Dict[str, object]:
    states = np.zeros((cfg.sequence_length, 5), dtype=np.float64)
    treatments = np.zeros((cfg.sequence_length, 3), dtype=np.float64)
    observations = np.zeros((cfg.sequence_length, 8), dtype=np.float64)
    masks = np.zeros((cfg.sequence_length, 8), dtype=np.float64)
    infection_start = int(rng.integers(0, max(1, cfg.input_window // 2)))
    impulse = np.zeros(cfg.sequence_length, dtype=np.float64)
    impulse[infection_start] = 1.0
    states[0] = sigmoid(BIAS + 1.4 * impulse[0] * np.asarray([1, 0, 0, 0, 0]) + rng.normal(0.0, 0.03, 5))
    for t in range(cfg.sequence_length):
        if t > 0:
            treatments[t] = _treatment_policy(states[t - 1], rng)
            delayed = np.zeros(3, dtype=np.float64)
            for j, delay in enumerate(DELAY):
                idx = t - int(delay)
                if idx >= 0:
                    delayed[j] = treatments[idx, j]
            states[t] = sigmoid(
                BIAS
                + A @ states[t - 1]
                + B @ delayed
                + 1.4 * impulse[t] * np.asarray([1, 0, 0, 0, 0])
                + rng.normal(0.0, 0.03, 5)
            )
        observations[t] = np.clip(
            MU_X + G @ states[t] + H @ treatments[t] + rng.normal(0.0, OBS_NOISE, 8),
            CLINICAL_MIN,
            CLINICAL_MAX,
        )
        masks[t] = _missing_mask(observations[t], states[t], rng)
    deltas = _delta_times(masks)
    values = observations.copy()
    values[masks == 0] = 0.0
    model_input = np.concatenate(
        [values[: cfg.input_window], masks[: cfg.input_window], deltas[: cfg.input_window], treatments[: cfg.input_window]],
        axis=1,
    )
    target = int(states[cfg.input_window : cfg.sequence_length, 4].max() >= 0.65)
    return {
        "episode_id": int(episode_id),
        "states": states,
        "treatments": treatments,
        "observations": observations,
        "masks": masks,
        "delta_time": deltas,
        "model_input": model_input,
        "target": target,
    }


def split_ids(n_samples: int, seed: int) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    ids = np.arange(n_samples)
    rng.shuffle(ids)
    n_train = int(0.60 * n_samples)
    n_val = int(0.20 * n_samples)
    return {
        "train": ids[:n_train].astype(int).tolist(),
        "validation": ids[n_train : n_train + n_val].astype(int).tolist(),
        "test": ids[n_train + n_val :].astype(int).tolist(),
    }


def _array_to_list(row: Dict[str, object]) -> Dict[str, object]:
    out = dict(row)
    for key in ("states", "treatments", "observations", "masks", "delta_time", "model_input"):
        out[key] = np.asarray(out[key]).astype(float).tolist()
    return out


def write_benchmark(out_dir: Path, cfg: BenchmarkConfig) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    episodes = [_array_to_list(generate_episode(i, cfg, rng)) for i in range(cfg.n_samples)]
    pd.DataFrame(episodes).to_parquet(out_dir / "episodes.parquet", index=False)
    graph = true_graph()
    (out_dir / "true_graph.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
    splits = split_ids(cfg.n_samples, cfg.seed)
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "med_circuitbench",
        "config": asdict(cfg),
        "files": {
            "episodes": "episodes.parquet",
            "true_graph": "true_graph.json",
            "splits": "splits.json",
        },
        "true_graph_sha256": hashlib.sha256((out_dir / "true_graph.json").read_bytes()).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {k: str(out_dir / v) for k, v in manifest["files"].items()}
