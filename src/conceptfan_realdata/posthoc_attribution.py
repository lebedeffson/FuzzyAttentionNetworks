from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import torch
import yaml
from scipy import stats

from .data import PreparedData, load_prepared
from .models import parameter_sha256
from .training import load_model_from_checkpoint


STATUS = "POSTHOC_ATTRIBUTION_AUDIT_COMPLETE"
METHODS = ("integrated_gradients", "gradient_shap")
METRICS = ("spearman", "kendall", "top1_agreement", "top3_jaccard", "sign_agreement", "cosine_similarity")
PROXY_NAMES = (
    "hemodynamic_instability",
    "respiratory_dysfunction",
    "renal_dysfunction",
    "neurological_dysfunction",
    "metabolic_inflammatory_dysregulation",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _proxy_source_variables(data_config: dict) -> tuple[dict[str, str], dict[str, str]]:
    direct: dict[str, str] = {}
    source: dict[str, str] = {}
    definitions = data_config["concepts"]["definitions"]
    for proxy, directions in definitions.items():
        for variables in directions.values():
            for variable in variables:
                if variable == "PFratio":
                    for base in ("PaO2", "FiO2"):
                        direct[base] = proxy
                        source[base] = "existing_proxy_formula:PFratio(PaO2,FiO2)"
                elif variable == "Urine6h":
                    direct["Urine"] = proxy
                    source["Urine"] = "existing_proxy_formula:Urine6h(Urine)"
                else:
                    if variable in direct and direct[variable] != proxy:
                        raise ValueError(f"Ambiguous primary proxy mapping for {variable}")
                    direct[variable] = proxy
                    source[variable] = f"existing_proxy_formula:{variable}"
    return direct, source


def build_channel_map(data: PreparedData, data_config: dict) -> pd.DataFrame:
    direct, sources = _proxy_source_variables(data_config)
    rows: list[dict] = []
    index = 0
    rationale = {
        "hemodynamic_instability": "variable used by the frozen hemodynamic proxy formula",
        "respiratory_dysfunction": "variable used directly or through P/F ratio by the frozen respiratory proxy formula",
        "renal_dysfunction": "variable used directly or through rolling urine by the frozen renal proxy formula",
        "neurological_dysfunction": "variable used by the frozen neurological proxy formula",
        "metabolic_inflammatory_dysregulation": "variable used by the frozen metabolic-inflammatory proxy formula",
    }
    for channel_type in ("V", "M", "D"):
        for variable in data.variables:
            proxy = direct.get(variable, "unassigned")
            primary = channel_type == "V" and proxy != "unassigned"
            rows.append(
                {
                    "channel_index": index,
                    "channel_name": f"{channel_type}_{variable}",
                    "base_variable": variable,
                    "channel_type": channel_type,
                    "proxy_concept": proxy if primary else ("observation_process" if channel_type in {"M", "D"} else "unassigned"),
                    "mapping_weight": 1.0 if primary else 0.0,
                    "mapping_source": sources.get(variable, "frozen_input_schema:unassigned"),
                    "clinical_rationale": rationale.get(proxy, "not used by any frozen proxy formula"),
                    "included_in_primary_analysis": primary,
                    "sensitivity_proxy_concept": proxy,
                }
            )
            index += 1
    for static_index in range(data.static.shape[1]):
        rows.append(
            {
                "channel_index": index,
                "channel_name": f"STATIC_{static_index}",
                "base_variable": f"static_{static_index}",
                "channel_type": "STATIC",
                "proxy_concept": "unassigned",
                "mapping_weight": 0.0,
                "mapping_source": "frozen_input_schema:static",
                "clinical_rationale": "static encoded covariate retained outside temporal physiological proxy groups",
                "included_in_primary_analysis": False,
                "sensitivity_proxy_concept": "unassigned",
            }
        )
        index += 1
    frame = pd.DataFrame(rows)
    expected = data.inputs("V+M+D").shape[-1]
    if len(frame) != expected or frame["channel_index"].tolist() != list(range(expected)):
        raise AssertionError("Generated channel map does not match the frozen model input")
    return frame


def validate_channel_map(frame: pd.DataFrame, input_dim: int) -> None:
    required = {
        "channel_index", "channel_name", "base_variable", "channel_type", "proxy_concept", "mapping_weight",
        "mapping_source", "clinical_rationale", "included_in_primary_analysis", "sensitivity_proxy_concept",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Channel map missing columns: {sorted(missing)}")
    if len(frame) != input_dim or frame["channel_index"].astype(int).tolist() != list(range(input_dim)):
        raise ValueError("Channel map indices do not exactly cover the frozen input")
    primary = frame["included_in_primary_analysis"].map(_bool)
    if not (frame.loc[primary, "channel_type"] == "V").all():
        raise ValueError("Primary proxy map may contain only physiological value channels")
    if not set(frame.loc[primary, "proxy_concept"]).issubset(PROXY_NAMES):
        raise ValueError("Primary proxy map contains an unknown group")
    if not np.allclose(frame.loc[primary, "mapping_weight"].astype(float), 1.0):
        raise ValueError("Primary hard-map weights must equal one")
    if not np.allclose(frame.loc[~primary, "mapping_weight"].astype(float), 0.0):
        raise ValueError("Channels outside the primary map must have zero primary weight")


def map_matrices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    primary = np.zeros((len(PROXY_NAMES), len(frame)), dtype=np.float64)
    sensitivity = np.zeros_like(primary)
    proxy_index = {name: index for index, name in enumerate(PROXY_NAMES)}
    for row in frame.itertuples(index=False):
        position = int(row.channel_index)
        if _bool(row.included_in_primary_analysis):
            primary[proxy_index[row.proxy_concept], position] = float(row.mapping_weight)
        if row.channel_type in {"V", "M", "D"} and row.sensitivity_proxy_concept in proxy_index:
            sensitivity[proxy_index[row.sensitivity_proxy_concept], position] = 1.0
    if np.any(primary.sum(axis=0) > 1.0) or np.any(sensitivity.sum(axis=0) > 1.0):
        raise ValueError("Proxy map double-counts at least one input channel")
    return primary, sensitivity


class LogitWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).logit


def _gradient(wrapper: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scaled = x.detach().requires_grad_(True)
    output = wrapper(scaled)
    gradient = torch.autograd.grad(output.sum(), scaled, create_graph=False)[0]
    return output.detach(), gradient.detach()


def integrated_gradients(
    wrapper: torch.nn.Module,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    n_steps: int,
    internal_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if baseline.ndim == inputs.ndim - 1:
        baseline = baseline.unsqueeze(0).expand_as(inputs)
    if baseline.shape != inputs.shape:
        raise ValueError("Integrated Gradients baseline shape mismatch")
    nodes, weights = np.polynomial.legendre.leggauss(n_steps)
    alphas = torch.as_tensor((nodes + 1.0) / 2.0, dtype=inputs.dtype, device=inputs.device)
    quadrature = torch.as_tensor(weights / 2.0, dtype=inputs.dtype, device=inputs.device)
    result = torch.empty_like(inputs)
    errors = torch.empty(inputs.shape[0], dtype=inputs.dtype, device=inputs.device)
    for start in range(0, len(inputs), internal_batch_size):
        stop = min(start + internal_batch_size, len(inputs))
        x = inputs[start:stop]
        base = baseline[start:stop]
        delta = x - base
        average_gradient = torch.zeros_like(x)
        for alpha, weight in zip(alphas, quadrature):
            _, gradient = _gradient(wrapper, base + alpha * delta)
            average_gradient.add_(gradient, alpha=weight.item())
        attribution = delta * average_gradient
        with torch.no_grad():
            difference = wrapper(x) - wrapper(base)
        result[start:stop] = attribution
        errors[start:stop] = torch.abs(attribution.flatten(1).sum(1) - difference)
    return result, errors


def gradient_shap(
    wrapper: torch.nn.Module,
    inputs: torch.Tensor,
    background: torch.Tensor,
    background_indices: np.ndarray,
    alphas: np.ndarray,
    stdevs: float,
) -> torch.Tensor:
    if background_indices.shape != alphas.shape or background_indices.shape[0] != len(inputs):
        raise ValueError("GradientSHAP sampling schedule mismatch")
    result = torch.zeros_like(inputs)
    for sample in range(background_indices.shape[1]):
        selected = background[torch.as_tensor(background_indices[:, sample], dtype=torch.long, device=inputs.device)]
        alpha = torch.as_tensor(alphas[:, sample], dtype=inputs.dtype, device=inputs.device).view(-1, 1, 1)
        noise = torch.zeros_like(inputs)
        if stdevs:
            noise.normal_(mean=0.0, std=stdevs)
        _, gradient = _gradient(wrapper, selected + alpha * (inputs + noise - selected))
        result += (inputs - selected) * gradient
    return result / float(background_indices.shape[1])


def _checkpoint_dirs(runs_root: Path, arm: str) -> list[Path]:
    return sorted(path for path in (runs_root / arm).glob("run_*") if (path / "checkpoint_best.pt").exists())


@dataclass
class FrozenInputs:
    data: PreparedData
    all_inputs: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    test_ids: np.ndarray
    median_baseline: np.ndarray
    background_indices_in_train: np.ndarray
    background_ids: np.ndarray
    gs_background_schedule: np.ndarray
    gs_alpha_schedule: np.ndarray


def prepare_frozen_inputs(data: PreparedData, config: dict) -> FrozenInputs:
    all_inputs = data.inputs("V+M+D")
    train_indices = data.indices("train")
    test_indices = data.indices("test")
    train_y = data.y[train_indices]
    size = int(config["gradient_shap"]["background_size"])
    rng = np.random.default_rng(int(config["gradient_shap"]["background_seed"]))
    positives = np.flatnonzero(train_y == 1)
    negatives = np.flatnonzero(train_y == 0)
    positive_count = size // 2
    selected = np.concatenate(
        [rng.choice(positives, positive_count, replace=False), rng.choice(negatives, size - positive_count, replace=False)]
    )
    rng.shuffle(selected)
    sample_count = int(config["gradient_shap"]["n_samples"])
    schedule_rng = np.random.default_rng(int(config["gradient_shap"]["background_seed"]) + 1)
    schedule = schedule_rng.integers(0, size, size=(len(test_indices), sample_count), dtype=np.int16)
    alphas = schedule_rng.uniform(0.0, 1.0, size=(len(test_indices), sample_count)).astype(np.float32)
    return FrozenInputs(
        data=data,
        all_inputs=all_inputs,
        train_indices=train_indices,
        test_indices=test_indices,
        test_ids=data.record_ids[test_indices],
        median_baseline=np.median(all_inputs[train_indices], axis=0).astype(np.float32),
        background_indices_in_train=selected,
        background_ids=data.record_ids[train_indices[selected]],
        gs_background_schedule=schedule,
        gs_alpha_schedule=alphas,
    )


def _compute_one_checkpoint(
    run_dir: Path,
    frozen: FrozenInputs,
    data_config: dict,
    config: dict,
    cache_path: Path,
    device: torch.device,
) -> dict:
    if cache_path.exists() and bool(config["runtime"].get("resume", True)):
        with np.load(cache_path, allow_pickle=False) as payload:
            return json.loads(str(payload["metadata"].item()))
    model, checkpoint = load_model_from_checkpoint(run_dir / "checkpoint_best.pt", data_config, device)
    if checkpoint["model_arm"] != "PlainTransformer" or checkpoint["channels"] != "V+M+D":
        raise ValueError(f"Unexpected checkpoint contract in {run_dir}")
    wrapper = LogitWrapper(model).eval()
    test = frozen.all_inputs[frozen.test_indices]
    background = torch.as_tensor(
        frozen.all_inputs[frozen.train_indices[frozen.background_indices_in_train]], dtype=torch.float32, device=device
    )
    zero = torch.zeros(test.shape[1:], dtype=torch.float32, device=device)
    median = torch.as_tensor(frozen.median_baseline, dtype=torch.float32, device=device)
    batch_size = int(config["runtime"]["test_batch_size"])
    ig_parts: list[np.ndarray] = []
    median_parts: list[np.ndarray] = []
    gs_parts: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    median_errors: list[np.ndarray] = []
    n_steps = int(config["integrated_gradients"]["n_steps"])
    for start in range(0, len(test), batch_size):
        stop = min(start + batch_size, len(test))
        x = torch.as_tensor(test[start:stop], dtype=torch.float32, device=device)
        ig, error = integrated_gradients(wrapper, x, zero, n_steps, int(config["integrated_gradients"]["internal_batch_size"]))
        ig_median, median_error = integrated_gradients(
            wrapper, x, median, n_steps, int(config["integrated_gradients"]["internal_batch_size"])
        )
        gs = gradient_shap(
            wrapper,
            x,
            background,
            frozen.gs_background_schedule[start:stop],
            frozen.gs_alpha_schedule[start:stop],
            float(config["gradient_shap"]["stdevs"]),
        )
        ig_parts.append(ig.cpu().numpy().astype(np.float32))
        median_parts.append(ig_median.cpu().numpy().astype(np.float32))
        gs_parts.append(gs.cpu().numpy().astype(np.float32))
        errors.append(error.cpu().numpy())
        median_errors.append(median_error.cpu().numpy())
    ig_values = np.concatenate(ig_parts)
    ig_median_values = np.concatenate(median_parts)
    gs_values = np.concatenate(gs_parts)
    completeness = np.concatenate(errors)
    completeness_median = np.concatenate(median_errors)
    threshold = float(config["integrated_gradients"]["completeness_median_threshold"])
    used_steps = n_steps
    if float(np.median(completeness)) > threshold:
        used_steps = int(config["integrated_gradients"]["retry_n_steps"])
        ig_parts, errors = [], []
        for start in range(0, len(test), batch_size):
            stop = min(start + batch_size, len(test))
            x = torch.as_tensor(test[start:stop], dtype=torch.float32, device=device)
            ig, error = integrated_gradients(
                wrapper, x, zero, used_steps, int(config["integrated_gradients"]["internal_batch_size"])
            )
            ig_parts.append(ig.cpu().numpy().astype(np.float32))
            errors.append(error.cpu().numpy())
        ig_values = np.concatenate(ig_parts)
        completeness = np.concatenate(errors)
    metadata = {
        "run_id": run_dir.name,
        "checkpoint_sha256": sha256_file(run_dir / "checkpoint_best.pt"),
        "parameter_sha256": parameter_sha256(model),
        "ig_n_steps_used": used_steps,
        "episodes": len(test),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        integrated_gradients=ig_values,
        integrated_gradients_median=ig_median_values,
        gradient_shap=gs_values,
        completeness=completeness.astype(np.float32),
        completeness_median=completeness_median.astype(np.float32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return metadata


def aggregate_attributions(values: np.ndarray, matrix: np.ndarray, epsilon: float) -> dict[str, np.ndarray]:
    signed_channels = values.sum(axis=2, dtype=np.float64)
    absolute_channels = np.abs(values).sum(axis=2, dtype=np.float64)
    signed_proxy = np.einsum("med,kd->mek", signed_channels, matrix, optimize=True)
    absolute_proxy = np.einsum("med,kd->mek", absolute_channels, matrix, optimize=True)
    assigned = matrix.sum(axis=0) > 0
    signed_unassigned = signed_channels[:, :, ~assigned].sum(axis=2)
    absolute_unassigned = absolute_channels[:, :, ~assigned].sum(axis=2)
    original_sum = signed_channels.sum(axis=2)
    grouped_sum = signed_proxy.sum(axis=2)
    conservation = original_sum - grouped_sum - signed_unassigned
    signed_norm = signed_proxy / (np.abs(signed_proxy).sum(axis=2, keepdims=True) + epsilon)
    absolute_norm = absolute_proxy / (absolute_proxy.sum(axis=2, keepdims=True) + epsilon)
    return {
        "signed_channels": signed_channels,
        "absolute_channels": absolute_channels,
        "signed_proxy": signed_proxy,
        "absolute_proxy": absolute_proxy,
        "signed_l1_normalized": signed_norm,
        "absolute_l1_normalized": absolute_norm,
        "original_sum": original_sum,
        "grouped_sum": grouped_sum,
        "signed_unassigned": signed_unassigned,
        "absolute_unassigned": absolute_unassigned,
        "conservation_error": conservation,
    }


def _rank_correlations(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_ranks = stats.rankdata(left, axis=1)
    right_ranks = stats.rankdata(right, axis=1)
    left_centered = left_ranks - left_ranks.mean(axis=1, keepdims=True)
    right_centered = right_ranks - right_ranks.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left_centered, axis=1) * np.linalg.norm(right_centered, axis=1)
    spearman = np.divide(
        (left_centered * right_centered).sum(axis=1), denominator, out=np.zeros(len(left)), where=denominator > 1e-12
    )
    if left.shape[1] <= 10:
        first, second = np.triu_indices(left.shape[1], 1)
        left_sign = np.sign(left[:, first] - left[:, second])
        right_sign = np.sign(right[:, first] - right[:, second])
        numerator = (left_sign * right_sign).sum(axis=1)
        non_tied_left = np.count_nonzero(left_sign, axis=1)
        non_tied_right = np.count_nonzero(right_sign, axis=1)
        tau_denominator = np.sqrt(non_tied_left * non_tied_right)
        kendall = np.divide(numerator, tau_denominator, out=np.zeros(len(left)), where=tau_denominator > 0)
    else:
        kendall = np.empty(len(left), dtype=np.float64)
        for index, (a, b) in enumerate(zip(left, right)):
            kendall[index] = 0.0 if np.std(a) < 1e-12 or np.std(b) < 1e-12 else float(stats.kendalltau(a, b).statistic)
    return np.nan_to_num(spearman), np.nan_to_num(kendall)


def episode_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, np.ndarray]:
    spearman, kendall = _rank_correlations(left, right)
    top_left = np.argsort(-np.abs(left), axis=1)
    top_right = np.argsort(-np.abs(right), axis=1)
    top_count = min(3, left.shape[1])
    top3 = np.asarray(
        [len(set(a[:top_count]) & set(b[:top_count])) / len(set(a[:top_count]) | set(b[:top_count])) for a, b in zip(top_left, top_right)],
        dtype=np.float64,
    )
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.divide((left * right).sum(axis=1), denominator, out=np.zeros(len(left)), where=denominator > 1e-12)
    return {
        "spearman": spearman,
        "kendall": kendall,
        "top1_agreement": (top_left[:, 0] == top_right[:, 0]).astype(np.float64),
        "top3_jaccard": top3,
        "sign_agreement": (np.sign(left) == np.sign(right)).mean(axis=1),
        "cosine_similarity": cosine,
    }


def build_pairwise_stability(
    vectors: dict[tuple[str, str, str], np.ndarray],
    model_ids: list[str],
    episode_ids: np.ndarray,
    output_path: Path,
) -> pd.DataFrame:
    writer: pq.ParquetWriter | None = None
    summaries: list[dict] = []
    try:
        for (method, feature_space, value_view), values in vectors.items():
            if values.shape[:2] != (len(model_ids), len(episode_ids)):
                raise ValueError(f"Bad stability array for {(method, feature_space, value_view)}: {values.shape}")
            for left_index, right_index in itertools.combinations(range(len(model_ids)), 2):
                metrics = episode_metrics(values[left_index], values[right_index])
                chunk = pd.DataFrame(
                    {
                        "method": method,
                        "feature_space": feature_space,
                        "value_view": value_view,
                        "model_a": model_ids[left_index],
                        "model_b": model_ids[right_index],
                        "episode_id": episode_ids,
                        **metrics,
                    }
                )
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table)
                summaries.append(
                    {
                        "method": method,
                        "feature_space": feature_space,
                        "value_view": value_view,
                        "model_a": model_ids[left_index],
                        "model_b": model_ids[right_index],
                        **{metric: float(np.mean(metric_values)) for metric, metric_values in metrics.items()},
                    }
                )
    finally:
        if writer is not None:
            writer.close()
    return pd.DataFrame(summaries)


def pair_summary_only(
    values: np.ndarray,
    model_ids: list[str],
    method: str,
    feature_space: str,
    value_view: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for left_index, right_index in itertools.combinations(range(len(model_ids)), 2):
        metrics = episode_metrics(values[left_index], values[right_index])
        rows.append(
            {
                "method": method,
                "feature_space": feature_space,
                "value_view": value_view,
                "model_a": model_ids[left_index],
                "model_b": model_ids[right_index],
                **{metric: float(metric_values.mean()) for metric, metric_values in metrics.items()},
            }
        )
    return pd.DataFrame(rows)


def summarize_pairwise(pair_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in pair_summary.groupby(["method", "feature_space", "value_view"], sort=True):
        method, feature_space, value_view = keys
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            q1, q3 = np.quantile(values, [0.25, 0.75])
            rows.append(
                {
                    "method": method,
                    "feature_space": feature_space,
                    "value_view": value_view,
                    "metric": metric,
                    "model_pairs": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "iqr": float(q3 - q1),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def _weighted_pair_values(group: pd.DataFrame, counts: np.ndarray, model_index: dict[str, int], metric: str) -> np.ndarray:
    weights = np.asarray(
        [counts[model_index[a]] * counts[model_index[b]] for a, b in group[["model_a", "model_b"]].itertuples(index=False, name=None)],
        dtype=np.int64,
    )
    values = group[metric].to_numpy(dtype=np.float64)
    return np.repeat(values, weights)


def model_level_bootstrap(
    pair_summary: pd.DataFrame,
    model_ids: list[str],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    source = pair_summary[
        (pair_summary["feature_space"] == "proxy_concept_v_only") & (pair_summary["value_view"] == "signed")
    ]
    grouped = {method: group.sort_values(["model_a", "model_b"]) for method, group in source.groupby("method")}
    if "ConceptFAN" not in grouped:
        raise ValueError("ConceptFAN reference stability is required for comparison")
    model_index = {model_id: index for index, model_id in enumerate(model_ids)}
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for method in ("integrated_gradients", "gradient_shap"):
        if method not in grouped:
            raise ValueError(f"Missing {method} pair summaries")
        mean_differences = np.empty(repetitions, dtype=np.float64)
        median_differences = np.empty(repetitions, dtype=np.float64)
        for repetition in range(repetitions):
            sampled = rng.integers(0, len(model_ids), size=len(model_ids))
            counts = np.bincount(sampled, minlength=len(model_ids))
            concept_values = _weighted_pair_values(grouped["ConceptFAN"], counts, model_index, "spearman")
            method_values = _weighted_pair_values(grouped[method], counts, model_index, "spearman")
            if len(concept_values) == 0 or len(method_values) == 0:
                mean_differences[repetition] = np.nan
                median_differences[repetition] = np.nan
            else:
                mean_differences[repetition] = method_values.mean() - concept_values.mean()
                median_differences[repetition] = np.median(method_values) - np.median(concept_values)
        for statistic, values in (("difference_in_mean_spearman", mean_differences), ("difference_in_median_spearman", median_differences)):
            values = values[np.isfinite(values)]
            low, high = np.quantile(values, [0.025, 0.975])
            probability = 2.0 * min(float(np.mean(values <= 0.0)), float(np.mean(values >= 0.0)))
            rows.append(
                {
                    "reference_method": "ConceptFAN",
                    "comparison_method": method,
                    "feature_space": "proxy_concept_v_only",
                    "value_view": "signed",
                    "statistic": statistic,
                    "bootstrap_iterations": len(values),
                    "estimate": float(np.mean(values)),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "two_sided_bootstrap_probability": min(probability, 1.0),
                    "advantage_supported": bool(low > 0.0 or high < 0.0),
                }
            )
    return pd.DataFrame(rows)


def _conceptfan_vectors(runs_root: Path, episode_ids: np.ndarray, epsilon: float) -> tuple[list[str], np.ndarray]:
    run_dirs = _checkpoint_dirs(runs_root, "ConceptFAN-NoAlpha")
    matrices: list[np.ndarray] = []
    for run_dir in run_dirs:
        frame = pd.read_parquet(run_dir / "contributions_test.parquet")
        if not np.array_equal(frame["RecordID"].to_numpy(), episode_ids):
            raise ValueError(f"ConceptFAN episode mismatch in {run_dir}")
        matrices.append(frame[[f"contribution_{index}" for index in range(5)]].to_numpy(dtype=np.float64))
    raw = np.stack(matrices)
    normalized = raw / (np.abs(raw).sum(axis=2, keepdims=True) + epsilon)
    return [path.name for path in run_dirs], normalized


def _write_temporal_dataset(cache_dir: Path, model_ids: list[str], episode_ids: np.ndarray, map_frame: pd.DataFrame, output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    time_count = None
    channel_names = map_frame["channel_name"].astype(str).to_numpy()
    for model_id in model_ids:
        with np.load(cache_dir / f"{model_id}.npz", allow_pickle=False) as payload:
            for method in METHODS:
                values = payload[method]
                time_count = values.shape[1]
                for start in range(0, len(episode_ids), 25):
                    stop = min(start + 25, len(episode_ids))
                    chunk = values[start:stop]
                    episodes = stop - start
                    table = pa.table(
                        {
                            "model_id": pa.array(np.repeat(model_id, chunk.size), type=pa.string()),
                            "episode_id": pa.array(np.repeat(episode_ids[start:stop], time_count * len(channel_names))),
                            "method": pa.array(np.repeat(method, chunk.size), type=pa.string()),
                            "time_id": pa.array(np.tile(np.repeat(np.arange(time_count, dtype=np.int16), len(channel_names)), episodes)),
                            "channel_id": pa.array(np.tile(np.arange(len(channel_names), dtype=np.int16), episodes * time_count)),
                            "channel_name": pa.array(np.tile(channel_names, episodes * time_count)),
                            "attribution": pa.array(chunk.reshape(-1)),
                        }
                    )
                    pq.write_table(
                        table,
                        output / f"{model_id}__{method}__{start:04d}.parquet",
                        compression="zstd",
                        use_dictionary=["model_id", "method", "channel_name"],
                    )


def _export_aggregates(
    cache_dir: Path,
    model_ids: list[str],
    episode_ids: np.ndarray,
    map_frame: pd.DataFrame,
    primary_matrix: np.ndarray,
    sensitivity_matrix: np.ndarray,
    epsilon: float,
    data_dir: Path,
) -> tuple[dict[tuple[str, str, str], np.ndarray], pd.DataFrame]:
    channel_rows: list[pd.DataFrame] = []
    proxy_rows: list[pd.DataFrame] = []
    conservation_rows: list[pd.DataFrame] = []
    vector_lists: dict[tuple[str, str, str], list[np.ndarray]] = {}
    channel_names = map_frame["channel_name"].astype(str).to_numpy()
    for model_id in model_ids:
        with np.load(cache_dir / f"{model_id}.npz", allow_pickle=False) as payload:
            for method in METHODS:
                raw = payload[method][None, ...]
                primary = aggregate_attributions(raw, primary_matrix, epsilon)
                sensitivity = aggregate_attributions(raw, sensitivity_matrix, epsilon)
                for value_view, values in (
                    ("signed", primary["signed_channels"]), ("absolute", primary["absolute_channels"])
                ):
                    normalized = values / (np.abs(values).sum(axis=2, keepdims=True) + epsilon)
                    vector_lists.setdefault((method, "full_input_channels", value_view), []).append(normalized[0])
                channel_rows.append(
                    pd.DataFrame(
                        {
                            "model_id": model_id,
                            "episode_id": np.repeat(episode_ids, len(channel_names)),
                            "method": method,
                            "channel_id": np.tile(np.arange(len(channel_names)), len(episode_ids)),
                            "channel_name": np.tile(channel_names, len(episode_ids)),
                            "signed_attribution": primary["signed_channels"].reshape(-1),
                            "absolute_attribution": primary["absolute_channels"].reshape(-1),
                        }
                    )
                )
                for map_name, aggregate in (("v_only", primary), ("vmd", sensitivity)):
                    vector_lists.setdefault((method, f"proxy_concept_{map_name}", "signed"), []).append(aggregate["signed_l1_normalized"][0])
                    vector_lists.setdefault((method, f"proxy_concept_{map_name}", "absolute"), []).append(aggregate["absolute_l1_normalized"][0])
                    proxy_rows.append(
                        pd.DataFrame(
                            {
                                "model_id": model_id,
                                "episode_id": np.repeat(episode_ids, len(PROXY_NAMES)),
                                "method": method,
                                "map": map_name,
                                "proxy_concept": np.tile(PROXY_NAMES, len(episode_ids)),
                                "signed_attribution": aggregate["signed_proxy"].reshape(-1),
                                "absolute_attribution": aggregate["absolute_proxy"].reshape(-1),
                                "signed_l1_normalized": aggregate["signed_l1_normalized"].reshape(-1),
                                "absolute_l1_normalized": aggregate["absolute_l1_normalized"].reshape(-1),
                            }
                        )
                    )
                    conservation_rows.append(
                        pd.DataFrame(
                            {
                                "model_id": model_id,
                                "episode_id": episode_ids,
                                "method": method,
                                "map": map_name,
                                "original_attribution_sum": aggregate["original_sum"][0],
                                "grouped_attribution_sum": aggregate["grouped_sum"][0],
                                "unassigned_attribution_sum": aggregate["signed_unassigned"][0],
                                "unassigned_absolute_fraction": aggregate["absolute_unassigned"][0] / (aggregate["absolute_channels"][0].sum(axis=1) + epsilon),
                                "conservation_error": aggregate["conservation_error"][0],
                            }
                        )
                    )
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(channel_rows, ignore_index=True).to_parquet(data_dir / "posthoc_channel_attributions.parquet", index=False, compression="zstd")
    pd.concat(proxy_rows, ignore_index=True).to_parquet(data_dir / "posthoc_proxy_concept_attributions.parquet", index=False, compression="zstd")
    conservation = pd.concat(conservation_rows, ignore_index=True)
    vectors = {key: np.stack(parts) for key, parts in vector_lists.items()}
    return vectors, conservation


def _fast_pair_spearman(values: np.ndarray) -> np.ndarray:
    ranks = stats.rankdata(values, axis=2)
    ranks -= ranks.mean(axis=2, keepdims=True)
    norms = np.linalg.norm(ranks, axis=2)
    outputs: list[float] = []
    for left, right in itertools.combinations(range(values.shape[0]), 2):
        denominator = norms[left] * norms[right]
        correlations = np.divide(
            (ranks[left] * ranks[right]).sum(axis=1), denominator, out=np.zeros(values.shape[1]), where=denominator > 1e-12
        )
        outputs.append(float(correlations.mean()))
    return np.asarray(outputs)


def mapping_controls(
    vectors_by_method: dict[str, np.ndarray],
    map_frame: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    primary = map_frame["included_in_primary_analysis"].map(_bool).to_numpy()
    group_labels = map_frame.loc[primary, "proxy_concept"].astype(str).to_numpy()
    assigned_indices = np.flatnonzero(primary)
    value_indices = np.flatnonzero(map_frame["channel_type"].eq("V").to_numpy())
    group_sizes = {name: int(np.sum(group_labels == name)) for name in PROXY_NAMES}
    rng = np.random.default_rng(int(config["mapping_controls"]["seed"]))
    rows: list[dict] = []

    def evaluate(control: str, control_index: int, selected: np.ndarray, labels: np.ndarray) -> None:
        matrix = np.zeros((len(PROXY_NAMES), len(map_frame)), dtype=np.float64)
        for proxy_index, proxy in enumerate(PROXY_NAMES):
            matrix[proxy_index, selected[labels == proxy]] = 1.0
        for method, channels in vectors_by_method.items():
            grouped = np.einsum("med,kd->mek", channels, matrix, optimize=True)
            pair_values = _fast_pair_spearman(grouped)
            rows.append(
                {
                    "control": control,
                    "control_index": control_index,
                    "method": method,
                    "model_pairs": len(pair_values),
                    "mean_pair_spearman": float(pair_values.mean()),
                    "median_pair_spearman": float(np.median(pair_values)),
                    "minimum_pair_spearman": float(pair_values.min()),
                    "maximum_pair_spearman": float(pair_values.max()),
                }
            )

    evaluate("true_clinical_proxy_map", 0, assigned_indices, group_labels)
    for index in range(int(config["mapping_controls"]["permutations"])):
        evaluate("permuted_map", index + 1, assigned_indices, rng.permutation(group_labels))
    for index in range(int(config["mapping_controls"]["random_same_size"])):
        selected = rng.choice(value_indices, size=len(assigned_indices), replace=False)
        labels = np.concatenate([np.repeat(name, group_sizes[name]) for name in PROXY_NAMES])
        rng.shuffle(labels)
        evaluate("random_same_size", index + 1, selected, labels)
    return pd.DataFrame(rows)


def _completeness_table(cache_dir: Path, model_ids: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for model_id in model_ids:
        with np.load(cache_dir / f"{model_id}.npz", allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            for baseline, key in (("zero_standardized", "completeness"), ("median_training_episode", "completeness_median")):
                values = payload[key].astype(np.float64)
                rows.append(
                    {
                        "model_id": model_id,
                        "baseline": baseline,
                        "n_steps": metadata["ig_n_steps_used"] if baseline == "zero_standardized" else 64,
                        "mean_absolute_completeness_error": float(values.mean()),
                        "median_absolute_completeness_error": float(np.median(values)),
                        "p95_absolute_completeness_error": float(np.quantile(values, 0.95)),
                        "maximum_absolute_completeness_error": float(values.max()),
                    }
                )
    return pd.DataFrame(rows)


def _sanity_checks(
    run_dirs: list[Path],
    frozen: FrozenInputs,
    data_config: dict,
    config: dict,
    cache_dir: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    determinism_rows: list[dict] = []
    count = int(config["sanity"]["determinism_checkpoints"])
    episodes = int(config["sanity"]["determinism_episodes"])
    background = torch.as_tensor(
        frozen.all_inputs[frozen.train_indices[frozen.background_indices_in_train]], dtype=torch.float32, device=device
    )
    for run_dir in run_dirs[:count]:
        model, _ = load_model_from_checkpoint(run_dir / "checkpoint_best.pt", data_config, device)
        wrapper = LogitWrapper(model).eval()
        x = torch.as_tensor(frozen.all_inputs[frozen.test_indices[:episodes]], dtype=torch.float32, device=device)
        zero = torch.zeros(x.shape[1:], dtype=torch.float32, device=device)
        with np.load(cache_dir / f"{run_dir.name}.npz", allow_pickle=False) as payload:
            cache_metadata = json.loads(str(payload["metadata"].item()))
            references = {
                "integrated_gradients": payload["integrated_gradients"][:episodes],
                "gradient_shap": payload["gradient_shap"][:episodes],
            }
        repeat_ig, _ = integrated_gradients(
            wrapper, x, zero, int(cache_metadata["ig_n_steps_used"]), int(config["integrated_gradients"]["internal_batch_size"])
        )
        repeat_gs = gradient_shap(
            wrapper,
            x,
            background,
            frozen.gs_background_schedule[:episodes],
            frozen.gs_alpha_schedule[:episodes],
            float(config["gradient_shap"]["stdevs"]),
        )
        for method, repeated in (("integrated_gradients", repeat_ig.cpu().numpy()), ("gradient_shap", repeat_gs.cpu().numpy())):
            maximum = float(np.max(np.abs(repeated - references[method])))
            tolerance = float(config["sanity"][f"{method}_tolerance"])
            determinism_rows.append(
                {"check": "determinism", "model_id": run_dir.name, "method": method, "episodes": episodes, "maximum_difference": maximum, "tolerance": tolerance, "passed": maximum < tolerance}
            )

    run_dir = run_dirs[0]
    model, _ = load_model_from_checkpoint(run_dir / "checkpoint_best.pt", data_config, device)
    wrapper = LogitWrapper(model).eval()
    random_episodes = int(config["sanity"]["randomization_episodes"])
    x = torch.as_tensor(frozen.all_inputs[frozen.test_indices[:random_episodes]], dtype=torch.float32, device=device)
    zero = torch.zeros(x.shape[1:], dtype=torch.float32, device=device)
    original, _ = integrated_gradients(wrapper, x, zero, 64, 32)
    torch.manual_seed(20260717)
    model.head.reset_parameters()
    randomized, _ = integrated_gradients(wrapper, x, zero, 64, 32)
    original_np = original.cpu().numpy().reshape(random_episodes, -1)
    randomized_np = randomized.cpu().numpy().reshape(random_episodes, -1)
    denominators = np.linalg.norm(original_np, axis=1) * np.linalg.norm(randomized_np, axis=1)
    cosine = np.divide((original_np * randomized_np).sum(axis=1), denominators, out=np.zeros(random_episodes), where=denominators > 1e-12)
    relative_change = np.linalg.norm(original_np - randomized_np, axis=1) / (np.linalg.norm(original_np, axis=1) + 1e-8)
    randomization = pd.DataFrame(
        [{
            "model_id": run_dir.name,
            "episodes": random_episodes,
            "randomized_module": "mortality_head",
            "mean_cosine_similarity": float(cosine.mean()),
            "mean_relative_l2_change": float(relative_change.mean()),
            "substantially_changed": bool(float(relative_change.mean()) > 0.1),
        }]
    )
    constant = torch.zeros((1,) + tuple(x.shape[1:]), dtype=torch.float32, device=device)
    constant_attr, _ = integrated_gradients(wrapper, constant, zero, 64, 1)
    determinism_rows.append(
        {
            "check": "constant_input",
            "model_id": run_dir.name,
            "method": "integrated_gradients",
            "episodes": 1,
            "maximum_difference": float(constant_attr.abs().max().item()),
            "tolerance": 1e-8,
            "passed": bool(torch.isfinite(constant_attr).all() and constant_attr.abs().max().item() < 1e-8),
        }
    )
    return pd.DataFrame(determinism_rows), randomization


def _plot_outputs(pair_summary: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    selected = pair_summary[
        (pair_summary["feature_space"] == "proxy_concept_v_only") & pair_summary["value_view"].isin(["signed", "absolute"])
    ]
    labels = ["ConceptFAN", "Integrated Gradients", "GradientSHAP"]
    methods = ["ConceptFAN", "integrated_gradients", "gradient_shap"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, view in zip(axes, ("signed", "absolute")):
        series = [selected[(selected["method"] == method) & (selected["value_view"] == view)]["spearman"].to_numpy() for method in methods]
        parts = axis.violinplot(series, showmeans=True, showmedians=True, widths=0.75)
        for body, color in zip(parts["bodies"], ("#2b6f6d", "#b24c3a", "#4878a8")):
            body.set_facecolor(color)
            body.set_alpha(0.7)
        axis.axhline(0.0, color="#444444", linewidth=0.8)
        axis.set_xticks(range(1, 4), labels, rotation=18, ha="right")
        axis.set_title("Signed L1-normalized" if view == "signed" else "Absolute L1-normalized")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Pair-level mean Spearman correlation")
    fig.tight_layout()
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(figures_dir / f"posthoc_stability_distribution.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    signed = selected[selected["value_view"] == "signed"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)
    image = None
    for axis, method, label in zip(axes, methods, labels):
        group = signed[signed["method"] == method]
        model_ids = sorted(set(group["model_a"]) | set(group["model_b"]))
        index = {name: position for position, name in enumerate(model_ids)}
        matrix = np.eye(len(model_ids), dtype=np.float64)
        for row in group.itertuples(index=False):
            left, right = index[row.model_a], index[row.model_b]
            matrix[left, right] = matrix[right, left] = float(row.spearman)
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
        axis.set_title(label)
        axis.set_xlabel("Model index")
        axis.set_ylabel("Model index")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.82, label="Mean Spearman")
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(figures_dir / f"posthoc_pairwise_heatmap.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _file_index(root: Path, exclude: Iterable[Path] = ()) -> list[dict]:
    excluded = {path.resolve() for path in exclude}
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.resolve() not in excluded):
        rows.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def run_audit(
    config_path: Path,
    data_config_path: Path,
    prepared_path: Path,
    runs_root: Path,
    output_dir: Path,
    map_path: Path,
    device_name: str,
    write_map_only: bool = False,
) -> dict:
    config = _load_yaml(config_path)
    data_config = _load_yaml(data_config_path)
    data = load_prepared(prepared_path)
    generated_map = build_channel_map(data, data_config)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    if map_path.exists():
        existing_map = pd.read_csv(map_path)
        if existing_map.fillna("").astype(str).to_csv(index=False) != generated_map.fillna("").astype(str).to_csv(index=False):
            raise ValueError("Committed channel map differs from the frozen source-of-truth derivation")
    else:
        generated_map.to_csv(map_path, index=False)
    map_frame = pd.read_csv(map_path)
    input_dim = int(data.metadata["input_dims"]["V+M+D"])
    validate_channel_map(map_frame, input_dim)
    if write_map_only:
        result = {"status": "CHANNEL_MAP_FROZEN", "path": str(map_path), "sha256": sha256_file(map_path), "channels": len(map_frame)}
        print(json.dumps(result, indent=2))
        return result

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("MANIFESTS", "CONFIG", "TABLES", "DATA", "FIGURES"):
        (output_dir / name).mkdir(exist_ok=True)
    shutil.copy2(config_path, output_dir / "CONFIG" / "posthoc_attribution_stability.yaml")
    shutil.copy2(map_path, output_dir / "CONFIG" / "channel_to_proxy_concept_map.csv")

    frozen = prepare_frozen_inputs(data, config)
    actual_shape = (data.v.shape[1], input_dim)
    expected_shape = (
        int(config["evaluation"]["expected_model_input_time"]), int(config["evaluation"]["expected_model_input_channels"])
    )
    if actual_shape != expected_shape:
        raise ValueError(f"Frozen input shape {actual_shape} differs from configured checkpoint contract {expected_shape}")
    run_dirs = _checkpoint_dirs(runs_root, "PlainTransformer")
    checkpoint_count = int(config["evaluation"]["checkpoint_count"])
    if len(run_dirs) != checkpoint_count:
        raise ValueError(f"Expected {checkpoint_count} PlainTransformer checkpoints, found {len(run_dirs)}")
    checkpoint_rows: list[dict] = []
    cache_dir = runs_root.parent / "posthoc_attribution_cache"
    for position, run_dir in enumerate(run_dirs, start=1):
        metadata = _compute_one_checkpoint(
            run_dir, frozen, data_config, config, cache_dir / f"{run_dir.name}.npz", device
        )
        checkpoint_rows.append({"position": position, "model_id": run_dir.name, "checkpoint": str(run_dir / "checkpoint_best.pt"), **metadata})
        print(json.dumps({"stage": "attribution", "checkpoint": position, "total": len(run_dirs), "model_id": run_dir.name}))
    checkpoint_index = pd.DataFrame(checkpoint_rows)
    if checkpoint_index["parameter_sha256"].nunique() != checkpoint_count:
        raise ValueError("PlainTransformer parameter hashes are not unique")
    checkpoint_index.to_csv(output_dir / "MANIFESTS" / "plain_transformer_checkpoint_sha256.csv", index=False)

    primary_matrix, sensitivity_matrix = map_matrices(map_frame)
    epsilon = float(config["aggregation"]["epsilon"])
    model_ids = [path.name for path in run_dirs]
    vectors, conservation = _export_aggregates(
        cache_dir, model_ids, frozen.test_ids, map_frame, primary_matrix, sensitivity_matrix, epsilon, output_dir / "DATA"
    )
    concept_model_ids, concept_signed = _conceptfan_vectors(runs_root, frozen.test_ids, epsilon)
    if concept_model_ids != model_ids:
        raise ValueError("ConceptFAN and PlainTransformer model IDs are not pair-aligned")
    vectors[("ConceptFAN", "proxy_concept_v_only", "signed")] = concept_signed
    vectors[("ConceptFAN", "proxy_concept_v_only", "absolute")] = np.abs(concept_signed) / (
        np.abs(concept_signed).sum(axis=2, keepdims=True) + epsilon
    )

    pair_summary = build_pairwise_stability(
        vectors, model_ids, frozen.test_ids, output_dir / "TABLES" / "posthoc_pairwise_stability.parquet"
    )
    pair_summary.to_csv(output_dir / "TABLES" / "posthoc_pairwise_stability.csv", index=False)
    summary = summarize_pairwise(pair_summary)
    summary.to_csv(output_dir / "TABLES" / "posthoc_attribution_summary.csv", index=False)
    summary.to_parquet(output_dir / "TABLES" / "posthoc_attribution_summary.parquet", index=False, compression="zstd")
    comparison = model_level_bootstrap(
        pair_summary,
        model_ids,
        int(config["statistics"]["bootstrap_iterations"]),
        int(config["statistics"]["bootstrap_seed"]),
    )
    comparison.to_csv(output_dir / "TABLES" / "posthoc_method_comparison.csv", index=False)

    completeness = _completeness_table(cache_dir, model_ids)
    completeness.to_csv(output_dir / "TABLES" / "posthoc_completeness.csv", index=False)
    median_signed: list[np.ndarray] = []
    median_absolute: list[np.ndarray] = []
    for model_id in model_ids:
        with np.load(cache_dir / f"{model_id}.npz", allow_pickle=False) as payload:
            aggregate = aggregate_attributions(payload["integrated_gradients_median"][None, ...], primary_matrix, epsilon)
            median_signed.append(aggregate["signed_l1_normalized"][0])
            median_absolute.append(aggregate["absolute_l1_normalized"][0])
    median_pair = pd.concat(
        [
            pair_summary_only(np.stack(median_signed), model_ids, "integrated_gradients_median_baseline", "proxy_concept_v_only", "signed"),
            pair_summary_only(np.stack(median_absolute), model_ids, "integrated_gradients_median_baseline", "proxy_concept_v_only", "absolute"),
        ],
        ignore_index=True,
    )
    baseline_sensitivity = summarize_pairwise(median_pair)
    baseline_sensitivity.to_csv(output_dir / "TABLES" / "posthoc_baseline_sensitivity.csv", index=False)
    conservation.to_parquet(output_dir / "TABLES" / "posthoc_conservation.parquet", index=False, compression="zstd")
    determinism, randomization = _sanity_checks(run_dirs, frozen, data_config, config, cache_dir, device)
    determinism.to_csv(output_dir / "TABLES" / "posthoc_determinism_and_constant_controls.csv", index=False)
    randomization.to_csv(output_dir / "TABLES" / "posthoc_randomization_control.csv", index=False)

    signed_channels: dict[str, np.ndarray] = {}
    for method in METHODS:
        method_values = []
        for model_id in model_ids:
            with np.load(cache_dir / f"{model_id}.npz", allow_pickle=False) as payload:
                method_values.append(payload[method].sum(axis=1, dtype=np.float64))
        signed_channels[method] = np.stack(method_values)
    controls = mapping_controls(signed_channels, map_frame, config)
    controls.to_csv(output_dir / "TABLES" / "posthoc_mapping_controls.csv", index=False)

    article = summary[
        (summary["feature_space"] == "proxy_concept_v_only")
        & (summary["value_view"] == "signed")
        & summary["metric"].isin(METRICS)
    ].copy()
    article.to_csv(output_dir / "TABLES" / "posthoc_article_table.csv", index=False)
    _plot_outputs(pair_summary, output_dir / "FIGURES")
    _write_temporal_dataset(cache_dir, model_ids, frozen.test_ids, map_frame, output_dir / "DATA" / "posthoc_episode_attributions.parquet")

    protocol_discrepancy = {
        "requested_template_shape": [
            int(config["evaluation"]["requested_template_time"]), int(config["evaluation"]["requested_template_channels"])
        ],
        "frozen_checkpoint_shape": list(actual_shape),
        "resolution": "The audit used the immutable checkpoint/preprocessing contract [48,119]; no resampling, channel deletion, or retraining was performed.",
        "channel_breakdown": {"V": len(data.variables), "M": len(data.variables), "D": len(data.variables), "STATIC": data.static.shape[1]},
    }
    manifest_path = output_dir / "MANIFESTS" / "posthoc_attribution_manifest.json"
    validation_path = output_dir / "MANIFESTS" / "posthoc_attribution_validation.json"
    finite_cache_attributions = True
    for model_id in model_ids:
        with np.load(cache_dir / f"{model_id}.npz", allow_pickle=False) as payload:
            finite_cache_attributions = finite_cache_attributions and all(np.isfinite(payload[key]).all() for key in METHODS)
    manifest = {
        "status": STATUS,
        "created_utc": utc_now(),
        "scientific_outcome_affects_status": False,
        "target": "mortality_logit",
        "checkpoint_count": len(model_ids),
        "checkpoint_pairs": math.comb(len(model_ids), 2),
        "test_episodes": len(frozen.test_ids),
        "test_episode_ids": frozen.test_ids.astype(int).tolist(),
        "gradient_shap_background_ids": frozen.background_ids.astype(int).tolist(),
        "gradient_shap_background_seed": int(config["gradient_shap"]["background_seed"]),
        "channel_map_sha256": sha256_file(map_path),
        "prepared_sha256": sha256_file(prepared_path),
        "split_sha256": data.metadata["split_sha256"],
        "protocol_discrepancy": protocol_discrepancy,
        "checkpoint_index": checkpoint_rows,
        "finite_cache_attributions": finite_cache_attributions,
        "files": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["files"] = _file_index(output_dir, exclude=(manifest_path, validation_path))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validation = verify_audit(output_dir, runs_root, map_path, validation_path)
    if not validation["passed"]:
        raise RuntimeError(f"Post-hoc read-only verification failed: {validation['failed_checks']}")
    print(json.dumps({"status": STATUS, "output": str(output_dir), "validation": validation_path.as_posix()}, indent=2))
    return manifest


def verify_audit(report_dir: Path, runs_root: Path, map_path: Path, output_json: Path | None = None) -> dict:
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    manifest_path = report_dir / "MANIFESTS" / "posthoc_attribution_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check("technical status is outcome-independent completion", manifest.get("status") == STATUS and not manifest.get("scientific_outcome_affects_status"), manifest.get("status"))
    run_dirs = _checkpoint_dirs(runs_root, "PlainTransformer")
    check("30 PlainTransformer checkpoints", len(run_dirs) == 30, len(run_dirs))
    current_hashes = [sha256_file(path / "checkpoint_best.pt") for path in run_dirs]
    indexed_hashes = [row["checkpoint_sha256"] for row in manifest["checkpoint_index"]]
    check("checkpoint SHA256 index matches source", current_hashes == indexed_hashes, len(indexed_hashes))
    parameter_hashes = [row["parameter_sha256"] for row in manifest["checkpoint_index"]]
    check("30 unique parameter SHA256", len(set(parameter_hashes)) == 30, len(set(parameter_hashes)))
    check("channel map SHA256", sha256_file(map_path) == manifest["channel_map_sha256"], sha256_file(map_path))

    required = [
        "CONFIG/posthoc_attribution_stability.yaml", "CONFIG/channel_to_proxy_concept_map.csv",
        "TABLES/posthoc_attribution_summary.csv", "TABLES/posthoc_pairwise_stability.csv",
        "TABLES/posthoc_method_comparison.csv", "TABLES/posthoc_completeness.csv",
        "TABLES/posthoc_baseline_sensitivity.csv", "TABLES/posthoc_randomization_control.csv",
        "TABLES/posthoc_article_table.csv", "DATA/posthoc_channel_attributions.parquet",
        "DATA/posthoc_proxy_concept_attributions.parquet", "DATA/posthoc_episode_attributions.parquet",
        "FIGURES/posthoc_stability_distribution.svg", "FIGURES/posthoc_stability_distribution.pdf",
        "FIGURES/posthoc_stability_distribution.png", "FIGURES/posthoc_pairwise_heatmap.svg",
        "FIGURES/posthoc_pairwise_heatmap.pdf", "FIGURES/posthoc_pairwise_heatmap.png",
    ]
    for relative in required:
        check(f"required artifact {relative}", (report_dir / relative).exists(), relative)

    pair = pd.read_csv(report_dir / "TABLES" / "posthoc_pairwise_stability.csv")
    pair_counts = pair.groupby(["method", "feature_space", "value_view"]).size()
    check("435 model pairs per exported method/view", bool((pair_counts == 435).all()), pair_counts.to_dict())
    episode_dataset = pads.dataset(report_dir / "DATA" / "posthoc_episode_attributions.parquet", format="parquet")
    expected_rows = 30 * 600 * 2 * 48 * 119
    check("full temporal attribution row count", episode_dataset.count_rows() == expected_rows, episode_dataset.count_rows())
    temporal_files = sorted((report_dir / "DATA" / "posthoc_episode_attributions.parquet").glob("*.parquet"))
    sample = pq.read_table(temporal_files[0], columns=["time_id", "channel_id"])
    extrema = {"time_id": int(sample["time_id"].to_numpy().max()), "channel_id": int(sample["channel_id"].to_numpy().max())}
    check("frozen attribution shape [48,119]", extrema["time_id"] == 47 and extrema["channel_id"] == 118, extrema)
    channel_id_frame = pd.read_parquet(report_dir / "DATA" / "posthoc_channel_attributions.parquet", columns=["episode_id"])
    episode_ids = np.sort(channel_id_frame["episode_id"].unique())
    check("same 600 episode IDs", len(episode_ids) == 600 and np.array_equal(episode_ids, np.sort(np.asarray(manifest["test_episode_ids"]))), len(episode_ids))

    channel = pd.read_parquet(report_dir / "DATA" / "posthoc_channel_attributions.parquet")
    proxy = pd.read_parquet(report_dir / "DATA" / "posthoc_proxy_concept_attributions.parquet")
    finite = np.isfinite(channel[["signed_attribution", "absolute_attribution"]].to_numpy()).all() and np.isfinite(
        proxy[["signed_attribution", "absolute_attribution", "signed_l1_normalized", "absolute_l1_normalized"]].to_numpy()
    ).all()
    check("all aggregate attributions finite", finite, f"channel_rows={len(channel)}, proxy_rows={len(proxy)}")
    check("signed and absolute results present", set(pair["value_view"]) == {"signed", "absolute"}, sorted(pair["value_view"].unique()))
    conservation = pd.read_parquet(report_dir / "TABLES" / "posthoc_conservation.parquet")
    maxima = conservation.groupby("method")["conservation_error"].apply(lambda values: float(np.max(np.abs(values))))
    check("attribution grouping conserves signed sum", maxima.get("integrated_gradients", math.inf) < 1e-6 and maxima.get("gradient_shap", math.inf) < 1e-5, maxima.to_dict())
    completeness = pd.read_csv(report_dir / "TABLES" / "posthoc_completeness.csv")
    zero_completeness = completeness[completeness["baseline"] == "zero_standardized"]
    check("IG completeness evaluated for all checkpoints", len(zero_completeness) == 30, len(zero_completeness))
    check("IG completeness retry policy satisfied", bool((zero_completeness["n_steps"] >= 64).all()), zero_completeness["n_steps"].value_counts().to_dict())
    determinism = pd.read_csv(report_dir / "TABLES" / "posthoc_determinism_and_constant_controls.csv")
    check("determinism and constant controls pass", bool(determinism["passed"].all()), determinism.to_dict(orient="records"))
    randomization = pd.read_csv(report_dir / "TABLES" / "posthoc_randomization_control.csv")
    check("parameter randomization changes attribution", bool(randomization["substantially_changed"].all()), randomization.to_dict(orient="records"))
    comparison = pd.read_csv(report_dir / "TABLES" / "posthoc_method_comparison.csv")
    check("model-level bootstrap has at least 2000 iterations", bool((comparison["bootstrap_iterations"] >= 2000).all()), comparison["bootstrap_iterations"].tolist())
    article = pd.read_csv(report_dir / "TABLES" / "posthoc_article_table.csv")
    summary = pd.read_csv(report_dir / "TABLES" / "posthoc_attribution_summary.csv")
    keys = ["method", "feature_space", "value_view", "metric"]
    merged = article.merge(summary, on=keys, suffixes=("_article", "_summary"), validate="one_to_one")
    numeric_columns = ["mean", "std", "median", "iqr", "ci95_low", "ci95_high", "minimum", "maximum"]
    consistent = all(np.allclose(merged[f"{column}_article"], merged[f"{column}_summary"]) for column in numeric_columns)
    check("article table agrees with summary", consistent and len(merged) == len(article), len(merged))
    check("no forbidden instability PASS status", "SHAP_INSTABILITY_PASS" not in json.dumps(manifest), manifest.get("status"))

    failed = [item["check"] for item in checks if not item["passed"]]
    report = {
        "status": STATUS if not failed else "POSTHOC_ATTRIBUTION_AUDIT_VALIDATION_FAILED",
        "created_utc": utc_now(),
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
    }
    destination = output_json or report_dir / "MANIFESTS" / "posthoc_attribution_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "passed": report["passed"], "failed_checks": failed}, indent=2))
    return report
