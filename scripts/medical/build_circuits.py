#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import yaml
import zarr

from scripts.medical._common import add_run_context_args, git_commit
from src.med_circuitbench.metrics import benjamini_hochberg, safe_pearson
from src.med_circuitbench.metrics.circuit_metrics import cie, completeness, error_coverage_at3, intervention_predictability
from src.med_circuitbench.models.transformer import ClinicalTransformer, TransformerConfig
from src.med_circuitbench.sctc.circuits import build_graph, enumerate_layer_increasing_paths, empirical_p_value
from src.med_circuitbench.sctc.transcoder import SparseClinicalTranscoder


def _load_array(path: Path) -> np.ndarray:
    return np.asarray(zarr.load(str(path)))


def _json_clean(value):
    if isinstance(value, dict):
        return {k: _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def _load_transformer(root: Path, dataset: str, device: str) -> ClinicalTransformer:
    ckpt = torch.load(root / dataset / "transformer" / "model.ckpt", map_location=device)
    model = ClinicalTransformer(TransformerConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _load_sctc(path: Path, device: str) -> SparseClinicalTranscoder:
    ckpt = torch.load(path, map_location=device)
    model = SparseClinicalTranscoder(d_model=int(ckpt["d_model"]), n_features=int(ckpt["n_features"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    setattr(model, "input_kind", ckpt.get("input_kind", "h_ffn"))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _directions(model: SparseClinicalTranscoder) -> tuple[np.ndarray, np.ndarray]:
    encoder = model.encoder[0].weight.detach().cpu().numpy()
    decoder = model.decoder.weight.detach().cpu().numpy().T
    return encoder, decoder


def _encode(model: SparseClinicalTranscoder, h: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(h, device=device, dtype=torch.float32))["z"].detach().cpu().numpy()


def _forward_with_replacement(
    transformer: ClinicalTransformer,
    x: np.ndarray,
    layer_id: int,
    replacement: np.ndarray,
    device: str,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    with torch.no_grad():
        out = transformer(
            torch.tensor(x, device=device, dtype=torch.float32),
            replacements={layer_id: torch.tensor(replacement, device=device, dtype=torch.float32)},
            return_activations=True,
        )
    h_layers = [h.detach().cpu().numpy() for h in out["h_ffn"]]
    a_layers = [a.detach().cpu().numpy() for a in out["a_ffn"]]
    return out["probability"].detach().cpu().numpy(), h_layers, a_layers


def _pick_windows(z: np.ndarray, feature_id: int, candidates: np.ndarray, n_windows: int) -> tuple[np.ndarray, np.ndarray]:
    activation = z[:, :, feature_id]
    strength = activation.mean(axis=1)
    order = np.argsort(-strength)[: min(n_windows, len(candidates))]
    selected = candidates[order]
    values = activation[order]
    return selected, values


def _intervention_response_selected(
    transformer: ClinicalTransformer,
    target_model: SparseClinicalTranscoder,
    x: np.ndarray,
    replacement: np.ndarray,
    h_target_base: np.ndarray,
    source_layer: int,
    target_feature: int,
    direction: np.ndarray,
    mode: str,
    eta: float,
    device: str,
    source_values: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if source_values is None:
        source_values = np.ones(replacement.shape[:2], dtype=np.float32)
    if mode == "ablate":
        replacement -= source_values[:, :, None] * direction
    elif mode == "push":
        replacement += eta * direction
    else:
        raise ValueError(mode)
    prob_int, h_layers_int, a_layers_int = _forward_with_replacement(transformer, x, source_layer, replacement, device)
    z_base = _encode(target_model, h_target_base, device)
    int_input = a_layers_int[target_model.layer_id] if getattr(target_model, "input_kind", "h_ffn") == "a_ffn" else h_layers_int[target_model.layer_id]  # type: ignore[attr-defined]
    z_int = _encode(target_model, int_input, device)
    base_signal = z_base[:, :, target_feature].reshape(-1)
    int_signal = z_int[:, :, target_feature].reshape(-1)
    scale = float(np.std(z_base[:, :, target_feature]) + 1e-8)
    dr = float(np.mean((int_signal - base_signal) / scale))
    return dr, prob_int, base_signal, int_signal


def _random_push_responses(
    transformer: ClinicalTransformer,
    target_model: SparseClinicalTranscoder,
    x: np.ndarray,
    a_source: np.ndarray,
    h_target_base: np.ndarray,
    source_layer: int,
    target_feature: int,
    directions: np.ndarray,
    eta: float,
    device: str,
    source_values: np.ndarray,
    chunk: int = 25,
) -> list[float]:
    z_base = _encode(target_model, h_target_base, device)
    base_signal = z_base[:, :, target_feature].reshape(-1)
    scale = float(np.std(z_base[:, :, target_feature]) + 1e-8)
    out: list[float] = []
    for start in range(0, len(directions), chunk):
        dirs = directions[start : start + chunk].astype(np.float32)
        repeated_x = np.concatenate([x] * len(dirs), axis=0)
        replacement = np.repeat(a_source[None, :, :, :], len(dirs), axis=0)
        for d_idx, direction in enumerate(dirs):
            replacement[d_idx] += eta * direction
        replacement = replacement.reshape(len(dirs) * len(x), *a_source.shape[1:])
        _, h_layers_int, a_layers_int = _forward_with_replacement(transformer, repeated_x, source_layer, replacement, device)
        int_input = a_layers_int[target_model.layer_id] if getattr(target_model, "input_kind", "h_ffn") == "a_ffn" else h_layers_int[target_model.layer_id]  # type: ignore[attr-defined]
        z_int = _encode(target_model, int_input, device).reshape(len(dirs), len(x), int_input.shape[1], -1)
        for d_idx in range(len(dirs)):
            int_signal = z_int[d_idx, :, :, target_feature].reshape(-1)
            out.append(float(np.mean((int_signal - base_signal) / scale)))
    return out


def _forward_chain_intervention(
    transformer: ClinicalTransformer,
    models: dict[int, SparseClinicalTranscoder],
    decoder_directions: dict[int, np.ndarray],
    x: np.ndarray,
    nodes: list[tuple[int, int]],
    mode: str,
    eta_by_node: dict[tuple[int, int], float],
    device: str,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    nodes_by_layer: dict[int, list[int]] = {}
    for layer, feature in nodes:
        nodes_by_layer.setdefault(int(layer), []).append(int(feature))
    with torch.no_grad():
        h = transformer.input_proj(torch.tensor(x, device=device, dtype=torch.float32)) + transformer.pos[:, : x.shape[1]]
        h_layers: list[np.ndarray] = []
        a_layers_out: list[np.ndarray] = []
        for layer_id, layer in enumerate(transformer.layers):
            attn, _ = layer.self_attn(h, h, h, need_weights=False)
            h_layer = layer.norm1(h + layer.dropout(attn))
            a_layer = layer.ffn(h_layer)
            if layer_id in nodes_by_layer:
                model = models[layer_id]
                source = a_layer if getattr(model, "input_kind", "h_ffn") == "a_ffn" else h_layer
                z = model(source)["z"]
                decoder = torch.tensor(decoder_directions[layer_id], device=device, dtype=torch.float32)
                for feature in nodes_by_layer[layer_id]:
                    direction = decoder[feature].view(1, 1, -1)
                    if mode == "ablate":
                        a_layer = a_layer - z[:, :, feature : feature + 1] * direction
                    elif mode == "push":
                        a_layer = a_layer + float(eta_by_node.get((layer_id, feature), 1.0)) * direction
                    else:
                        raise ValueError(mode)
            h = layer.norm2(h_layer + layer.dropout(a_layer))
            h_layers.append(h_layer.detach().cpu().numpy())
            a_layers_out.append(a_layer.detach().cpu().numpy())
        logit = transformer.head(h.mean(dim=1)).squeeze(-1)
        prob = torch.sigmoid(logit)
    return prob.detach().cpu().numpy(), logit.detach().cpu().numpy(), h_layers, a_layers_out


def _chain_strength(
    path: list[tuple[int, int]],
    z_by_layer: dict[int, np.ndarray],
    q95_by_node: dict[tuple[int, int], float],
) -> np.ndarray:
    values = []
    for node in path:
        layer, feature = node
        denom = float(q95_by_node.get(node, 0.0)) + 1e-8
        values.append(np.clip(z_by_layer[layer][:, :, feature].mean(axis=1) / denom, 0.0, 1.0))
    stacked = np.stack(values, axis=0)
    return np.exp(np.mean(np.log(stacked + 1e-6), axis=0)) - 1e-6


def _error_coverage_from_ablate_push(base_prob: np.ndarray, ablate_prob: np.ndarray, push_prob: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = (base_prob >= 0.5).astype(int)
    fp = (pred == 1) & (y == 0)
    fn = (pred == 0) & (y == 1)
    fp_cov = (base_prob - ablate_prob >= 0.05) & fp
    fn_cov = (push_prob - base_prob >= 0.05) & fn
    total = int(fp.sum() + fn.sum())
    return {
        "ErrorCoverageAt3": float((fp_cov.sum() + fn_cov.sum()) / total) if total else float("nan"),
        "FP_CoverageAt3": float(fp_cov.sum() / fp.sum()) if fp.sum() else float("nan"),
        "FN_CoverageAt3": float(fn_cov.sum() / fn.sum()) if fn.sum() else float("nan"),
    }


def _off_target_for_layers(
    models: dict[int, SparseClinicalTranscoder],
    base_h: list[np.ndarray],
    base_a: list[np.ndarray],
    int_h: list[np.ndarray],
    int_a: list[np.ndarray],
    z_std_by_layer: dict[int, np.ndarray],
    chain_nodes: set[tuple[int, int]],
    device: str,
) -> float:
    values = []
    for layer, model in models.items():
        base_input = base_a[layer] if getattr(model, "input_kind", "h_ffn") == "a_ffn" else base_h[layer]
        int_input = int_a[layer] if getattr(model, "input_kind", "h_ffn") == "a_ffn" else int_h[layer]
        z_base = _encode(model, base_input, device)
        z_int = _encode(model, int_input, device)
        std = z_std_by_layer[layer].reshape(1, 1, -1) + 1e-8
        mask = np.ones(z_base.shape[-1], dtype=bool)
        for node_layer, feature in chain_nodes:
            if node_layer == layer and feature < len(mask):
                mask[feature] = False
        if mask.any():
            values.append(np.abs(z_int[:, :, mask] - z_base[:, :, mask]) / std[:, :, mask])
    return float(np.mean(np.concatenate([v.reshape(-1) for v in values]))) if values else float("nan")


def _bh_accept(rows: list[dict], min_dr: float, fdr_alpha: float, require_consistency: bool) -> list[dict]:
    adjusted = benjamini_hochberg([float(row["p_value"]) for row in rows]) if rows else []
    for row, adj in zip(rows, adjusted):
        row["adjusted_p_value"] = float(adj)
        row["accepted"] = bool(
            row["adjusted_p_value"] < fdr_alpha
            and abs(float(row.get("DR_ablation", row["DR"]))) >= min_dr
            and (not require_consistency or bool(row["sign_consistent"]))
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", default="sctc", choices=["sctc", "sae", "linear_probes", "random_directions", "single_features"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_run_context_args(parser)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg["dataset"].get("seed", 42))
    rng = np.random.default_rng(seed)
    root = Path(cfg.get("artifacts", {}).get("root", "artifacts/medical"))
    method = args.method
    sctc_dir = root / args.dataset / method
    activation_dir = root / args.dataset / "activations"

    feature_catalog = pd.read_parquet(sctc_dir / "feature_catalog.parquet")
    if "eligible" in feature_catalog.columns:
        feature_catalog = feature_catalog[feature_catalog["eligible"] == True]  # noqa: E712
    layers = sorted(int(x) for x in feature_catalog["layer"].unique().tolist())
    out = root / args.dataset / f"{method}_circuits"
    out.mkdir(parents=True, exist_ok=True)
    if len(layers) < 2:
        pd.DataFrame(
            columns=[
                "method",
                "seed",
                "source_layer",
                "source_feature",
                "target_layer",
                "target_feature",
                "association",
                "association_corr",
                "association_geometry",
                "DR",
                "DR_ablate",
                "DR_ablation",
                "DR_push",
                "p_value",
                "adjusted_p_value",
                "accepted",
                "n_windows",
                "n_random_directions",
                "runtime_seconds",
            ]
        ).to_parquet(out / "edge_catalog.parquet", index=False)
        pd.DataFrame(columns=["source_layer", "source_feature", "target_layer", "target_feature", "random_id", "DR_random"]).to_parquet(
            out / "random_intervention_summary.parquet", index=False
        )
        (out / "circuit_catalog.json").write_text("[]\n", encoding="utf-8")
        manifest = {
            "dataset": args.dataset,
            "method": method,
            "commit": git_commit(),
            "seed": seed,
            "status": "NO_GO_LESS_THAN_TWO_SCTC_LAYERS",
            "intervention_method": "forward_replacement",
            "candidate_edges": 0,
            "accepted_edges": 0,
            "circuits": 0,
            "files": {
                "edge_catalog": "edge_catalog.parquet",
                "random_intervention_summary": "random_intervention_summary.parquet",
                "circuit_catalog": "circuit_catalog.json",
            },
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(manifest)
        return

    h_layers = _load_array(activation_dir / "h_ffn").astype(np.float32)
    a_layers = _load_array(activation_dir / "a_ffn").astype(np.float32)
    x = _load_array(activation_dir / "x").astype(np.float32)
    split = _load_array(activation_dir / "split")
    target = _load_array(activation_dir / "target").astype(np.int64)
    base_probability = _load_array(activation_dir / "probability").astype(np.float32)
    val_idx = np.where(split == 1)[0]
    if len(val_idx) == 0:
        raise SystemExit("validation split is empty")

    top_features = int(cfg["interventions"]["top_features"])
    candidates_per_feature = int(cfg["interventions"]["candidates_per_feature"])
    n_windows = int(cfg["interventions"]["high_activation_windows"])
    n_random = int(cfg["interventions"]["random_directions"])
    min_dr = float(cfg["interventions"].get("minimum_abs_dr", cfg["interventions"].get("minimum_dr", 0.10)))
    fdr_alpha = float(cfg["interventions"]["fdr_alpha"])
    require_consistency = bool(cfg["interventions"].get("require_ablate_push_consistency", False))

    transformer = _load_transformer(root, args.dataset, args.device)
    sctc_models = {layer: _load_sctc(sctc_dir / "checkpoints" / f"layer_{layer}.ckpt", args.device) for layer in layers}
    for layer, model in sctc_models.items():
        setattr(model, "layer_id", layer)
    z_layers = {}
    for layer, model in sctc_models.items():
        source_tensor = a_layers[layer, val_idx] if getattr(model, "input_kind", "h_ffn") == "a_ffn" else h_layers[layer, val_idx]
        z_layers[layer] = _encode(model, source_tensor, args.device)
    directions = {layer: _directions(model) for layer, model in sctc_models.items()}
    train_idx = np.where(split == 0)[0]
    z_train_layers = {}
    for layer, model in sctc_models.items():
        idx = train_idx if len(train_idx) else val_idx
        source_tensor = a_layers[layer, idx] if getattr(model, "input_kind", "h_ffn") == "a_ffn" else h_layers[layer, idx]
        z_train_layers[layer] = _encode(model, source_tensor, args.device)
    z_std_by_layer = {layer: np.std(z.reshape(-1, z.shape[-1]), axis=0) + 1e-8 for layer, z in z_train_layers.items()}
    q95_by_node: dict[tuple[int, int], float] = {}
    eta_by_node: dict[tuple[int, int], float] = {}
    for layer, z in z_train_layers.items():
        mean_z = z.mean(axis=1)
        for feature in range(z.shape[-1]):
            q95_by_node[(layer, feature)] = float(np.quantile(mean_z[:, feature], 0.95))
            eta_by_node[(layer, feature)] = float(np.std(z[:, :, feature]) + 1e-8)

    rows: list[dict] = []
    random_rows: list[dict] = []
    for source_layer, target_layer in zip(layers, layers[1:]):
        source_frame = feature_catalog[feature_catalog.layer == source_layer].sort_values("mean_activation", ascending=False)
        target_frame = feature_catalog[feature_catalog.layer == target_layer].sort_values("mean_activation", ascending=False)
        source_features = source_frame.head(top_features)["feature_id"].astype(int).tolist()
        target_features = target_frame.head(max(top_features, candidates_per_feature))["feature_id"].astype(int).tolist()
        target_encoder, _ = directions[target_layer]
        _, source_decoder = directions[source_layer]
        z_source = z_layers[source_layer]
        z_target = z_layers[target_layer]

        for source_feature in source_features:
            source_direction = source_decoder[source_feature].astype(np.float32)
            candidate_scores = []
            for target_feature in target_features:
                target_direction = target_encoder[target_feature].astype(np.float32)
                assoc_corr = abs(safe_pearson(z_source[:, :, source_feature].reshape(-1), z_target[:, :, target_feature].reshape(-1)))
                assoc_geometry = abs(
                    float(
                        np.dot(source_direction, target_direction)
                        / ((np.linalg.norm(source_direction) + 1e-12) * (np.linalg.norm(target_direction) + 1e-12))
                    )
                )
                candidate_scores.append((max(assoc_corr, assoc_geometry), assoc_corr, assoc_geometry, target_feature))

            selected_idx, source_values = _pick_windows(z_source, source_feature, val_idx, n_windows)
            eta = float(np.std(z_source[:, :, source_feature]) + 1e-8)
            x_sel = x[selected_idx]
            a_source_sel = a_layers[source_layer, selected_idx]
            h_target_base = (
                a_layers[target_layer, selected_idx]
                if getattr(sctc_models[target_layer], "input_kind", "h_ffn") == "a_ffn"
                else h_layers[target_layer, selected_idx]
            )
            base_prob_sel = base_probability[selected_idx]
            y_sel = target[selected_idx]

            for association, assoc_corr, assoc_geometry, target_feature in sorted(candidate_scores, reverse=True)[:candidates_per_feature]:
                edge_started = time.perf_counter()
                dr_ablate, prob_ablate, _, _ = _intervention_response_selected(
                    transformer,
                    sctc_models[target_layer],
                    x_sel,
                    a_source_sel.copy(),
                    h_target_base,
                    source_layer,
                    target_feature,
                    source_direction,
                    "ablate",
                    eta,
                    args.device,
                    source_values,
                )
                dr_push, prob_push, _, _ = _intervention_response_selected(
                    transformer,
                    sctc_models[target_layer],
                    x_sel,
                    a_source_sel.copy(),
                    h_target_base,
                    source_layer,
                    target_feature,
                    source_direction,
                    "push",
                    eta,
                    args.device,
                    source_values,
                )
                dr = float(0.5 * (dr_push - dr_ablate))
                random_dr = []
                rejection_reason = ""
                if abs(dr_ablate) >= min_dr:
                    random_directions = rng.normal(size=(n_random, source_direction.shape[0])).astype(np.float32)
                    random_directions *= (np.linalg.norm(source_direction) + 1e-8) / (np.linalg.norm(random_directions, axis=1, keepdims=True) + 1e-8)
                    random_dr = _random_push_responses(
                        transformer,
                        sctc_models[target_layer],
                        x_sel,
                        a_source_sel.copy(),
                        h_target_base,
                        source_layer,
                        target_feature,
                        random_directions,
                        eta,
                        args.device,
                        source_values,
                    )
                    for random_id, rnd_push in enumerate(random_dr):
                        random_rows.append(
                            {
                                "source_layer": source_layer,
                                "source_feature": source_feature,
                                "target_layer": target_layer,
                                "target_feature": target_feature,
                                "random_id": random_id,
                                "DR_random": float(rnd_push),
                            }
                        )
                    p_value = empirical_p_value(dr_ablate, random_dr)
                else:
                    rejection_reason = "below_min_dr_skip_random_null"
                    p_value = 1.0
                metrics = cie(base_prob_sel, prob_ablate)
                ip = intervention_predictability(source_values.mean(axis=1), base_prob_sel - prob_ablate)
                err_cov = error_coverage_at3(base_prob_sel, prob_ablate, y_sel)
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "source_layer": source_layer,
                        "source_feature": source_feature,
                        "target_layer": target_layer,
                        "target_feature": int(target_feature),
                        "association_corr": float(assoc_corr),
                        "association_geometry": float(assoc_geometry),
                        "association": float(association),
                        "DR_ablate": float(dr_ablate),
                        "DR_ablation": float(dr_ablate),
                        "DR_push": float(dr_push),
                        "DR": dr,
                        "DR_combined": dr,
                        "sign_consistent": bool(dr_push > 0 and dr_ablate < 0),
                        "p_value": float(p_value),
                        "adjusted_p_value": 1.0,
                        "accepted": False,
                        "rejection_reason": rejection_reason,
                        "n_windows": int(len(selected_idx)),
                        "n_random_directions": int(len(random_dr)),
                        "runtime_seconds": float(time.perf_counter() - edge_started),
                        "mean_probability_delta": float(np.mean(base_prob_sel - prob_ablate)),
                        "CIE_abs": metrics["CIE_abs"],
                        "CIE_signed": metrics["CIE_signed"],
                        "IP_pearson": ip["IP_pearson"],
                        "IP_spearman": ip["IP_spearman"],
                        "Completeness": completeness(metrics["CIE_abs"], metrics["CIE_abs"]),
                        "OTE": float(np.mean(np.abs(prob_push - prob_ablate))),
                        **err_cov,
                    }
                )

    rows = _bh_accept(rows, min_dr, fdr_alpha, require_consistency)
    edge_rows = pd.DataFrame(rows)
    accepted_edges = edge_rows[edge_rows["accepted"]].to_dict("records") if len(edge_rows) else []
    graph = build_graph(
        [
            type(
                "Edge",
                (),
                {
                    "source_layer": int(row["source_layer"]),
                    "source_feature": int(row["source_feature"]),
                    "target_layer": int(row["target_layer"]),
                    "target_feature": int(row["target_feature"]),
                    "dr": float(row["DR"]),
                    "adjusted_p_value": float(row["adjusted_p_value"]),
                    "accepted": bool(row["accepted"]),
                },
            )()
            for row in rows
        ]
    )
    paths = enumerate_layer_increasing_paths(graph, min_edges=int(cfg["circuits"].get("minimum_edges", 2)))
    paths = paths[: int(cfg["circuits"].get("maximum_candidate_paths", 500))]
    all_nodes = sorted({(int(row["source_layer"]), int(row["source_feature"])) for row in accepted_edges} | {(int(row["target_layer"]), int(row["target_feature"])) for row in accepted_edges})
    circuits = []
    for rank, path in enumerate(paths[: int(cfg["circuits"]["top_k"])], start=1):
        path = [(int(layer), int(feature)) for layer, feature in path]
        strength_all = _chain_strength(path, z_layers, q95_by_node)
        order = np.argsort(-strength_all)[: min(n_windows, len(val_idx))]
        selected_idx = val_idx[order]
        chain_strength_selected = strength_all[order]
        x_sel = x[selected_idx]
        y_sel = target[selected_idx]
        base_prob_sel = base_probability[selected_idx]
        with torch.no_grad():
            base_out = transformer(torch.tensor(x_sel, device=args.device, dtype=torch.float32), return_activations=True)
        base_h_layers = [h.detach().cpu().numpy() for h in base_out["h_ffn"]]
        base_a_layers = [a.detach().cpu().numpy() for a in base_out["a_ffn"]]
        prob_ablate, logit_ablate, h_ablate, a_ablate = _forward_chain_intervention(
            transformer, sctc_models, {layer: directions[layer][1] for layer in layers}, x_sel, path, "ablate", eta_by_node, args.device
        )
        prob_push, logit_push, _, _ = _forward_chain_intervention(
            transformer, sctc_models, {layer: directions[layer][1] for layer in layers}, x_sel, path, "push", eta_by_node, args.device
        )
        metrics = cie(base_prob_sel, prob_ablate)
        ip = intervention_predictability(chain_strength_selected, base_prob_sel - prob_ablate)
        err_cov = _error_coverage_from_ablate_push(base_prob_sel, prob_ablate, prob_push, y_sel)
        all_prob_ablate, _, _, _ = _forward_chain_intervention(
            transformer, sctc_models, {layer: directions[layer][1] for layer in layers}, x_sel, all_nodes, "ablate", eta_by_node, args.device
        )
        all_effect = float(np.mean(np.abs(all_prob_ablate - base_prob_sel)))
        ote = _off_target_for_layers(sctc_models, base_h_layers, base_a_layers, h_ablate, a_ablate, z_std_by_layer, set(path), args.device)
        node_drop_effects = []
        for node in path:
            reduced = [p for p in path if p != node]
            reduced_prob, _, _, _ = _forward_chain_intervention(
                transformer, sctc_models, {layer: directions[layer][1] for layer in layers}, x_sel, reduced, "ablate", eta_by_node, args.device
            )
            node_drop_effects.append(float(np.mean(np.abs(reduced_prob - base_prob_sel))))
        minimal = bool(all(effect <= 0.9 * metrics["CIE_abs"] for effect in node_drop_effects)) if node_drop_effects else False
        edge_values = [float(graph.edges[path[i], path[i + 1]]["DR"]) for i in range(len(path) - 1)]
        circuits.append(
            {
                "circuit_id": f"{method}_{rank}",
                "method": method,
                "seed": seed,
                "rank": rank,
                "nodes": [{"layer": int(layer), "feature_id": int(feature)} for layer, feature in path],
                "edges": [
                    {
                        "source": {"layer": int(path[i][0]), "feature_id": int(path[i][1])},
                        "target": {"layer": int(path[i + 1][0]), "feature_id": int(path[i + 1][1])},
                        "DR": float(edge_values[i]),
                    }
                    for i in range(len(edge_values))
                ],
                "CIE_abs": metrics["CIE_abs"],
                "CIE_signed": metrics["CIE_signed"],
                "IP_pearson": ip["IP_pearson"],
                "IP_spearman": ip["IP_spearman"],
                "Completeness": completeness(metrics["CIE_abs"], all_effect),
                "OTE": ote,
                **err_cov,
                "minimal": minimal,
                "stability": 0.0,
                "n_windows": int(len(selected_idx)),
                "intervention_method": "sequential_forward_chain_ablation",
                "base_probability_mean": float(np.mean(base_prob_sel)),
                "intervened_probability_mean": float(np.mean(prob_ablate)),
                "base_logit_mean": float(np.mean(np.log(base_prob_sel / (1.0 - base_prob_sel + 1e-8) + 1e-8))),
                "intervened_logit_mean": float(np.mean(logit_ablate)),
                "chain_strength_mean": float(np.mean(chain_strength_selected)),
                "node_drop_cie": node_drop_effects,
            }
        )

    edge_rows.to_parquet(out / "edge_catalog.parquet", index=False)
    pd.DataFrame(random_rows).to_parquet(out / "random_intervention_summary.parquet", index=False)
    (out / "circuit_catalog.json").write_text(json.dumps(_json_clean(circuits), indent=2, allow_nan=False), encoding="utf-8")
    manifest = {
        "dataset": args.dataset,
        "method": method,
        "commit": git_commit(),
        "seed": seed,
        "status": "PASS" if circuits else "NO_GO_NO_ACCEPTED_CIRCUIT",
        "intervention_method": "forward_replacement",
        "candidate_edges": int(len(edge_rows)),
        "accepted_edges": int(edge_rows["accepted"].sum()) if len(edge_rows) else 0,
        "circuits": len(circuits),
        "files": {
            "edge_catalog": "edge_catalog.parquet",
            "random_intervention_summary": "random_intervention_summary.parquet",
            "circuit_catalog": "circuit_catalog.json",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print({"edge_catalog": str(out / "edge_catalog.parquet"), "circuit_catalog": str(out / "circuit_catalog.json"), **manifest})


if __name__ == "__main__":
    main()
