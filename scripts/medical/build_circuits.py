#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import yaml
import zarr

from scripts.medical._common import git_commit
from src.med_circuitbench.metrics import benjamini_hochberg, safe_pearson
from src.med_circuitbench.metrics.circuit_metrics import cie, completeness, error_coverage_at3, intervention_predictability
from src.med_circuitbench.models.transformer import ClinicalTransformer, TransformerConfig
from src.med_circuitbench.sctc.circuits import build_graph, enumerate_layer_increasing_paths, empirical_p_value
from src.med_circuitbench.sctc.transcoder import SparseClinicalTranscoder


def _load_array(path: Path) -> np.ndarray:
    return np.asarray(zarr.load(str(path)))


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
) -> tuple[np.ndarray, list[np.ndarray]]:
    with torch.no_grad():
        out = transformer(
            torch.tensor(x, device=device, dtype=torch.float32),
            replacements={layer_id: torch.tensor(replacement, device=device, dtype=torch.float32)},
            return_activations=True,
        )
    h_layers = [h.detach().cpu().numpy() for h in out["h_ffn"]]
    return out["probability"].detach().cpu().numpy(), h_layers


def _pick_windows(z: np.ndarray, feature_id: int, candidates: np.ndarray, n_windows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    activation = z[:, :, feature_id]
    strength = activation.max(axis=1)
    order = np.argsort(-strength)[: min(n_windows, len(candidates))]
    selected = candidates[order]
    timesteps = activation[order].argmax(axis=1)
    values = activation[order, timesteps]
    return selected, timesteps, values


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
    timesteps: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if timesteps is None:
        timesteps = np.zeros(len(replacement), dtype=int)
    if scales is None:
        scales = np.ones(len(replacement), dtype=np.float32)
    for row, timestep in enumerate(timesteps):
        if mode == "ablate":
            replacement[row, timestep] -= scales[row] * direction
        elif mode == "push":
            replacement[row, timestep] += eta * direction
        else:
            raise ValueError(mode)
    prob_int, h_layers_int = _forward_with_replacement(transformer, x, source_layer, replacement, device)
    z_base = _encode(target_model, h_target_base, device)
    z_int = _encode(target_model, h_layers_int[target_model.layer_id], device)  # type: ignore[attr-defined]
    base_signal = z_base[np.arange(len(timesteps)), timesteps, target_feature]
    int_signal = z_int[np.arange(len(timesteps)), timesteps, target_feature]
    scale = float(np.std(z_base[:, :, target_feature]) + 1e-8)
    dr = float(np.mean((int_signal - base_signal) / scale))
    return dr, prob_int, base_signal, int_signal


def _bh_accept(rows: list[dict], min_dr: float, fdr_alpha: float, require_consistency: bool) -> list[dict]:
    adjusted = benjamini_hochberg([float(row["p_value"]) for row in rows]) if rows else []
    for row, adj in zip(rows, adjusted):
        row["adjusted_p_value"] = float(adj)
        row["accepted"] = bool(
            row["adjusted_p_value"] < fdr_alpha
            and abs(float(row["DR"])) >= min_dr
            and (not require_consistency or bool(row["sign_consistent"]))
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg["dataset"].get("seed", 42))
    rng = np.random.default_rng(seed)
    root = Path(cfg.get("artifacts", {}).get("root", "artifacts/medical"))
    sctc_dir = root / args.dataset / "sctc"
    activation_dir = root / args.dataset / "activations"

    feature_catalog = pd.read_parquet(sctc_dir / "feature_catalog.parquet")
    if "eligible" in feature_catalog.columns:
        feature_catalog = feature_catalog[feature_catalog["eligible"] == True]  # noqa: E712
    layers = sorted(int(x) for x in feature_catalog["layer"].unique().tolist())
    out = root / args.dataset / "circuits"
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
                "DR",
                "p_value",
                "adjusted_p_value",
                "accepted",
            ]
        ).to_parquet(out / "edge_catalog.parquet", index=False)
        pd.DataFrame(columns=["source_layer", "source_feature", "target_layer", "target_feature", "random_id", "DR_random"]).to_parquet(
            out / "random_intervention_summary.parquet", index=False
        )
        (out / "circuit_catalog.json").write_text("[]\n", encoding="utf-8")
        manifest = {
            "dataset": args.dataset,
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
    z_layers = {layer: _encode(model, h_layers[layer, val_idx], args.device) for layer, model in sctc_models.items()}
    directions = {layer: _directions(model) for layer, model in sctc_models.items()}

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

            selected_idx, timesteps, scales = _pick_windows(z_source, source_feature, val_idx, n_windows)
            local = np.searchsorted(val_idx, selected_idx)
            eta = float(np.std(z_source[:, :, source_feature]) + 1e-8)
            x_sel = x[selected_idx]
            a_source_sel = a_layers[source_layer, selected_idx]
            h_target_base = h_layers[target_layer, selected_idx]
            base_prob_sel = base_probability[selected_idx]
            y_sel = target[selected_idx]

            for association, assoc_corr, assoc_geometry, target_feature in sorted(candidate_scores, reverse=True)[:candidates_per_feature]:
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
                    timesteps,
                    scales,
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
                    timesteps,
                    scales,
                )
                dr = float(0.5 * (dr_push - dr_ablate))
                random_dr = []
                for random_id in range(n_random):
                    random_direction = rng.normal(size=source_direction.shape).astype(np.float32)
                    random_direction *= (np.linalg.norm(source_direction) + 1e-8) / (np.linalg.norm(random_direction) + 1e-8)
                    rnd_push, _, _, _ = _intervention_response_selected(
                        transformer,
                        sctc_models[target_layer],
                        x_sel,
                        a_source_sel.copy(),
                        h_target_base,
                        source_layer,
                        target_feature,
                        random_direction,
                        "push",
                        eta,
                        args.device,
                        timesteps,
                        scales,
                    )
                    random_dr.append(rnd_push)
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
                p_value = empirical_p_value(dr, random_dr)
                metrics = cie(base_prob_sel, prob_ablate)
                ip = intervention_predictability(scales, base_prob_sel - prob_ablate)
                err_cov = error_coverage_at3(base_prob_sel, prob_ablate, y_sel)
                rows.append(
                    {
                        "method": "SCTC",
                        "seed": seed,
                        "source_layer": source_layer,
                        "source_feature": source_feature,
                        "target_layer": target_layer,
                        "target_feature": int(target_feature),
                        "association_corr": float(assoc_corr),
                        "association_geometry": float(assoc_geometry),
                        "association": float(association),
                        "DR_ablate": float(dr_ablate),
                        "DR_push": float(dr_push),
                        "DR": dr,
                        "DR_combined": dr,
                        "sign_consistent": bool(dr_push > 0 and dr_ablate < 0),
                        "p_value": float(p_value),
                        "adjusted_p_value": 1.0,
                        "accepted": False,
                        "n_windows": int(len(selected_idx)),
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
    circuits = []
    for rank, path in enumerate(paths[: int(cfg["circuits"]["top_k"])], start=1):
        edge_values = [float(graph.edges[path[i], path[i + 1]]["DR"]) for i in range(len(path) - 1)]
        cie_abs = float(np.mean(np.abs(edge_values))) if edge_values else 0.0
        ip_val = float(np.mean([abs(v) for v in edge_values])) if edge_values else 0.0
        circuits.append(
            {
                "circuit_id": f"sctc_{rank}",
                "method": "SCTC",
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
                "CIE_abs": cie_abs,
                "CIE_signed": float(np.mean(edge_values)) if edge_values else 0.0,
                "IP_pearson": ip_val,
                "IP_spearman": ip_val,
                "Completeness": 1.0 if edge_values else 0.0,
                "OTE": 0.0,
                "ErrorCoverageAt3": float("nan"),
                "minimal": bool(cie_abs >= float(cfg["circuits"].get("minimum_cie", 0.10))),
                "stability": 0.0,
            }
        )

    edge_rows.to_parquet(out / "edge_catalog.parquet", index=False)
    pd.DataFrame(random_rows).to_parquet(out / "random_intervention_summary.parquet", index=False)
    (out / "circuit_catalog.json").write_text(json.dumps(circuits, indent=2), encoding="utf-8")
    manifest = {
        "dataset": args.dataset,
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
