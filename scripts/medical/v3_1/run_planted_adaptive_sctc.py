#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fan.sctc.adaptive import AdaptiveSCTCTrainConfig, layer_specific_capacity, train_adaptive_sctc
from fan.sctc.evaluation import fidelity_metrics
from med_circuitbench.planted.model import NODE_LAYERS, STATE_NAMES, PlantedCircuitModel
from scripts.medical.v2.run_v2_program import make_episodes, split_frame, subset_cols
from scripts.medical.v3.run_research_program import DEVICE, prepare_arrays, sctc_fidelity_predictions


def flatten_layer(model: PlantedCircuitModel, states: np.ndarray, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        out = model(torch.from_numpy(states).float().to(DEVICE))
    return (
        out.layers[layer].detach().cpu().numpy(),
        out.logit.detach().cpu().numpy(),
        out.probability.detach().cpu().numpy(),
        out.nodes.detach().cpu().numpy(),
    )


def node_matching(model: PlantedCircuitModel, transcoder, activation: np.ndarray, nodes: np.ndarray, layer: int, seed: int) -> pd.DataFrame:
    with torch.no_grad():
        z = transcoder(torch.from_numpy(activation).float().to(DEVICE))["z"].detach().cpu().numpy().reshape(-1, transcoder.n_features)
    decoder = transcoder.decoder.weight.detach().cpu().T.numpy()
    rows = []
    layer_nodes = [name for name, node_layer in NODE_LAYERS.items() if node_layer == layer]
    flat_nodes = nodes.reshape(-1, nodes.shape[-1])
    for feature_id in range(transcoder.n_features):
        for node in layer_nodes:
            idx = STATE_NAMES.index(node)
            target = flat_nodes[:, idx]
            corr = 0.0 if np.std(z[:, feature_id]) == 0 or np.std(target) == 0 else abs(float(stats.pearsonr(z[:, feature_id], target).statistic))
            direction = model.directions[idx].detach().cpu().numpy()
            cosine = abs(float(np.dot(decoder[feature_id], direction) / ((np.linalg.norm(decoder[feature_id]) + 1e-8) * (np.linalg.norm(direction) + 1e-8))))
            rows.append(
                {
                    "seed": seed,
                    "layer": layer,
                    "feature_id": feature_id,
                    "node": node,
                    "activation_correlation": corr,
                    "decoder_cosine": cosine,
                    "accepted": bool(corr >= 0.30 or cosine >= 0.30),
                }
            )
    return pd.DataFrame(rows)


def run_seed(seed: int, cfg: dict, method_cfg: dict, output: Path, max_epochs: int | None = None, limit_layers: int | None = None, limit_candidates: int | None = None) -> pd.DataFrame:
    frame = make_episodes(seed, cfg, "clean")
    arrays = prepare_arrays(split_frame(frame, seed), subset_cols("full_input"))
    model = PlantedCircuitModel(seed=seed, d_model=int(cfg["model"]["d_model"])).to(DEVICE)
    rows = []
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    layers = list(range(4))
    if limit_layers is not None:
        layers = layers[: int(limit_layers)]
    for layer in layers:
        train_act, train_logit, _, _ = flatten_layer(model, arrays["c_train_seq"], layer)
        val_act, val_logit, val_prob, val_nodes = flatten_layer(model, arrays["c_val_seq"], layer)
        rank_capacity = layer_specific_capacity(
            torch.from_numpy(train_act).float(),
            multiplier=float(method_cfg["adaptive_sctc"]["layer_specific_capacity"]["effective_rank_multiplier"][1]),
            minimum=int(method_cfg["adaptive_sctc"]["layer_specific_capacity"]["min_features"]),
            maximum=int(method_cfg["adaptive_sctc"]["layer_specific_capacity"]["max_features"]),
        )
        registered = method_cfg["adaptive_sctc"]["planted_grid"]
        candidates = sorted({int(item["n_features"]) for item in registered} | {rank_capacity})
        if limit_candidates is not None:
            candidates = candidates[: int(limit_candidates)]
        for n_features in candidates:
            target_top_k = min(max(8, n_features // 4), 16)

            def forward_from_layer(repl: torch.Tensor, layer: int = layer) -> torch.Tensor:
                return model.downstream_from_layer(layer, repl).logit.max(dim=1).values

            train_cfg = AdaptiveSCTCTrainConfig(
                n_features=n_features,
                target_top_k=target_top_k,
                epochs=int(max_epochs or method_cfg["adaptive_sctc"]["epochs"]["max"]),
                patience=int(method_cfg["adaptive_sctc"]["epochs"]["early_stopping_patience"]),
                resample_every_steps=int(method_cfg["adaptive_sctc"]["dead_feature_resampling"]["every_steps"]),
            )
            transcoder, log = train_adaptive_sctc(
                torch.from_numpy(train_act).float(),
                torch.from_numpy(train_logit.max(axis=1)).float(),
                forward_from_layer,
                train_cfg,
                DEVICE,
            )
            ckpt = seed_dir / f"layer{layer}_{n_features}_adaptive.pt"
            torch.save({"model_state_dict": transcoder.state_dict(), "seed": seed, "layer": layer, "n_features": n_features, "top_k": target_top_k}, ckpt)
            log.to_csv(seed_dir / f"layer{layer}_{n_features}_adaptive_training.csv", index=False)
            dummy = np.zeros((val_act.shape[0], val_act.shape[1], 1), dtype=np.float32)
            pred = sctc_fidelity_predictions(
                transcoder,
                val_act,
                dummy,
                arrays["y_val"],
                val_logit.max(axis=1),
                val_prob.max(axis=1),
                lambda _x, repl, layer=layer: model.downstream_from_layer(layer, repl).logit.max(dim=1).values,
            )
            pred.to_parquet(seed_dir / f"layer{layer}_{n_features}_adaptive_predictions.parquet", index=False)
            met = fidelity_metrics(pred)
            with torch.no_grad():
                z = transcoder(torch.from_numpy(val_act).float().to(DEVICE))["z"].detach().cpu()
            matches = node_matching(model, transcoder, val_act, val_nodes, layer, seed)
            matches.to_parquet(seed_dir / f"layer{layer}_{n_features}_adaptive_node_matching.parquet", index=False)
            rows.append(
                {
                    "seed": seed,
                    "layer": layer,
                    "n_features": n_features,
                    "top_k": target_top_k,
                    "effective_rank_capacity": rank_capacity,
                    "L0_per_token": float((z > 0).float().sum(dim=-1).mean().item()),
                    "dead_feature_fraction": float(((z > 0).float().mean(dim=(0, 1)) < 1e-5).float().mean().item()),
                    "node_match_count": int(matches["accepted"].sum()),
                    "checkpoint": str(ckpt),
                    **met,
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medical/v3/full.yaml")
    parser.add_argument("--method-config", default="configs/medical/v3_1/method_improvements.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", default="artifacts/medical/v3_1/planted_adaptive_sctc")
    parser.add_argument("--max-epochs", type=int, help="Runtime override for smoke checks; full registered run omits this.")
    parser.add_argument("--limit-layers", type=int, help="Runtime override for smoke checks; full registered run omits this.")
    parser.add_argument("--limit-candidates", type=int, help="Runtime override for smoke checks; full registered run omits this.")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    method_cfg = yaml.safe_load(Path(args.method_config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frames = [run_seed(seed, cfg, method_cfg, output, args.max_epochs, args.limit_layers, args.limit_candidates) for seed in args.seeds]
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output / "planted_adaptive_sctc_grid.csv", index=False)
    gate = {
        "status": "PLANTED_ADAPTIVE_PASS" if ((result["dead_feature_fraction"] < 0.5) & (result["delta_AUPRC"] <= 0.01)).any() else "PLANTED_ADAPTIVE_NEGATIVE",
        "best_dead_feature_fraction": float(result["dead_feature_fraction"].min()),
        "best_delta_AUPRC": float(result["delta_AUPRC"].min()),
        "config": str(args.method_config),
    }
    (output / "planted_adaptive_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
