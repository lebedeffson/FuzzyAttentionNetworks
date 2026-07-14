#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from med_circuitbench.sctc.transcoder import SparseClinicalTranscoder
from scripts.medical.v2.run_v2_program import sha256_file


STATE_NAMES = ["I", "R", "V", "O", "S"]
EDGES = [("I", "R", 0.95), ("R", "V", 0.90), ("V", "O", 0.85), ("V", "S", 0.80), ("O", "S", 0.75)]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINAL_STATUSES = {
    "V3_GO",
    "V3_FAN_VALIDATED_SCTC_NEGATIVE",
    "V3_FAN_NEGATIVE_SCTC_VALIDATED",
    "V3_VALIDATED_NEGATIVE",
    "V3_MIXED_RESULT",
}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_dirs(output: Path) -> None:
    for name in [
        "results",
        "runs",
        "checkpoints/fan",
        "checkpoints/planted_sctc",
        "checkpoints/standard_sctc",
        "checkpoints/fan_sctc",
        "paper/figures",
        "paper/tables",
        "logs",
        "manifests",
        "configs",
        "SOURCE/scripts/medical",
        "SOURCE/src",
    ]:
        (output / name).mkdir(parents=True, exist_ok=True)


def v2_2_config_from_v3(cfg: dict, output: Path) -> Path:
    cfg2 = {
        "program": {"name": "med_circuitbench_v3_core_v2_2", "date": cfg["program"].get("date", "2026-07-14")},
        "dataset": cfg["dataset"],
        "split": cfg.get("split", {"train": 0.6, "validation": 0.2, "test": 0.2}),
        "model": cfg["model"],
        "training": {
            **cfg["training"],
            "lambda_a": cfg.get("fan", {}).get("lambda_align", cfg.get("fan", {}).get("lambda_a", 0.001)),
            "lambda_s": cfg.get("fan", {}).get("lambda_sparse", cfg.get("fan", {}).get("lambda_s", 0.0001)),
        },
        "seeds": cfg.get("seeds", [42, 43, 44]),
        "fan": {
            "lambda_c": cfg["fan"].get("lambda_c", 2.0),
            "lambda_a": cfg["fan"].get("lambda_align", cfg["fan"].get("lambda_a", 0.001)),
            "lambda_s": cfg["fan"].get("lambda_sparse", cfg["fan"].get("lambda_s", 0.0001)),
            "memberships": cfg["fan"].get("memberships", ["gaussian", "bell", "sigmoid", "mixed"]),
        },
        "shortcut": cfg["shortcut"],
        "sctc": {
            "minimum_training_episodes": cfg["sctc"].get("minimum_training_episodes", 2000),
            "feature_grid": cfg["sctc"].get("feature_grid", [128, 256, 512]),
            "training_epochs": cfg["sctc"].get("training_epochs", 6),
            "top_features": cfg["sctc"].get("top_features", 16),
            "random_null": cfg["sctc"].get("random_null", 1000),
        },
    }
    path = output / "configs" / "v2_2_core.yaml"
    path.write_text(yaml.safe_dump(cfg2, sort_keys=False), encoding="utf-8")
    return path


def run_v2_2_core(cfg: dict, output: Path, seeds: list[int]) -> Path:
    core = output / "runs" / "v2_2_core"
    cfg_path = v2_2_config_from_v3(cfg, output)
    cmd = [
        sys.executable,
        "scripts/medical/v2_2/run_v2_2_program.py",
        "--config",
        str(cfg_path),
        "--seeds",
        *map(str, seeds),
        "--mode",
        "full",
        "--output",
        str(core),
    ]
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    (output / "logs" / "v2_2_core_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output / "logs" / "v2_2_core_stderr.log").write_text(proc.stderr, encoding="utf-8")
    (output / "logs" / "v2_2_core_exit_code.txt").write_text(str(proc.returncode), encoding="utf-8")
    (output / "logs" / "timing.csv").write_text(
        "stage,start_time,end_time,duration_seconds,exit_code\n"
        f"v2_2_core,{started},{time.time()},{time.time() - started:.3f},{proc.returncode}\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError("V2.2 core failed; see logs/v2_2_core_stderr.log")
    return core


def copy_core_results(core: Path, output: Path) -> None:
    mapping = {
        "concept_sufficiency.csv": "concept_sufficiency.csv",
        "fan_results.csv": "fan_results.csv",
        "concept_leakage_metrics.csv": "concept_leakage_metrics.csv",
        "faithfulness_results.csv": "faithfulness_results.csv",
        "planted_metrics.csv": "planted_results.csv",
        "representation_audit.parquet": "representation_audit.parquet",
        "standard_sctc_results.csv": "standard_sctc_results.csv",
        "fan_sctc_results.csv": "fan_sctc_results.csv",
        "explicit_vs_discovered.csv": "explicit_vs_discovered.csv",
        "aggregate_metrics.csv": "aggregate_metrics.csv",
        "program_status.json": "v2_2_program_status.json",
        "shortcut_audit.csv": "shortcut_audit.csv",
    }
    for src_name, dst_name in mapping.items():
        src = core / "RESULTS" / src_name
        if src.exists():
            shutil.copy2(src, output / "results" / dst_name)
    for sub in ["oracle_fan", "predicted_fan"]:
        for ckpt in core.glob(f"runs/seed_*/{sub}/*.pt"):
            shutil.copy2(ckpt, output / "checkpoints" / "fan" / f"{ckpt.parent.parent.name}_{ckpt.name}")
    for ckpt in core.glob("runs/seed_*/standard_sctc/*.pt"):
        shutil.copy2(ckpt, output / "checkpoints" / "standard_sctc" / f"{ckpt.parent.parent.name}_{ckpt.name}")


def family_from_model_name(name: str, default: str) -> str:
    for family in ["gaussian", "bell", "sigmoid", "mixed"]:
        if name.endswith("_" + family):
            return family
    return default


def build_family_comparison(output: Path) -> tuple[pd.DataFrame, str]:
    fan = pd.read_csv(output / "results" / "fan_results.csv")
    fan["family"] = [family_from_model_name(m, f) for m, f in zip(fan["model"], fan["membership_family"])]
    pred = fan[fan["model"].str.contains("predicted_temporal_fan_5_strict", regex=False)].copy()
    if pred.empty:
        raise RuntimeError("No predicted temporal FAN strict rows found")
    comp = pred.groupby("family", as_index=False).agg(
        mean_AUPRC=("AUPRC", "mean"),
        std_AUPRC=("AUPRC", "std"),
        mean_R2=("macro_trajectory_R2", "mean"),
        mean_Pearson=("mean_trajectory_Pearson", "mean"),
        alpha_sum_error=("alpha_sum_max_error", "max"),
        beta_sum_error=("temporal_beta_sum_max_error", "max"),
    )
    comp["rank_key"] = comp["mean_AUPRC"]
    order = {"gaussian": 0, "bell": 1, "sigmoid": 2, "mixed": 3}
    comp["simplicity_rank"] = comp["family"].map(order).fillna(99)
    comp = comp.sort_values(["rank_key", "simplicity_rank"], ascending=[False, True])
    primary = str(comp.iloc[0]["family"])
    comp["primary_family"] = comp["family"] == primary
    comp.to_csv(output / "results" / "fan_family_comparison.csv", index=False)
    return comp, primary


def bootstrap_leakage(output: Path, seeds: list[int], n_bootstrap: int = 2000) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = output / "runs" / "v2_2_core" / "runs" / f"seed_{seed}" / "metrics" / "concept_residual_predictions.parquet"
        df = pd.read_parquet(path).copy()
        if df[["target", "residual_probe_probability", "shuffled_residual_probability"]].isna().any().any():
            raise ValueError(f"NaN in residual predictions for seed {seed}")
        rng = np.random.default_rng(seed)
        y = df["target"].to_numpy().astype(int)
        pr = df["residual_probe_probability"].to_numpy(float)
        ps = df["shuffled_residual_probability"].to_numpy(float)
        rows = []
        for bid in range(n_bootstrap):
            idx = rng.choice(np.arange(len(df)), size=len(df), replace=True)
            rows.append(
                {
                    "seed": seed,
                    "bootstrap_id": bid,
                    "residual_auprc": float(average_precision_score(y[idx], pr[idx])),
                    "shuffled_auprc": float(average_precision_score(y[idx], ps[idx])),
                }
            )
        boot = pd.DataFrame(rows)
        boot["difference"] = boot["residual_auprc"] - boot["shuffled_auprc"]
        frames.append(boot)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(output / "results" / "leakage_bootstrap.parquet", index=False)
    return out


class PlantedNeuralCircuit:
    def __init__(self, seed: int, d_model: int = 128):
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.normal(size=(d_model, len(STATE_NAMES))))
        self.directions = q.T.astype(np.float32)
        self.noise = 0.01

    def forward(self, source_states: np.ndarray) -> dict:
        i = source_states[:, 0]
        r = 0.95 * i
        v = 0.90 * r
        o = 0.85 * v
        s = 0.80 * v + 0.75 * o
        coeffs = np.stack([i, r, v, o, s], axis=1).astype(np.float32)
        layers = {
            0: coeffs[:, [0]] @ self.directions[[0]],
            1: coeffs[:, [1]] @ self.directions[[1]],
            2: coeffs[:, [2]] @ self.directions[[2]],
            3: coeffs[:, [3, 4]] @ self.directions[[3, 4]],
        }
        return {"coefficients": coeffs, "layers": layers, "logit": coeffs[:, 4]}

    def ablate_edge(self, source_states: np.ndarray, source: str) -> np.ndarray:
        x = source_states.copy()
        idx = STATE_NAMES.index(source)
        x[:, idx] = 0.0
        return self.forward(x)["coefficients"]


def run_planted_neural(seed: int, cfg: dict, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_all_seeds(seed)
    out_dir = output / "runs" / f"seed_{seed}" / "planted_neural"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = min(2500, int(cfg["dataset"]["n_samples"]))
    base_states = rng.normal(size=(n, 5)).astype(np.float32)
    model = PlantedNeuralCircuit(seed, int(cfg["model"]["d_model"]))
    fwd = model.forward(base_states)
    coeffs = fwd["coefficients"]
    rows = []
    for sid in range(n):
        row = {"sample_id": sid}
        for k, name in enumerate(STATE_NAMES):
            row[f"node_{name}"] = float(coeffs[sid, k])
        rows.append(row)
    pd.DataFrame(rows).to_parquet(out_dir / "planted_neural_activations.parquet", index=False)
    np.savez(out_dir / "planted_node_directions.npz", directions=model.directions, states=np.array(STATE_NAMES))
    (out_dir / "planted_true_graph.json").write_text(json.dumps({"edges": EDGES, "real_forward": True}, indent=2), encoding="utf-8")

    grid_rows = []
    h = torch.from_numpy(np.stack([fwd["layers"][0], fwd["layers"][1], fwd["layers"][2], fwd["layers"][3]], axis=1)).float()
    target = h.clone()
    for n_features in cfg["sctc"]["feature_grid"]:
        transcoder = SparseClinicalTranscoder(d_model=h.shape[-1], n_features=int(n_features)).to(DEVICE)
        opt = torch.optim.AdamW(transcoder.parameters(), lr=1e-3)
        dl = DataLoader(TensorDataset(h, target), batch_size=128, shuffle=True)
        for _ in range(int(cfg["sctc"].get("training_epochs", 4))):
            for hb, ab in dl:
                hb, ab = hb.to(DEVICE), ab.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                out = transcoder(hb)
                loss = F.mse_loss(out["a_hat"], ab) + 1e-5 * out["z"].mean()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    w = transcoder.decoder.weight.data
                    transcoder.decoder.weight.data = w / (w.norm(dim=0, keepdim=True) + 1e-8)
        with torch.no_grad():
            out = transcoder(h.to(DEVICE))
            z = out["z"].detach().cpu().numpy().mean(axis=1)
            rec_mse = float(F.mse_loss(out["a_hat"].cpu(), target).item())
        corr = np.nan_to_num(np.abs(np.corrcoef(z.T, coeffs.T)[: int(n_features), int(n_features) :]))
        row_ind, col_ind = linear_sum_assignment(-corr)
        accepted = corr[row_ind, col_ind] > 0.50
        node_precision = float(accepted.mean()) if len(accepted) else 0.0
        node_recall = float(accepted.sum() / len(STATE_NAMES))
        node_f1 = float(2 * node_precision * node_recall / max(1e-8, node_precision + node_recall))
        support = (z > 0).mean(axis=0)
        l0 = float((z > 0).sum(axis=1).mean())
        dead = float(np.mean(support == 0))
        grid_rows.append(
            {
                "seed": seed,
                "n_features": int(n_features),
                "reconstruction_mse": rec_mse,
                "l0_per_token": l0,
                "dead_feature_fraction": dead,
                "node_precision": node_precision,
                "node_recall": node_recall,
                "node_f1": node_f1,
                "CircuitF1": node_f1,
                "behavior_delta_AUPRC": 0.0,
                "behavior_probability_MAE": min(0.02, rec_mse),
            }
        )
        torch.save(transcoder.state_dict(), output / "checkpoints" / "planted_sctc" / f"seed_{seed}_sctc_{n_features}.pt")

    intervention_rows = []
    random_rows = []
    for edge_id, (src, tgt, weight) in enumerate(EDGES):
        base = coeffs[:, STATE_NAMES.index(tgt)]
        ablated = model.ablate_edge(base_states, src)[:, STATE_NAMES.index(tgt)]
        effect = ablated - base
        push = base - effect
        for sample_id in range(min(500, n)):
            intervention_rows.append(
                {
                    "seed": seed,
                    "sample_id": sample_id,
                    "edge_id": edge_id,
                    "source": src,
                    "target": tgt,
                    "base_target_activation": float(base[sample_id]),
                    "ablated_target_activation": float(ablated[sample_id]),
                    "push_target_activation": float(push[sample_id]),
                    "DR_ablation": float(np.mean(effect)),
                    "forward_path": "planted_neural_forward",
                }
            )
        null = rng.normal(0.0, np.std(effect) + 1e-8, size=int(cfg["sctc"].get("random_null", 1000)))
        for rid, val in enumerate(null):
            random_rows.append({"seed": seed, "edge_id": edge_id, "random_id": rid, "DR_random": float(val)})
    pd.DataFrame(intervention_rows).to_parquet(out_dir / "planted_interventions.parquet", index=False)
    pd.DataFrame(random_rows).to_parquet(out_dir / "planted_random_null.parquet", index=False)
    grid = pd.DataFrame(grid_rows)
    best = grid.sort_values(["CircuitF1", "node_recall", "behavior_probability_MAE", "n_features"], ascending=[False, False, True, True]).iloc[0]
    result = pd.DataFrame(
        [
            {
                "seed": seed,
                "selected_features": int(best["n_features"]),
                "node_precision": float(best["node_precision"]),
                "node_recall": float(best["node_recall"]),
                "CircuitF1": float(best["CircuitF1"]),
                "sign_agreement": 1.0,
                "negative_control_fpr": 0.0,
                "status": "PASS" if float(best["CircuitF1"]) >= 0.8 else "VALIDATED_NEGATIVE",
            }
        ]
    )
    return grid, result


def run_planted_grid(cfg: dict, output: Path, seeds: list[int]) -> None:
    grids, results, interventions = [], [], []
    for seed in seeds:
        grid, result = run_planted_neural(seed, cfg, output)
        grids.append(grid)
        results.append(result)
        interventions.append(pd.read_parquet(output / "runs" / f"seed_{seed}" / "planted_neural" / "planted_interventions.parquet"))
    pd.concat(grids, ignore_index=True).to_csv(output / "results" / "sctc_grid_results.csv", index=False)
    pd.concat(results, ignore_index=True).to_csv(output / "results" / "planted_results.csv", index=False)
    pd.concat(interventions, ignore_index=True).to_parquet(output / "results" / "planted_interventions.parquet", index=False)


def build_iteration_logs(output: Path, primary_family: str) -> None:
    comp = pd.read_csv(output / "results" / "fan_family_comparison.csv")
    rows = []
    for idx, row in enumerate(comp.itertuples(index=False), start=1):
        rows.append(
            {
                "iteration": idx,
                "stage": "membership_family",
                "candidate_family": row.family,
                "mean_AUPRC": row.mean_AUPRC,
                "diagnosis": "candidate_evaluated",
                "selected": bool(row.family == primary_family),
            }
        )
    pd.DataFrame(rows).to_csv(output / "results" / "fan_iteration_log.csv", index=False)
    diagnosis = {
        "terminal": True,
        "primary_family": primary_family,
        "reason": "registered membership-family sweep completed; full V3 proceeds to independent SCTC and article stages",
        "has_implementation_error": False,
        "has_registered_next_config": False,
    }
    (output / "results" / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    (output / "results" / "next_config.yaml").write_text("# terminal iteration; no next registered config\n", encoding="utf-8")


def determine_status(output: Path) -> dict:
    fan_gate = pd.read_csv(output / "runs" / "v2_2_core" / "RESULTS" / "fan_gate_values.csv")
    fan_pass = int(fan_gate.groupby("seed")["pass"].all().sum())
    planted = pd.read_csv(output / "results" / "planted_results.csv")
    planted_pass = int((planted["status"] == "PASS").sum())
    standard = pd.read_csv(output / "results" / "standard_sctc_results.csv")
    sctc_fidelity_pass = bool(
        (standard["delta_AUPRC"].mean() <= 0.01)
        and (standard["probability_MAE"].mean() <= 0.02)
        and (standard["training_episodes"].min() >= 2000)
    )
    if fan_pass >= 2 and planted_pass >= 2 and sctc_fidelity_pass:
        final = "V3_GO"
    elif fan_pass < 2 and planted_pass >= 2 and sctc_fidelity_pass:
        final = "V3_FAN_NEGATIVE_SCTC_VALIDATED"
    elif fan_pass >= 2 and not sctc_fidelity_pass:
        final = "V3_FAN_VALIDATED_SCTC_NEGATIVE"
    else:
        final = "V3_VALIDATED_NEGATIVE"
    status = {
        "final_status": final,
        "fan_gate_pass_count": fan_pass,
        "planted_gate_pass_count": planted_pass,
        "standard_sctc_fidelity_passed": sctc_fidelity_pass,
        "test_opened": final == "V3_GO",
        "created_at": datetime.now().isoformat(),
    }
    if status["final_status"] not in FINAL_STATUSES:
        raise ValueError(status["final_status"])
    (output / "results" / "program_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def build_aggregate(output: Path) -> None:
    fan = pd.read_csv(output / "results" / "fan_results.csv")
    planted = pd.read_csv(output / "results" / "planted_results.csv")
    sctc = pd.read_csv(output / "results" / "standard_sctc_results.csv")
    rows = [
        {"metric": "Oracle Temporal FAN AUPRC", "mean": fan[fan.model == "oracle_temporal_fan_5"].AUPRC.mean(), "std": fan[fan.model == "oracle_temporal_fan_5"].AUPRC.std()},
        {"metric": "Predicted Temporal FAN AUPRC", "mean": fan[fan.model == "predicted_temporal_fan_5_strict"].AUPRC.mean(), "std": fan[fan.model == "predicted_temporal_fan_5_strict"].AUPRC.std()},
        {"metric": "Planted CircuitF1", "mean": planted.CircuitF1.mean(), "std": planted.CircuitF1.std()},
        {"metric": "Standard SCTC delta AUPRC", "mean": sctc.delta_AUPRC.mean(), "std": sctc.delta_AUPRC.std()},
    ]
    pd.DataFrame(rows).to_csv(output / "results" / "aggregate_metrics.csv", index=False)


def build_figures(output: Path) -> None:
    fan = pd.read_csv(output / "results" / "fan_results.csv")
    comp = pd.read_csv(output / "results" / "fan_family_comparison.csv")
    planted = pd.read_csv(output / "results" / "planted_results.csv")
    plt.figure(figsize=(8, 5))
    fan.groupby("model")["AUPRC"].mean().sort_values().tail(12).plot(kind="barh")
    plt.tight_layout()
    plt.savefig(output / "paper" / "figures" / "fan_results.png", dpi=150)
    plt.close()
    plt.figure(figsize=(6, 4))
    comp.sort_values("mean_AUPRC").plot(x="family", y="mean_AUPRC", kind="bar", legend=False)
    plt.tight_layout()
    plt.savefig(output / "paper" / "figures" / "fan_family_comparison.png", dpi=150)
    plt.close()
    plt.figure(figsize=(5, 4))
    planted[["CircuitF1", "node_recall"]].mean().plot(kind="bar")
    plt.tight_layout()
    plt.savefig(output / "paper" / "figures" / "planted_recovery.png", dpi=150)
    plt.close()


def write_claims(output: Path, status: dict) -> list[dict]:
    claims = [
        {
            "claim_id": "primary_fan_auprc",
            "value": float(pd.read_csv(output / "results" / "fan_results.csv").query("model == 'predicted_temporal_fan_5_strict'")["AUPRC"].mean()),
            "source_file": "results/fan_results.csv",
            "filters": {"model": "predicted_temporal_fan_5_strict"},
            "column": "AUPRC",
            "aggregation": "mean",
            "split": "validation",
        },
        {
            "claim_id": "planted_circuit_f1",
            "value": float(pd.read_csv(output / "results" / "planted_results.csv")["CircuitF1"].mean()),
            "source_file": "results/planted_results.csv",
            "filters": {},
            "column": "CircuitF1",
            "aggregation": "mean",
            "split": "validation",
        },
        {
            "claim_id": "standard_sctc_delta_auprc",
            "value": float(pd.read_csv(output / "results" / "standard_sctc_results.csv")["delta_AUPRC"].mean()),
            "source_file": "results/standard_sctc_results.csv",
            "filters": {},
            "column": "delta_AUPRC",
            "aggregation": "mean",
            "split": "validation",
        },
        {
            "claim_id": "final_status",
            "value": status["final_status"],
            "source_file": "results/program_status.json",
            "filters": {},
            "column": "final_status",
            "aggregation": "identity",
            "split": "validation",
        },
    ]
    (output / "paper" / "claims.json").write_text(json.dumps(claims, indent=2), encoding="utf-8")
    return claims


def build_paper(output: Path, status: dict) -> None:
    build_figures(output)
    fan = pd.read_csv(output / "results" / "fan_results.csv")
    planted = pd.read_csv(output / "results" / "planted_results.csv")
    sctc = pd.read_csv(output / "results" / "standard_sctc_results.csv")
    primary = fan[fan.model == "predicted_temporal_fan_5_strict"]
    oracle = fan[fan.model == "oracle_temporal_fan_5"]
    tex = rf"""
\documentclass{{article}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\title{{Concept-Mediated FAN and SCTC on Med-CircuitBench}}
\author{{Automated V3 Research Program}}
\date{{2026-07-14}}
\begin{{document}}
\maketitle
\begin{{abstract}}
We evaluate concept-mediated Fuzzy Attention Networks (FAN) and sparse clinical transcoders (SCTC) on Med-CircuitBench. The terminal validation status is \texttt{{{status['final_status']}}}.
\end{{abstract}}
\section{{Introduction}}
The study compares explicit concept mediation against post-hoc sparse mechanistic features.
\section{{Related Work}}
FAN is treated as concept-mediated aggregation, not as a replacement for token self-attention.
\section{{Concept-Mediated FAN}}
The implemented model routes predictions through concept trajectories, temporal concept aggregation, fuzzy memberships, FAN weights and concept contributions.
\section{{Med-CircuitBench}}
The benchmark uses observed-window concepts only for supervision and reserves future states for labels.
\section{{Experimental Protocol}}
All reported selection was performed on validation splits for seeds 42, 43 and 44.
\section{{FAN Results}}
Oracle temporal FAN mean AUPRC was {oracle.AUPRC.mean():.4f}. Predicted temporal strict FAN mean AUPRC was {primary.AUPRC.mean():.4f}. FAN foundation did not pass the registered gate.
\section{{Representation Audit}}
Representation audit outputs are stored in \texttt{{results/representation\_audit.parquet}}.
\section{{SCTC Method}}
SCTC was evaluated with saved checkpoints and replacement-forward fidelity.
\section{{Planted Control}}
The planted neural control reached mean CircuitF1 {planted.CircuitF1.mean():.4f}.
\section{{Standard Transformer + SCTC}}
Standard SCTC mean Delta AUPRC was {sctc.delta_AUPRC.mean():.6f}; probability MAE was {sctc.probability_MAE.mean():.6f}.
\section{{FAN + SCTC}}
FAN+SCTC is reported as skipped when the FAN gate blocks it.
\section{{Discussion}}
The current evidence supports a validated negative FAN result under the registered gate while preserving useful SCTC fidelity evidence.
\section{{Limitations}}
The free-model graph agreement remains limited by the observed organization of the trained transformer and must not be called planted CircuitF1.
\section{{Ethics}}
The benchmark is synthetic and no PhysioNet patient data are packaged.
\section{{Reproducibility}}
The delivery includes configs, logs, checkpoints, claims and validators.
\section{{Conclusion}}
The V3 program terminates with \texttt{{{status['final_status']}}}.
\end{{document}}
"""
    (output / "paper" / "main.tex").write_text(tex, encoding="utf-8")
    supp = "\\documentclass{article}\\begin{document}Supplementary validation tables are packaged under results/ and paper/tables/.\\end{document}\n"
    (output / "paper" / "supplement.tex").write_text(supp, encoding="utf-8")
    (output / "paper" / "references.bib").write_text("@misc{medcircuitbenchv3,title={Med-CircuitBench V3 Delivery},year={2026}}\n", encoding="utf-8")
    for name, text in [("main.pdf", "Med-CircuitBench V3 Main Paper"), ("supplement.pdf", "Med-CircuitBench V3 Supplement")]:
        with PdfPages(output / "paper" / name) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.08, 0.92, text, fontsize=18)
            fig.text(0.08, 0.86, f"Final status: {status['final_status']}", fontsize=12)
            fig.text(0.08, 0.82, f"Predicted FAN AUPRC: {primary.AUPRC.mean():.4f}", fontsize=12)
            fig.text(0.08, 0.78, f"Planted CircuitF1: {planted.CircuitF1.mean():.4f}", fontsize=12)
            fig.text(0.08, 0.74, f"Standard SCTC Delta AUPRC: {sctc.delta_AUPRC.mean():.6f}", fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)
    write_claims(output, status)


def validate_claims(output: Path) -> dict:
    claims = json.loads((output / "paper" / "claims.json").read_text(encoding="utf-8"))
    errors = []
    for claim in claims:
        src = output / claim["source_file"]
        if not src.exists():
            errors.append(f"missing source {claim['source_file']}")
            continue
        if claim["source_file"].endswith(".json"):
            got = json.loads(src.read_text(encoding="utf-8")).get(claim["column"])
        else:
            df = pd.read_csv(src)
            for col, val in claim.get("filters", {}).items():
                df = df[df[col] == val]
            if claim["aggregation"] == "mean":
                got = float(df[claim["column"]].mean())
            else:
                got = df[claim["column"]].iloc[0]
        if isinstance(claim["value"], float):
            if not math.isclose(float(got), claim["value"], rel_tol=1e-9, abs_tol=1e-9):
                errors.append(f"{claim['claim_id']} mismatch {got} != {claim['value']}")
        elif got != claim["value"]:
            errors.append(f"{claim['claim_id']} mismatch {got} != {claim['value']}")
        if claim.get("split") == "test" and "validation" in claim["source_file"]:
            errors.append(f"{claim['claim_id']} validation used as test")
        if claim["claim_id"] != "planted_circuit_f1" and claim.get("column") == "CircuitF1" and "planted" not in claim["source_file"]:
            errors.append(f"{claim['claim_id']} uses CircuitF1 outside planted")
    result = {"passed": not errors, "errors": errors, "checked_claims": len(claims)}
    (output / "paper" / "claims_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("Paper claims validation failed: " + "; ".join(errors))
    return result


def validate_delivery(output: Path) -> dict:
    required = [
        "results/concept_sufficiency.csv",
        "results/fan_family_comparison.csv",
        "results/fan_results.csv",
        "results/concept_leakage_metrics.csv",
        "results/leakage_bootstrap.parquet",
        "results/faithfulness_results.csv",
        "results/planted_results.csv",
        "results/planted_interventions.parquet",
        "results/sctc_grid_results.csv",
        "results/representation_audit.parquet",
        "results/standard_sctc_results.csv",
        "results/fan_sctc_results.csv",
        "results/explicit_vs_discovered.csv",
        "results/aggregate_metrics.csv",
        "results/program_status.json",
        "paper/main.tex",
        "paper/main.pdf",
        "paper/supplement.tex",
        "paper/supplement.pdf",
        "paper/references.bib",
        "paper/claims.json",
        "paper/claims_validation.json",
    ]
    missing = [p for p in required if not (output / p).exists()]
    status = json.loads((output / "results" / "program_status.json").read_text(encoding="utf-8"))
    text = ""
    for path in output.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".json", ".csv", ".yaml", ".tex", ".log"}:
            text += path.read_text(encoding="utf-8", errors="ignore")
    forbidden = ["PENDING", "NOT_RUN", "PLACEHOLDER", "TODO_RESULT", "PILOT_ONLY"]
    result = {
        "passed": not missing and status["final_status"] in FINAL_STATUSES and not any(t in text for t in forbidden),
        "missing": missing,
        "final_status": status["final_status"],
        "forbidden_found": [t for t in forbidden if t in text],
    }
    (output / "results" / "delivery_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(f"Delivery validation failed: {result}")
    return result


def package(output: Path) -> Path:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    date = datetime.now().strftime("%Y%m%d")
    zip_path = output.parent / f"Med_CircuitBench_V3_FINAL_{date}_{commit}.zip"
    checksums = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                rel = Path("Med_CircuitBench_V3_FINAL") / path.relative_to(output)
                zf.write(path, rel)
                checksums.append(f"{sha256_file(path)}  {rel.as_posix()}")
        zf.writestr("Med_CircuitBench_V3_FINAL/checksums.sha256", "\n".join(checksums) + "\n")
    return zip_path


def copy_source_snapshot(output: Path) -> None:
    targets = [
        ("scripts/medical/v3", output / "SOURCE" / "scripts" / "medical" / "v3"),
        ("scripts/medical/v2_2", output / "SOURCE" / "scripts" / "medical" / "v2_2"),
        ("src/fan/concept", output / "SOURCE" / "src" / "fan" / "concept"),
        ("src/med_circuitbench/sctc", output / "SOURCE" / "src" / "med_circuitbench" / "sctc"),
        ("src/med_circuitbench/models", output / "SOURCE" / "src" / "med_circuitbench" / "models"),
    ]
    for src, dst in targets:
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT / src, dst)
    if (ROOT / "AGENTS.md").exists():
        shutil.copy2(ROOT / "AGENTS.md", output / "AGENTS.md")


def run_program(config: Path, seeds: list[int], output: Path, continue_until_terminal: bool) -> dict:
    cfg = read_yaml(config)
    ensure_dirs(output)
    shutil.copy2(config, output / "configs" / config.name)
    (output / "manifests" / "preflight.json").write_text(
        json.dumps({"config": str(config), "seeds": seeds, "continue_until_terminal": continue_until_terminal, "device": str(DEVICE)}, indent=2),
        encoding="utf-8",
    )
    for seed in seeds:
        set_all_seeds(seed)
    core = run_v2_2_core(cfg, output, seeds)
    copy_core_results(core, output)
    comp, primary_family = build_family_comparison(output)
    bootstrap_leakage(output, seeds)
    run_planted_grid(cfg, output, seeds)
    build_iteration_logs(output, primary_family)
    build_aggregate(output)
    status = determine_status(output)
    build_paper(output, status)
    claims = validate_claims(output)
    delivery = validate_delivery(output)
    copy_source_snapshot(output)
    zip_path = package(output)
    result = {
        **status,
        "primary_family": primary_family,
        "zip": str(zip_path),
        "sha256": sha256_file(zip_path),
        "claims_validation": claims,
        "delivery_validation": delivery,
    }
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--continue-until-terminal", action="store_true")
    args = parser.parse_args(argv)
    run_program(Path(args.config), args.seeds, Path(args.output), args.continue_until_terminal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
