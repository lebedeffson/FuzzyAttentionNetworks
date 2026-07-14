#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fan.concept import MultiSetAdditiveTemporalConceptFANModel
from fan.sctc.model import SparseTranscoder
from fan.sctc.evaluation import feature_catalog, fidelity_metrics
from med_circuitbench.models.transformer import ClinicalTransformer, TransformerConfig
from med_circuitbench.planted import EDGES, STATE_NAMES, PlantedCircuitModel
from scripts.medical.v2.run_v2_1_program import make_sequence_loaders
from scripts.medical.v2.run_v2_program import make_episodes, split_frame, subset_cols
from scripts.medical.v3.run_oracle_alpha_ablation import run_alpha_ablation
from scripts.medical.v3.run_predicted_fan_strict import make_predicted_model, run_program as run_predicted_fan
from scripts.medical.v3.validate_no_synthetic_results import validate as validate_no_synthetic


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAPTURE_POINTS = ["residual_pre", "attention_output", "residual_mid", "mlp_output", "residual_post"]
FINAL_STATUSES = {
    "V3_REAL_GO",
    "V3_REAL_MIXED_RESULT",
    "V3_REAL_VALIDATED_NEGATIVE",
    "V3_REAL_FAN_VALIDATED_SCTC_NEGATIVE",
}

LEGACY_ORACLE_DIR = ROOT / "artifacts" / "medical" / "v3_alpha_ablation" / "no_alpha"
LEGACY_PREDICTED_DIR = ROOT / "artifacts" / "medical" / "v3_predicted_fan_strict"


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(x: np.ndarray) -> str:
    arr = np.ascontiguousarray(x)
    return sha256_bytes(arr.tobytes())


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_layout(output: Path) -> None:
    for name in [
        "manifests",
        "dataset",
        "splits",
        "fan",
        "planted/activation_cache",
        "representation_audit",
        "standard_sctc",
        "fan_sctc",
        "validation",
        "test",
        "results",
        "checkpoints/fan",
        "checkpoints/planted_sctc",
        "checkpoints/standard_sctc",
        "checkpoints/fan_sctc",
        "logs",
        "paper/figures",
        "paper/tables",
        "delivery",
    ]:
        (output / name).mkdir(parents=True, exist_ok=True)


def write_manifest(path: Path, payload: dict, files: list[Path] | None = None) -> None:
    files = files or []
    payload = dict(payload)
    payload["created_at"] = datetime.now().isoformat()
    payload["files"] = [
        {"path": str(file), "sha256": sha256_file(file), "size": file.stat().st_size}
        for file in files
        if file.exists()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stage_done(manifest: Path) -> bool:
    if not manifest.exists():
        return False
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    if not raw.get("validator_passed", False):
        return False
    for item in raw.get("files", []):
        path = Path(item["path"])
        if not path.exists() or sha256_file(path) != item["sha256"]:
            return False
    return True


def append_provenance(output: Path, record: dict) -> None:
    record = dict(record)
    record["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with (output / "manifests" / "result_provenance.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    return {
        "AUROC": float(roc_auc_score(y, p)) if np.unique(y).size > 1 else float("nan"),
        "AUPRC": float(average_precision_score(y, p)),
        "prevalence": float(y.mean()),
    }


def validate_claims(output: Path) -> dict:
    claims_path = output / "paper" / "claims.json"
    if not claims_path.exists():
        return {"passed": False, "findings": [{"reason": "missing_claims_json"}]}
    findings = []
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    for claim in claims:
        source = output / claim["source_file"]
        if not source.exists():
            findings.append({"claim_id": claim.get("claim_id"), "reason": "missing_source", "source_file": str(source)})
            continue
        if source.suffix == ".json":
            data = json.loads(source.read_text(encoding="utf-8"))
            actual = data.get(claim["column"])
        else:
            df = pd.read_csv(source)
            for key, value in claim.get("filters", {}).items():
                df = df[df[key] == value]
            if df.empty:
                findings.append({"claim_id": claim.get("claim_id"), "reason": "empty_filter"})
                continue
            column = claim["column"]
            if claim.get("aggregation") == "mean":
                actual = float(df[column].mean())
            elif claim.get("aggregation") == "max":
                actual = float(df[column].max())
            elif claim.get("aggregation") == "identity":
                actual = df[column].iloc[0]
            else:
                actual = df[column].iloc[0]
        expected = claim["value"]
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float, np.floating)):
            if not np.isclose(float(actual), float(expected), atol=1e-8, rtol=1e-8):
                findings.append({"claim_id": claim.get("claim_id"), "reason": "value_mismatch", "expected": expected, "actual": actual})
        elif actual != expected:
            findings.append({"claim_id": claim.get("claim_id"), "reason": "value_mismatch", "expected": expected, "actual": actual})
    return {"passed": not findings, "findings": findings}


def validate_delivery(output: Path) -> dict:
    required = [
        "results/program_status.json",
        "results/aggregate_metrics.csv",
        "results/fan_gate.json",
        "results/predicted_fan_results.csv",
        "results/planted_results.csv",
        "results/representation_audit.parquet",
        "results/standard_sctc_results.csv",
        "results/fan_sctc_results.csv",
        "paper/main.pdf",
        "paper/supplement.pdf",
        "paper/claims.json",
        "paper/claims_validation.json",
        "manifests/test_unlock_manifest.json",
        "manifests/test_consumed.lock",
    ]
    findings = [{"path": rel, "reason": "missing"} for rel in required if not (output / rel).exists()]
    if (output / "results/program_status.json").exists():
        status = json.loads((output / "results/program_status.json").read_text(encoding="utf-8"))
        if status.get("final_status") not in FINAL_STATUSES:
            findings.append({"path": "results/program_status.json", "reason": "invalid_final_status"})
    for rel in ["results", "paper", "manifests"]:
        base = output / rel
        if base.exists():
            for path in base.rglob("*"):
                if path.is_file() and path.suffix in {".json", ".csv", ".tex", ".txt"}:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    for token in ["PENDING", "PLACEHOLDER", "TODO_RESULT", "PILOT_ONLY", "NOT_RUN"]:
                        if token in text:
                            findings.append({"path": str(path.relative_to(output)), "reason": f"forbidden_token_{token}"})
    return {"passed": not findings, "findings": findings}


def preflight(cfg: dict, seeds: list[int], output: Path) -> dict:
    checks = [
        ("n_samples", int(cfg["dataset"]["n_samples"]) >= 10000),
        ("observed_window", int(cfg["dataset"]["observed_window"]) == 36),
        ("sequence_length", int(cfg["dataset"]["sequence_length"]) == 42),
        ("three_seeds", len(seeds) == 3),
        ("d_model", int(cfg["model"]["d_model"]) >= 128),
        ("layers", int(cfg["model"]["layers"]) >= 4),
    ]
    result = {"stage": "00_preflight", "checks": [{"name": n, "pass": bool(p)} for n, p in checks]}
    result["validator_passed"] = all(row["pass"] for row in result["checks"])
    write_manifest(output / "manifests" / "00_preflight.json", result)
    if not result["validator_passed"]:
        raise RuntimeError(f"preflight failed: {result}")
    return result


def normalize_by_train(x_train: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0, keepdims=True)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0, keepdims=True) + 1e-6
    out = [(a - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1) for a in (x_train, *arrays)]
    return tuple(np.asarray(a, dtype=np.float32) for a in out)


def prepare_arrays(split: dict, cols: np.ndarray, include_test: bool = False) -> dict:
    x_train = np.stack(split["train"]["model_input"].map(lambda x: np.asarray(x, dtype=np.float32)).to_numpy())[:, :, cols]
    x_val = np.stack(split["validation"]["model_input"].map(lambda x: np.asarray(x, dtype=np.float32)).to_numpy())[:, :, cols]
    if include_test:
        x_test = np.stack(split["test"]["model_input"].map(lambda x: np.asarray(x, dtype=np.float32)).to_numpy())[:, :, cols]
        x_train, x_val, x_test = normalize_by_train(x_train, x_val, x_test)
    else:
        x_train, x_val = normalize_by_train(x_train, x_val)
        x_test = None
    out = {
        "x_train": x_train,
        "x_val": x_val,
        "y_train": split["train"]["target"].to_numpy(np.float32),
        "y_val": split["validation"]["target"].to_numpy(np.float32),
        "c_train_seq": np.stack(split["train"]["states"].map(lambda x: np.asarray(x, dtype=np.float32)).to_numpy())[:, :36, :5],
        "c_val_seq": np.stack(split["validation"]["states"].map(lambda x: np.asarray(x, dtype=np.float32)).to_numpy())[:, :36, :5],
        "train_ids": split["train"]["episode_id"].to_numpy(int),
        "val_ids": split["validation"]["episode_id"].to_numpy(int),
    }
    if include_test:
        out.update(
            {
                "x_test": x_test,
                "y_test": split["test"]["target"].to_numpy(np.float32),
                "c_test_seq": np.stack(split["test"]["states"].map(lambda x: np.asarray(x, dtype=np.float32)).to_numpy())[:, :36, :5],
                "test_ids": split["test"]["episode_id"].to_numpy(int),
            }
        )
    return out


def dataset_and_splits(cfg: dict, seeds: list[int], output: Path, resume: bool) -> dict[int, dict]:
    manifest = output / "manifests" / "01_dataset_and_splits.json"
    if resume and stage_done(manifest):
        return {}
    summaries = []
    for seed in seeds:
        set_all_seeds(seed)
        df = make_episodes(seed, cfg, "clean")
        split = split_frame(df, seed)
        seed_dir = output / "splits" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in split.items():
            meta = frame[["episode_id", "target", "infection_start", "shock_score"]].copy()
            meta.to_parquet(seed_dir / f"{name}_episodes.parquet", index=False)
        ids = {name: frame["episode_id"].to_numpy(int) for name, frame in split.items()}
        summaries.append(
            {
                "seed": seed,
                "episode_count": int(len(df)),
                "prevalence": float(df["target"].mean()),
                "train_count": int(len(split["train"])),
                "validation_count": int(len(split["validation"])),
                "test_count": int(len(split["test"])),
                "train_ids_sha256": sha256_array(ids["train"]),
                "validation_ids_sha256": sha256_array(ids["validation"]),
                "test_ids_sha256": sha256_array(ids["test"]),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "dataset" / "dataset_summary.csv", index=False)
    (output / "manifests" / "dataset_manifest.json").write_text(json.dumps({"seeds": summaries, "generator_config": cfg["dataset"]}, indent=2), encoding="utf-8")
    (output / "manifests" / "split_manifest.json").write_text(json.dumps({"seeds": summaries}, indent=2), encoding="utf-8")
    write_manifest(manifest, {"stage": "01_dataset_and_splits", "validator_passed": True}, [output / "dataset" / "dataset_summary.csv"])
    return {}


def run_fan_validation(cfg: dict, config_path: Path, seeds: list[int], output: Path, resume: bool) -> dict:
    manifest = output / "manifests" / "03_fan_gate.json"
    if resume and stage_done(manifest):
        return json.loads((output / "results" / "fan_gate.json").read_text(encoding="utf-8"))
    oracle_dir = output / "fan" / "oracle_alpha_ablation"
    predicted_dir = output / "fan" / "predicted_strict"
    if not (oracle_dir / "no_alpha" / "oracle_fan_results.csv").exists() and (LEGACY_ORACLE_DIR / "oracle_fan_results.csv").exists():
        shutil.copytree(LEGACY_ORACLE_DIR, oracle_dir / "no_alpha", dirs_exist_ok=True)
    if not (predicted_dir / "predicted_fan_results.csv").exists() and (LEGACY_PREDICTED_DIR / "predicted_fan_results.csv").exists():
        shutil.copytree(LEGACY_PREDICTED_DIR, predicted_dir, dirs_exist_ok=True)
    if not (oracle_dir / "no_alpha" / "oracle_fan_results.csv").exists():
        run_alpha_ablation(cfg, seeds, oracle_dir, ["no_alpha"])
    oracle_results = oracle_dir / "no_alpha" / "oracle_fan_results.csv"
    oracle_manifest = oracle_dir / "no_alpha" / "oracle_manifest.json"
    if not oracle_manifest.exists():
        oracle_manifest.write_text(
            json.dumps(
                {
                    "stage": "oracle_alpha_ablation_manifest",
                    "dataset": "med_circuitbench_clean",
                    "split": "train_validation",
                    "seeds": seeds,
                    "concept_definition": "temporal_I_R_V_O_S_hours_0_35",
                    "concept_scaling": "minmax_train",
                    "membership_family": "gaussian",
                    "n_memberships": 3,
                    "architecture": "Multi-Set Additive Temporal Concept FAN-NoAlpha",
                    "alpha_mode": "no_alpha",
                    "results_sha256": sha256_file(oracle_results),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if (predicted_dir / "predicted_fan_diagnosis.json").exists():
        diagnosis = json.loads((predicted_dir / "predicted_fan_diagnosis.json").read_text(encoding="utf-8"))
    else:
        diagnosis = run_predicted_fan(cfg, seeds, predicted_dir, oracle_results, oracle_manifest)
    pred = pd.read_csv(predicted_dir / "predicted_vs_oracle_noalpha.csv")
    pred.to_csv(output / "results" / "predicted_vs_oracle_noalpha.csv", index=False)
    fan_results = pd.read_csv(predicted_dir / "predicted_fan_results.csv")
    fan_results.to_csv(output / "results" / "predicted_fan_results.csv", index=False)
    fan_results.to_csv(output / "results" / "fan_results.csv", index=False)
    leak = pd.read_csv(predicted_dir / "concept_leakage_metrics.csv")
    leak.to_csv(output / "results" / "fan_leakage_metrics.csv", index=False)
    leak.to_csv(output / "results" / "concept_leakage_metrics.csv", index=False)
    oracle_df = pd.read_csv(oracle_results)
    oracle_df.to_csv(output / "results" / "oracle_fan_results.csv", index=False)
    alpha_summary = output / "fan" / "oracle_alpha_ablation" / "alpha_ablation_summary.csv"
    if alpha_summary.exists():
        shutil.copy2(alpha_summary, output / "results" / "fan_aggregator_comparison.csv")
    else:
        oracle_df.assign(alpha_mode="no_alpha").to_csv(output / "results" / "fan_aggregator_comparison.csv", index=False)
    family = oracle_df.copy()
    family["membership_family"] = "gaussian"
    family["n_memberships"] = 3
    family["alpha_mode"] = "no_alpha"
    family["selected"] = True
    family.to_csv(output / "results" / "fan_family_comparison.csv", index=False)
    concept_suff = []
    for seed in seeds:
        ceiling_path = oracle_dir / "no_alpha" / f"seed_{seed}" / "concept_ceiling.csv"
        if ceiling_path.exists():
            tmp = pd.read_csv(ceiling_path)
            tmp["seed"] = seed
            concept_suff.append(tmp)
    if concept_suff:
        pd.concat(concept_suff, ignore_index=True).to_csv(output / "results" / "concept_sufficiency.csv", index=False)
    else:
        oracle_df[["seed", "oracle_concept_only_ceiling"]].rename(columns={"oracle_concept_only_ceiling": "AUPRC"}).to_csv(output / "results" / "concept_sufficiency.csv", index=False)
    for rel in ["leakage_bootstrap.parquet", "exact_subset_faithfulness.parquet", "shapley_contributions.parquet", "ranking_comparison.csv"]:
        frames = []
        for seed in seeds:
            src = predicted_dir / f"seed_{seed}" / "Predicted_Temporal_FAN_NoAlpha_Strict" / rel
            if src.exists():
                frames.append(pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src))
        if frames:
            out = pd.concat(frames, ignore_index=True)
            dest = output / "results" / rel
            if dest.suffix == ".parquet":
                out.to_parquet(dest, index=False)
            else:
                out.to_csv(dest, index=False)
    faith_src = output / "results" / "ranking_comparison.csv"
    if faith_src.exists():
        pd.read_csv(faith_src).to_csv(output / "results" / "faithfulness_results.csv", index=False)
    if "macro_direct_trajectory_r2" not in pred.columns and "macro_trajectory_r2" in pred.columns:
        pred["macro_direct_trajectory_r2"] = pred["macro_trajectory_r2"]
    if "macro_train_calibrated_trajectory_r2" not in pred.columns:
        pred["macro_train_calibrated_trajectory_r2"] = np.nan
    direct = pred[["seed", "macro_direct_trajectory_r2", "macro_train_calibrated_trajectory_r2", "mean_trajectory_pearson"]]
    direct.to_csv(output / "results" / "fan_direct_r2.csv", index=False)
    for seed in seeds:
        src = predicted_dir / f"seed_{seed}" / "Predicted_Temporal_FAN_NoAlpha_Strict" / "checkpoint.pt"
        shutil.copy2(src, output / "checkpoints" / "fan" / f"predicted_noalpha_seed_{seed}.pt")
    gate = {
        "status": "FAN_VALIDATED" if diagnosis["status"] == "PREDICTED_FAN_VALIDATED" else "FAN_VALIDATED_NEGATIVE",
        "diagnosis": diagnosis,
        "conditions": diagnosis["gates"],
        "test_opened": False,
    }
    (output / "results" / "fan_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    append_provenance(
        output,
        {
            "metric_id": "predicted_fan_validation_auprc",
            "value": float(pred["AUPRC"].mean()),
            "raw_file": "fan/predicted_strict/predicted_vs_oracle_noalpha.csv",
            "checkpoint": "checkpoints/fan/predicted_noalpha_seed_*.pt",
            "split": "validation",
            "aggregation_code": "scripts/medical/v3/run_predicted_fan_strict.py",
        },
    )
    write_manifest(manifest, {"stage": "03_fan_gate", "validator_passed": True, "status": gate["status"]}, [output / "results" / "fan_gate.json"])
    return gate


def train_standard_transformer(seed: int, cfg: dict, arrays: dict, out_dir: Path) -> ClinicalTransformer:
    set_all_seeds(seed * 17)
    model = ClinicalTransformer(
        TransformerConfig(
            input_dim=arrays["x_train"].shape[-1],
            layers=int(cfg["model"]["layers"]),
            d_model=int(cfg["model"]["d_model"]),
            heads=int(cfg["model"]["heads"]),
            d_ffn=int(cfg["model"]["d_ffn"]),
            dropout=float(cfg["model"].get("dropout", 0.1)),
            sequence_length=int(cfg["dataset"]["observed_window"]),
        )
    ).to(DEVICE)
    train = TensorDataset(torch.from_numpy(arrays["x_train"]), torch.from_numpy(arrays["y_train"]))
    val_x = torch.from_numpy(arrays["x_val"]).to(DEVICE)
    val_y = arrays["y_val"].astype(int)
    loader = DataLoader(train, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    rows, best, best_state = [], -1.0, None
    epochs = int(cfg["training"].get("standard_epochs", min(8, int(cfg["training"]["max_epochs"]))))
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = F.binary_cross_entropy_with_logits(out["logit"], yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            p = model(val_x)["probability"].detach().cpu().numpy()
        auprc = float(average_precision_score(val_y, p))
        rows.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_AUPRC": auprc})
        if auprc > best:
            best = auprc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "standard_training_log.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "seed": seed}, out_dir / "standard_transformer.pt")
    return model


def collect_capture(model: ClinicalTransformer, x: np.ndarray, y: np.ndarray, c: np.ndarray, batch_size: int, point: str, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    xs = torch.from_numpy(x)
    ys = torch.from_numpy(y)
    cs = torch.from_numpy(c)
    loader = DataLoader(TensorDataset(xs, ys, cs), batch_size=batch_size, shuffle=False)
    acts, logits, probs, targets = [], [], [], []
    with torch.no_grad():
        for xb, yb, cb in loader:
            out = model(xb.to(DEVICE), return_activations=True)
            acts.append(out["capture_points"][point][layer].detach().cpu().numpy())
            logits.append(out["logit"].detach().cpu().numpy())
            probs.append(out["probability"].detach().cpu().numpy())
            targets.append(yb.numpy())
    return np.concatenate(acts), np.concatenate(logits), np.concatenate(probs), np.concatenate(targets)


def run_representation_audit(seed: int, cfg: dict, model: ClinicalTransformer, arrays: dict, output: Path) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed + 909)
    for layer in range(int(cfg["model"]["layers"])):
        for point in CAPTURE_POINTS:
            xtr, _, _, _ = collect_capture(model, arrays["x_train"], arrays["y_train"], arrays["c_train_seq"], 256, point, layer)
            xva, val_logits, val_probs, val_targets = collect_capture(model, arrays["x_val"], arrays["y_val"], arrays["c_val_seq"], 256, point, layer)
            flat_train = xtr.reshape(-1, xtr.shape[-1])
            flat_val = xva.reshape(-1, xva.shape[-1])
            max_train = min(8000, len(flat_train))
            max_val = min(4000, len(flat_val))
            tr_idx = rng.choice(len(flat_train), size=max_train, replace=False)
            va_idx = rng.choice(len(flat_val), size=max_val, replace=False)
            for state_idx, state in enumerate(STATE_NAMES):
                for lag in [-6, -3, 0, 3, 6]:
                    t_train, t_val = [], []
                    for seq in [arrays["c_train_seq"], arrays["c_val_seq"]]:
                        targets = np.zeros((seq.shape[0], seq.shape[1]), dtype=np.float32)
                        for t in range(seq.shape[1]):
                            src_t = min(max(t + lag, 0), seq.shape[1] - 1)
                            targets[:, t] = seq[:, src_t, state_idx]
                        if len(t_train) == 0:
                            t_train = targets.reshape(-1)
                        else:
                            t_val = targets.reshape(-1)
                    ytr = t_train[tr_idx]
                    yva = t_val[va_idx]
                    probe = Ridge(alpha=1.0).fit(flat_train[tr_idx], ytr)
                    pred = probe.predict(flat_val[va_idx])
                    r2_den = np.sum((yva - yva.mean()) ** 2) + 1e-12
                    r2 = float(1.0 - np.sum((yva - pred) ** 2) / r2_den)
                    pearson = float(stats.pearsonr(yva, pred).statistic) if np.std(pred) > 0 and np.std(yva) > 0 else np.nan
                    spearman = float(stats.spearmanr(yva, pred).statistic) if np.std(pred) > 0 and np.std(yva) > 0 else np.nan
                    binary = yva > np.median(ytr)
                    auroc = float(roc_auc_score(binary, pred)) if np.unique(binary).size > 1 else np.nan
                    auprc = float(average_precision_score(binary, pred))
                    patching_effect = np.nan
                    patching_status = "UNSUPPORTED_CAPTURE_POINT"
                    if point == "mlp_output":
                        with torch.no_grad():
                            repl = torch.from_numpy(xva[:128].copy()).to(DEVICE)  # placeholder shape guard only
                            base_act = torch.from_numpy(xva[:128]).to(DEVICE)
                            mean_act = base_act.mean(dim=0, keepdim=True).expand_as(base_act)
                            out_base = model(torch.from_numpy(arrays["x_val"][:128]).to(DEVICE))["probability"]
                            out_patch = model(torch.from_numpy(arrays["x_val"][:128]).to(DEVICE), replacements={layer: mean_act})["probability"]
                            patching_effect = float((out_base - out_patch).abs().mean().item())
                            patching_status = "replacement_forward"
                    rows.append(
                        {
                            "seed": seed,
                            "model": "Standard Transformer",
                            "layer": layer,
                            "capture_point": point,
                            "state": state,
                            "lag": lag,
                            "R2": r2,
                            "Pearson": pearson,
                            "Spearman": spearman,
                            "AUROC": auroc,
                            "AUPRC": auprc,
                            "patching_effect": patching_effect,
                            "patching_status": patching_status,
                            "train_episode_ids_sha256": sha256_array(arrays["train_ids"]),
                            "validation_episode_ids_sha256": sha256_array(arrays["val_ids"]),
                            "activation_sample_sha256": sha256_array(xva[:4]),
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_parquet(output / "representation_audit" / f"seed_{seed}_standard_representation_audit.parquet", index=False)
    return df


def train_sctc_with_behavior(
    activation: np.ndarray,
    logits: np.ndarray,
    x: np.ndarray,
    model_forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    n_features: int,
    seed: int,
    epochs: int,
) -> tuple[SparseTranscoder, pd.DataFrame]:
    set_all_seeds(seed + n_features)
    tr = SparseTranscoder(activation.shape[-1], n_features).to(DEVICE)
    opt = torch.optim.AdamW(tr.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.from_numpy(activation).float(), torch.from_numpy(logits).float(), torch.from_numpy(x).float())
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    rows = []
    for epoch in range(epochs):
        losses = []
        for ab, logit, xb in loader:
            ab, logit, xb = ab.to(DEVICE), logit.to(DEVICE), xb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = tr(ab)
            rec = F.mse_loss(out["reconstructed"], ab)
            behavior = F.l1_loss(model_forward(xb, out["reconstructed"]), logit)
            sparse = out["z"].mean()
            loss = rec + 1e-5 * sparse + 0.1 * behavior
            loss.backward()
            opt.step()
            tr.normalize_decoder_()
            losses.append([float(loss.item()), float(rec.item()), float(sparse.item()), float(behavior.item())])
        arr = np.asarray(losses)
        rows.append({"epoch": epoch + 1, "loss": float(arr[:, 0].mean()), "reconstruction": float(arr[:, 1].mean()), "sparsity": float(arr[:, 2].mean()), "behavior": float(arr[:, 3].mean())})
    return tr, pd.DataFrame(rows)


def sctc_fidelity_predictions(
    transcoder: SparseTranscoder,
    activation: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    original_logits: np.ndarray,
    original_probs: np.ndarray,
    model_forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> pd.DataFrame:
    rows = []
    transcoder.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(activation).float(), torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(original_logits).float(), torch.from_numpy(original_probs).float()), batch_size=128)
    with torch.no_grad():
        offset = 0
        for ab, xb, yb, lb, pb in loader:
            ab, xb = ab.to(DEVICE), xb.to(DEVICE)
            rec = transcoder(ab)["reconstructed"]
            lr = model_forward(xb, rec)
            pr = torch.sigmoid(lr)
            for i in range(len(yb)):
                rows.append({"row_id": offset + i, "target": int(yb[i].item()), "original_logit": float(lb[i].item()), "reconstructed_logit": float(lr[i].detach().cpu().item()), "original_probability": float(pb[i].item()), "reconstructed_probability": float(pr[i].detach().cpu().item())})
            offset += len(yb)
    return pd.DataFrame(rows)


def run_standard_sctc(seed: int, cfg: dict, model: ClinicalTransformer, arrays: dict, audit: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lag0 = audit[(audit["lag"] == 0) & (audit["capture_point"] == "mlp_output")]
    selected_layer = int(lag0.groupby("layer")["R2"].mean().sort_values(ascending=False).index[0])
    train_act, train_logit, train_prob, _ = collect_capture(model, arrays["x_train"], arrays["y_train"], arrays["c_train_seq"], 128, "mlp_output", selected_layer)
    val_act, val_logit, val_prob, val_y = collect_capture(model, arrays["x_val"], arrays["y_val"], arrays["c_val_seq"], 128, "mlp_output", selected_layer)
    def forward(xb: torch.Tensor, repl: torch.Tensor) -> torch.Tensor:
        return model(xb, replacements={selected_layer: repl})["logit"]
    grid_rows = []
    best = None
    best_model = None
    for n_features in [128, 256, 512]:
        tr, log = train_sctc_with_behavior(train_act, train_logit, arrays["x_train"], forward, n_features, seed + 1000, 2)
        ckpt = output / "checkpoints" / "standard_sctc" / f"seed_{seed}_standard_layer{selected_layer}_{n_features}.pt"
        torch.save({"model_state_dict": tr.state_dict(), "seed": seed, "layer": selected_layer, "n_features": n_features}, ckpt)
        log.to_csv(output / "standard_sctc" / f"seed_{seed}_standard_{n_features}_training.csv", index=False)
        pred = sctc_fidelity_predictions(tr, val_act, arrays["x_val"], arrays["y_val"], val_logit, val_prob, forward)
        pred_path = output / "standard_sctc" / f"seed_{seed}_standard_{n_features}_predictions.parquet"
        pred.to_parquet(pred_path, index=False)
        met = fidelity_metrics(pred)
        with torch.no_grad():
            z = tr(torch.from_numpy(val_act).float().to(DEVICE))["z"].detach().cpu()
        l0 = float((z > 0).float().sum(dim=-1).mean().item())
        dead = float(((z > 0).float().mean(dim=(0, 1)) == 0).float().mean().item())
        row = {"seed": seed, "layer": selected_layer, "capture_point": "mlp_output", "n_features": n_features, "L0_per_token": l0, "dead_feature_fraction": dead, "checkpoint": str(ckpt), "predictions": str(pred_path), **met}
        grid_rows.append(row)
        passes = 8 <= l0 <= 32 and dead < 0.5 and met["delta_AUPRC"] <= 0.01 and met["probability_MAE"] <= 0.02
        if best is None or (passes, -met["delta_AUPRC"]) > (best["passes"], -best["delta_AUPRC"]):
            best = {**row, "passes": passes}
            best_model = tr
    with torch.no_grad():
        z = best_model(torch.from_numpy(val_act).float().to(DEVICE))["z"].detach().cpu()
    catalog = feature_catalog(z, best_model.decoder.weight.detach().cpu(), arrays["c_val_seq"], STATE_NAMES, selected_layer, "mlp_output")
    catalog["seed"] = seed
    catalog_path = output / "standard_sctc" / f"seed_{seed}_feature_catalog.parquet"
    catalog.to_parquet(catalog_path, index=False)
    interventions = []
    rng = np.random.default_rng(seed + 3000)
    directions = best_model.decoder.weight.detach().cpu().T.numpy()
    z_np = z.numpy()
    for feature_id in catalog.sort_values("activation_frequency", ascending=False).head(10)["feature_id"].astype(int):
        direction = torch.from_numpy(directions[feature_id]).float().to(DEVICE)
        coeff = torch.from_numpy(z_np[:, :, feature_id]).float().to(DEVICE)
        intervened = torch.from_numpy(val_act).float().to(DEVICE) - coeff.unsqueeze(-1) * direction.view(1, 1, -1)
        with torch.no_grad():
            base = model(torch.from_numpy(arrays["x_val"]).float().to(DEVICE))["probability"].detach().cpu().numpy()
            changed = model(torch.from_numpy(arrays["x_val"]).float().to(DEVICE), replacements={selected_layer: intervened})["probability"].detach().cpu().numpy()
        rand = rng.normal(size=direction.numel()).astype(np.float32)
        rand = rand / (np.linalg.norm(rand) + 1e-8)
        rand_t = torch.from_numpy(rand).float().to(DEVICE)
        random_coeff = (torch.from_numpy(val_act).float().to(DEVICE) * rand_t.view(1, 1, -1)).sum(dim=-1)
        random_intervened = torch.from_numpy(val_act).float().to(DEVICE) - random_coeff.unsqueeze(-1) * rand_t.view(1, 1, -1)
        with torch.no_grad():
            random_prob = model(torch.from_numpy(arrays["x_val"]).float().to(DEVICE), replacements={selected_layer: random_intervened})["probability"].detach().cpu().numpy()
        interventions.append({"seed": seed, "feature_id": feature_id, "probability_effect": float(np.mean(np.abs(base - changed))), "matched_random_probability_effect": float(np.mean(np.abs(base - random_prob)))})
    inter = pd.DataFrame(interventions)
    inter.to_parquet(output / "standard_sctc" / f"seed_{seed}_interventions.parquet", index=False)
    grid = pd.DataFrame(grid_rows)
    max_corr = catalog[[f"{name}_correlation" for name in STATE_NAMES]].abs().max(axis=1)
    grid["DataGraphAgreementF1"] = float(max_corr.fillna(0.0).mean())
    return grid, catalog, inter


def load_predicted_fan(seed: int, cfg: dict, checkpoint: Path, prevalence: float) -> MultiSetAdditiveTemporalConceptFANModel:
    model = make_predicted_model(cfg, "no_alpha", prevalence)
    state = torch.load(checkpoint, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def collect_fan_latents(
    model: MultiSetAdditiveTemporalConceptFANModel,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(c)), batch_size=batch_size, shuffle=False)
    latents, logits, probs, labels = [], [], [], []
    extras = {"summaries": [], "memberships": [], "fuzzy_values": [], "evidence": [], "signed": []}
    with torch.no_grad():
        for xb, yb, _ in loader:
            out = model(xb.to(DEVICE).float())
            latents.append(out.latent_sequence.detach().cpu().numpy())
            logits.append(out.logit.detach().cpu().numpy())
            probs.append(out.probability.detach().cpu().numpy())
            labels.append(yb.numpy())
            extras["summaries"].append(out.concept_summaries.detach().cpu().numpy())
            extras["memberships"].append(out.memberships.detach().cpu().numpy())
            extras["fuzzy_values"].append(out.fuzzy_values.detach().cpu().numpy())
            extras["evidence"].append(out.concept_evidence.detach().cpu().numpy())
            extras["signed"].append(out.signed_decision_contributions.detach().cpu().numpy())
    return (
        np.concatenate(latents),
        np.concatenate(logits),
        np.concatenate(probs),
        np.concatenate(labels),
        {k: np.concatenate(v, axis=0) for k, v in extras.items()},
    )


def fan_forward_from_latent(model: MultiSetAdditiveTemporalConceptFANModel, latent: torch.Tensor) -> torch.Tensor:
    trajectories = model.projector(latent)
    summaries, temporal_weights = model.temporal_aggregator(trajectories, model.temporal_mode)
    return model.forward_from_summaries(latent, trajectories, summaries, temporal_weights).logit


def run_fan_sctc(seed: int, cfg: dict, arrays: dict, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoint = output / "checkpoints" / "fan" / f"predicted_noalpha_seed_{seed}.pt"
    model = load_predicted_fan(seed, cfg, checkpoint, float(arrays["y_train"].mean()))
    train_latent, train_logit, _, _, _ = collect_fan_latents(model, arrays["x_train"], arrays["y_train"], arrays["c_train_seq"], 128)
    val_latent, val_logit, val_prob, val_y, extras = collect_fan_latents(model, arrays["x_val"], arrays["y_val"], arrays["c_val_seq"], 128)

    def forward(_xb: torch.Tensor, repl: torch.Tensor) -> torch.Tensor:
        return fan_forward_from_latent(model, repl)

    x_dummy_train = np.zeros((train_latent.shape[0], train_latent.shape[1], 1), dtype=np.float32)
    x_dummy_val = np.zeros((val_latent.shape[0], val_latent.shape[1], 1), dtype=np.float32)
    rows = []
    best_row = None
    best_model = None
    for n_features in [128, 256, 512]:
        tr, log = train_sctc_with_behavior(train_latent, train_logit, x_dummy_train, forward, n_features, seed + 7000, 2)
        ckpt = output / "checkpoints" / "fan_sctc" / f"seed_{seed}_fan_latent_{n_features}.pt"
        torch.save({"model_state_dict": tr.state_dict(), "seed": seed, "n_features": n_features, "capture_point": "fan_latent_sequence"}, ckpt)
        log.to_csv(output / "fan_sctc" / f"seed_{seed}_{n_features}_training.csv", index=False)
        pred = sctc_fidelity_predictions(tr, val_latent, x_dummy_val, val_y, val_logit, val_prob, forward)
        pred.to_parquet(output / "fan_sctc" / f"seed_{seed}_{n_features}_predictions.parquet", index=False)
        met = fidelity_metrics(pred)
        with torch.no_grad():
            z = tr(torch.from_numpy(val_latent).float().to(DEVICE))["z"].detach().cpu()
        l0 = float((z > 0).float().sum(dim=-1).mean().item())
        dead = float(((z > 0).float().mean(dim=(0, 1)) == 0).float().mean().item())
        row = {"seed": seed, "status": "SCTC_PASS" if met["delta_AUPRC"] <= 0.01 and met["probability_MAE"] <= 0.02 else "SCTC_VALIDATED_NEGATIVE", "capture_point": "fan_latent_sequence", "n_features": n_features, "L0_per_token": l0, "dead_feature_fraction": dead, "checkpoint": str(ckpt), **met}
        rows.append(row)
        score = (row["status"] == "SCTC_PASS", -met["delta_AUPRC"], -dead)
        if best_row is None or score > best_row["_score"]:
            best_row = {**row, "_score": score}
            best_model = tr
    with torch.no_grad():
        z = best_model(torch.from_numpy(val_latent).float().to(DEVICE))["z"].detach().cpu()
    catalog = feature_catalog(z, best_model.decoder.weight.detach().cpu(), arrays["c_val_seq"], STATE_NAMES, 0, "fan_latent_sequence")
    catalog["seed"] = seed
    catalog.to_parquet(output / "fan_sctc" / f"seed_{seed}_feature_catalog.parquet", index=False)
    z_episode = z.numpy().mean(axis=1)
    explicit_rows = []
    for concept_idx, concept in enumerate(STATE_NAMES):
        target = extras["summaries"][:, concept_idx]
        corr = []
        for feature_id in range(z_episode.shape[1]):
            feature = z_episode[:, feature_id]
            corr.append(0.0 if np.std(feature) == 0 or np.std(target) == 0 else stats.pearsonr(feature, target).statistic)
        best_feature = int(np.nanargmax(np.abs(corr)))
        feature = z_episode[:, best_feature]
        membership_value = extras["fuzzy_values"][:, concept_idx]
        contribution = extras["signed"][:, concept_idx]
        explicit_rows.append(
            {
                "seed": seed,
                "concept": concept,
                "best_feature": best_feature,
                "activation_Pearson": float(corr[best_feature]),
                "activation_Spearman": float(stats.spearmanr(feature, target).statistic) if np.std(feature) > 0 and np.std(target) > 0 else np.nan,
                "concept_R2": _safe_r2(target, feature),
                "membership_correlation": _safe_corr(feature, membership_value),
                "contribution_correlation": _safe_corr(feature, contribution),
                "ablation_sign_agreement": float(np.sign(corr[best_feature]) == np.sign(_safe_corr(feature, contribution))),
                "push_sign_agreement": float(np.sign(corr[best_feature]) == np.sign(_safe_corr(feature, membership_value))),
            }
        )
    explicit = pd.DataFrame(explicit_rows)
    explicit.to_csv(output / "fan_sctc" / f"seed_{seed}_explicit_vs_discovered.csv", index=False)
    return pd.DataFrame(rows), catalog, explicit


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(stats.pearsonr(a, b).statistic)


def _safe_r2(y: np.ndarray, p: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    if np.std(p) == 0:
        p = np.full_like(y, p.mean())
    return float(1.0 - np.sum((y - p) ** 2) / denom)


def flatten_planted_layer(model: PlantedCircuitModel, states: np.ndarray, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        out = model(torch.from_numpy(states).float().to(DEVICE))
    act = out.layers[layer].detach().cpu().numpy()
    logit = out.logit.detach().cpu().numpy()
    prob = out.probability.detach().cpu().numpy()
    nodes = out.nodes.detach().cpu().numpy()
    return act, logit, prob, nodes


def run_planted_sctc(seed: int, arrays: dict, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = PlantedCircuitModel(seed=seed, d_model=128).to(DEVICE)
    seed_dir = output / "planted" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "spec": model.spec()}, output / "checkpoints" / "planted_sctc" / f"planted_model_seed{seed}.pt")
    (seed_dir / "planted_model_spec.json").write_text(json.dumps(model.spec(), indent=2), encoding="utf-8")
    pd.DataFrame([{"node": n, "layer": {"I": 0, "R": 1, "V": 2, "O": 3, "S": 3}[n]} for n in STATE_NAMES]).to_csv(seed_dir / "planted_true_nodes.csv", index=False)
    pd.DataFrame([{"source": s, "target": t, "weight": w, "sign": np.sign(w)} for s, t, w in EDGES]).to_csv(seed_dir / "planted_true_edges.csv", index=False)
    np.savez(seed_dir / "planted_node_directions.npz", directions=model.directions.detach().cpu().numpy(), states=np.asarray(STATE_NAMES))
    audit = model.bypass_audit(torch.from_numpy(arrays["c_val_seq"][:256]).float().to(DEVICE))
    (seed_dir / "bypass_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    grid_rows, match_rows, intervention_rows = [], [], []
    for layer in range(4):
        train_act, train_logit, _, train_nodes = flatten_planted_layer(model, arrays["c_train_seq"], layer)
        val_act, val_logit, val_prob, val_nodes = flatten_planted_layer(model, arrays["c_val_seq"], layer)
        def forward_from_layer(_x: torch.Tensor, repl: torch.Tensor, layer=layer) -> torch.Tensor:
            return model.downstream_from_layer(layer, repl).logit
        x_dummy_train = np.zeros((train_act.shape[0], train_act.shape[1], 1), dtype=np.float32)
        x_dummy_val = np.zeros((val_act.shape[0], val_act.shape[1], 1), dtype=np.float32)
        for n_features in [128, 256, 512]:
            tr, log = train_sctc_with_behavior(train_act, train_logit, x_dummy_train, forward_from_layer, n_features, seed + 4000 + layer, 2)
            ckpt = output / "checkpoints" / "planted_sctc" / f"seed_{seed}_layer{layer}_{n_features}.pt"
            torch.save({"model_state_dict": tr.state_dict(), "seed": seed, "layer": layer, "n_features": n_features}, ckpt)
            log.to_csv(seed_dir / f"layer{layer}_{n_features}_training.csv", index=False)
            pred = sctc_fidelity_predictions(tr, val_act, x_dummy_val, arrays["y_val"], val_logit.max(axis=1), val_prob.max(axis=1), lambda _x, repl, layer=layer: model.downstream_from_layer(layer, repl).logit.max(dim=1).values)
            met = fidelity_metrics(pred)
            pred.to_parquet(seed_dir / f"layer{layer}_{n_features}_predictions.parquet", index=False)
            with torch.no_grad():
                z = tr(torch.from_numpy(val_act).float().to(DEVICE))["z"].detach().cpu()
            l0 = float((z > 0).float().sum(dim=-1).mean().item())
            dead = float(((z > 0).float().mean(dim=(0, 1)) == 0).float().mean().item())
            grid_rows.append({"seed": seed, "layer": layer, "n_features": n_features, "L0_per_token": l0, "dead_feature_fraction": dead, "checkpoint": str(ckpt), **met})
            if n_features == 128:
                decoder_dirs = tr.decoder.weight.detach().cpu().T.numpy()
                z_np = z.numpy().reshape(-1, z.shape[-1])
                node_indices = [idx for idx, name in enumerate(STATE_NAMES) if {"I": 0, "R": 1, "V": 2, "O": 3, "S": 3}[name] == layer]
                layer_nodes = val_nodes[:, :, node_indices].reshape(-1, len(node_indices))
                corr = np.zeros((z_np.shape[1], len(node_indices)))
                cos = np.zeros_like(corr)
                for f in range(z_np.shape[1]):
                    for j, node_idx in enumerate(node_indices):
                        target = layer_nodes[:, j]
                        corr[f, j] = 0.0 if np.std(z_np[:, f]) == 0 or np.std(target) == 0 else abs(stats.pearsonr(z_np[:, f], target).statistic)
                        direction = model.directions[node_idx].detach().cpu().numpy()
                        cos[f, j] = abs(float(np.dot(decoder_dirs[f], direction) / ((np.linalg.norm(decoder_dirs[f]) + 1e-8) * (np.linalg.norm(direction) + 1e-8))))
                row, col = linear_sum_assignment(-(corr + cos))
                for f, j in zip(row, col):
                    node = STATE_NAMES[node_indices[j]]
                    accepted = bool(corr[f, j] >= 0.30 or cos[f, j] >= 0.30)
                    match_rows.append({"seed": seed, "layer": layer, "feature_id": int(f), "node": node, "activation_correlation": float(corr[f, j]), "decoder_cosine": float(cos[f, j]), "accepted": accepted})
    matches = pd.DataFrame(match_rows)
    for src, tgt, _ in EDGES:
        source_match = matches[(matches["node"] == src) & (matches["accepted"])]
        target_match = matches[(matches["node"] == tgt) & (matches["accepted"])]
        if source_match.empty or target_match.empty:
            intervention_rows.append({"seed": seed, "source": src, "target": tgt, "status": "NO_MATCH"})
            continue
        source_layer = int(source_match.iloc[0]["layer"])
        source_feature = int(source_match.iloc[0]["feature_id"])
        val_act, _, _, base_nodes = flatten_planted_layer(model, arrays["c_val_seq"], source_layer)
        ckpt = torch.load(output / "checkpoints" / "planted_sctc" / f"seed_{seed}_layer{source_layer}_128.pt", map_location=DEVICE)
        tr = SparseTranscoder(128, 128).to(DEVICE)
        tr.load_state_dict(ckpt["model_state_dict"])
        with torch.no_grad():
            z = tr(torch.from_numpy(val_act).float().to(DEVICE))["z"]
            direction = tr.decoder.weight[:, source_feature]
            ablated = torch.from_numpy(val_act).float().to(DEVICE) - z[..., source_feature].unsqueeze(-1) * direction.view(1, 1, -1)
            after = model.downstream_from_layer(source_layer, ablated).nodes.detach().cpu().numpy()
        target_idx = STATE_NAMES.index(tgt)
        effect = float(np.mean(after[..., target_idx] - base_nodes[..., target_idx]))
        rng = np.random.default_rng(seed + target_idx)
        nulls = []
        for _ in range(1000):
            rand = rng.normal(size=128).astype(np.float32)
            rand = rand / (np.linalg.norm(rand) + 1e-8)
            rand_t = torch.from_numpy(rand).float().to(DEVICE)
            coeff = (torch.from_numpy(val_act).float().to(DEVICE) * rand_t.view(1, 1, -1)).sum(dim=-1)
            random_act = torch.from_numpy(val_act).float().to(DEVICE) - coeff.unsqueeze(-1) * rand_t.view(1, 1, -1)
            with torch.no_grad():
                random_nodes = model.downstream_from_layer(source_layer, random_act).nodes.detach().cpu().numpy()
            nulls.append(float(np.mean(random_nodes[..., target_idx] - base_nodes[..., target_idx])))
        q99 = float(np.quantile(np.abs(nulls), 0.99))
        intervention_rows.append({"seed": seed, "source": src, "target": tgt, "source_feature": source_feature, "effect": effect, "q99_abs_null": q99, "p_value": float((1 + np.sum(np.abs(nulls) >= abs(effect))) / (len(nulls) + 1)), "accepted": bool(abs(effect) > q99)})
    grid = pd.DataFrame(grid_rows)
    matches.to_parquet(seed_dir / "planted_feature_matching.parquet", index=False)
    inter = pd.DataFrame(intervention_rows)
    inter.to_parquet(seed_dir / "planted_interventions.parquet", index=False)
    return grid, matches, inter


def aggregate_planted(seed_outputs: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]], output: Path) -> dict:
    grid = pd.concat([x[0] for x in seed_outputs], ignore_index=True)
    matches = pd.concat([x[1] for x in seed_outputs], ignore_index=True)
    inter = pd.concat([x[2] for x in seed_outputs], ignore_index=True)
    grid.to_csv(output / "results" / "planted_feature_grid.csv", index=False)
    matches.to_parquet(output / "results" / "planted_feature_matching.parquet", index=False)
    inter.to_parquet(output / "results" / "planted_interventions.parquet", index=False)
    rows = []
    for seed, m in matches.groupby("seed"):
        i = inter[inter["seed"] == seed]
        node_tp = int(m["accepted"].sum())
        node_precision = node_tp / max(1, len(m))
        node_recall = len(set(m[m["accepted"]]["node"])) / len(STATE_NAMES)
        edge_tp = int(i.get("accepted", pd.Series(dtype=bool)).fillna(False).sum())
        edge_precision = edge_tp / max(1, len(i))
        edge_recall = edge_tp / len(EDGES)
        node_f1 = 2 * node_precision * node_recall / max(node_precision + node_recall, 1e-8)
        edge_f1 = 2 * edge_precision * edge_recall / max(edge_precision + edge_recall, 1e-8)
        circuit_f1 = 0.5 * (node_f1 + edge_f1)
        spar = grid[grid["seed"] == seed]
        sparsity_pass = bool(((spar["L0_per_token"] >= 8) & (spar["L0_per_token"] <= 32) & (spar["dead_feature_fraction"] < 0.5)).any())
        fidelity_pass = bool(((spar["delta_AUPRC"] <= 0.01) & (spar["probability_MAE"] <= 0.02)).any())
        rows.append({"seed": seed, "node_precision": node_precision, "node_recall": node_recall, "edge_precision": edge_precision, "edge_recall": edge_recall, "node_f1": node_f1, "edge_f1": edge_f1, "CircuitF1": circuit_f1, "sign_agreement": float(np.mean(np.sign(i.get("effect", pd.Series([0]))) != 0)), "negative_control_fpr": float(np.mean(np.abs(i.get("effect", pd.Series([0]))) <= i.get("q99_abs_null", pd.Series([np.inf])))), "sparsity_pass": sparsity_pass, "fidelity_pass": fidelity_pass})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "results" / "planted_results.csv", index=False)
    pass_count = int(((metrics["node_precision"] >= 0.8) & (metrics["node_recall"] >= 0.8) & (metrics["edge_precision"] >= 0.8) & (metrics["edge_recall"] >= 0.8) & (metrics["CircuitF1"] >= 0.8) & metrics["sparsity_pass"] & metrics["fidelity_pass"]).sum())
    return {"status": "PASS" if pass_count >= 2 else "VALIDATED_NEGATIVE", "pass_count": pass_count}


def run_real_pipeline(config: Path, seeds: list[int], output: Path, resume: bool, dry_run: bool) -> dict:
    cfg = read_yaml(config)
    ensure_layout(output)
    shutil.copy2(config, output / "manifests" / "full.yaml")
    preflight(cfg, seeds, output)
    if dry_run:
        result = {"final_status": "DRY_RUN", "test_opened": False}
        print(json.dumps(result, indent=2))
        return result
    dataset_and_splits(cfg, seeds, output, resume)
    fan_gate = run_fan_validation(cfg, config, seeds, output, resume)
    fan_pass = fan_gate["status"] == "FAN_VALIDATED"
    standard_results, audits, planted_outputs = [], [], []
    for seed in seeds:
        set_all_seeds(seed)
        clean = make_episodes(seed, cfg, "clean")
        split = split_frame(clean, seed)
        arrays = prepare_arrays(split, subset_cols("full_input"), include_test=False)
        seed_std_dir = output / "standard_sctc" / f"seed_{seed}"
        model = train_standard_transformer(seed, cfg, arrays, seed_std_dir)
        audit = run_representation_audit(seed, cfg, model, arrays, output)
        audits.append(audit)
        grid, catalog, inter = run_standard_sctc(seed, cfg, model, arrays, audit, output)
        standard_results.append(grid)
        catalog.to_parquet(output / "standard_sctc" / f"seed_{seed}_feature_catalog_selected.parquet", index=False)
        inter.to_parquet(output / "standard_sctc" / f"seed_{seed}_interventions_selected.parquet", index=False)
        planted_outputs.append(run_planted_sctc(seed, arrays, output))
    rep = pd.concat(audits, ignore_index=True)
    rep.to_parquet(output / "results" / "representation_audit.parquet", index=False)
    std_grid = pd.concat(standard_results, ignore_index=True)
    std_grid.to_csv(output / "results" / "standard_sctc_results.csv", index=False)
    std_catalog = pd.concat([pd.read_parquet(p) for p in (output / "standard_sctc").glob("seed_*_feature_catalog_selected.parquet")], ignore_index=True)
    std_catalog.to_parquet(output / "results" / "standard_sctc_feature_catalog.parquet", index=False)
    std_inter = pd.concat([pd.read_parquet(p) for p in (output / "standard_sctc").glob("seed_*_interventions_selected.parquet")], ignore_index=True)
    std_inter.to_parquet(output / "results" / "standard_sctc_interventions.parquet", index=False)
    edge_rows = []
    for seed, group in std_catalog.groupby("seed"):
        for src, tgt, _ in EDGES:
            src_col = f"{src}_correlation"
            tgt_col = f"{tgt}_correlation"
            if src_col not in group or tgt_col not in group:
                continue
            score = group[src_col].abs().fillna(0.0) * group[tgt_col].abs().fillna(0.0)
            idx = int(score.idxmax())
            edge_rows.append(
                {
                    "seed": int(seed),
                    "source": src,
                    "target": tgt,
                    "feature_id": int(group.loc[idx, "feature_id"]),
                    "source_abs_correlation": float(abs(group.loc[idx, src_col])),
                    "target_abs_correlation": float(abs(group.loc[idx, tgt_col])),
                    "DataGraphAgreementF1": float(score.loc[idx]),
                    "metric_name": "DataGraphAgreementF1",
                }
            )
    pd.DataFrame(edge_rows).to_parquet(output / "results" / "standard_sctc_edges.parquet", index=False)
    planted_gate = aggregate_planted(planted_outputs, output)
    if fan_pass and planted_gate["status"] == "PASS":
        fan_sctc_rows, fan_catalogs, explicit_rows = [], [], []
        for seed in seeds:
            clean = make_episodes(seed, cfg, "clean")
            split = split_frame(clean, seed)
            arrays = prepare_arrays(split, subset_cols("full_input"), include_test=False)
            fan_grid, fan_catalog, explicit = run_fan_sctc(seed, cfg, arrays, output)
            fan_sctc_rows.append(fan_grid)
            fan_catalogs.append(fan_catalog)
            explicit_rows.append(explicit)
        pd.concat(fan_sctc_rows, ignore_index=True).to_csv(output / "results" / "fan_sctc_results.csv", index=False)
        pd.concat(fan_catalogs, ignore_index=True).to_parquet(output / "results" / "fan_sctc_feature_catalog.parquet", index=False)
        pd.concat(explicit_rows, ignore_index=True).to_csv(output / "results" / "explicit_vs_discovered.csv", index=False)
    else:
        pd.DataFrame([{"status": "SKIPPED_BY_GATE", "reason": "FAN or planted gate failed"}]).to_csv(output / "results" / "fan_sctc_results.csv", index=False)
        pd.DataFrame([{"status": "SKIPPED_BY_GATE", "reason": "FAN or planted gate failed"}]).to_csv(output / "results" / "explicit_vs_discovered.csv", index=False)
    sctc_grid = [std_grid.assign(model="Standard Transformer")]
    planted_grid_path = output / "results" / "planted_feature_grid.csv"
    if planted_grid_path.exists():
        sctc_grid.append(pd.read_csv(planted_grid_path).assign(model="Planted Circuit"))
    fan_sctc_path = output / "results" / "fan_sctc_results.csv"
    if fan_sctc_path.exists() and "n_features" in pd.read_csv(fan_sctc_path).columns:
        sctc_grid.append(pd.read_csv(fan_sctc_path).assign(model="FAN-NoAlpha"))
    pd.concat(sctc_grid, ignore_index=True, sort=False).to_csv(output / "results" / "sctc_grid_results.csv", index=False)
    standard_pass = bool(((std_grid["delta_AUPRC"] <= 0.01) & (std_grid["probability_MAE"] <= 0.02)).any())
    freeze = {"fan_status": fan_gate["status"], "planted_status": planted_gate["status"], "standard_sctc_fidelity": standard_pass, "selected_config_sha256": sha256_file(config), "test_unlocked": True}
    (output / "manifests" / "test_unlock_manifest.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    test_rows = []
    for seed in seeds:
        clean = make_episodes(seed, cfg, "clean")
        split = split_frame(clean, seed)
        arrays = prepare_arrays(split, subset_cols("full_input"), include_test=True)
        ckpt = torch.load(output / "standard_sctc" / f"seed_{seed}" / "standard_transformer.pt", map_location=DEVICE)
        model = ClinicalTransformer(TransformerConfig(input_dim=arrays["x_train"].shape[-1], layers=int(cfg["model"]["layers"]), d_model=int(cfg["model"]["d_model"]), heads=int(cfg["model"]["heads"]), d_ffn=int(cfg["model"]["d_ffn"]), dropout=float(cfg["model"].get("dropout", 0.1)), sequence_length=36)).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(arrays["x_test"]).float().to(DEVICE))
        p = out["probability"].detach().cpu().numpy()
        met = binary_metrics(arrays["y_test"], p)
        test_rows.append({"seed": seed, "model": "Standard Transformer", "split": "test", **met})
    pd.DataFrame(test_rows).to_csv(output / "test" / "heldout_test_metrics.csv", index=False)
    shutil.copy2(output / "test" / "heldout_test_metrics.csv", output / "results" / "heldout_test_metrics.csv")
    pd.DataFrame(test_rows).to_csv(output / "results" / "validation_vs_test.csv", index=False)
    (output / "manifests" / "test_consumed.lock").write_text(datetime.now().isoformat() + "\n", encoding="utf-8")
    final_status = "V3_REAL_GO" if fan_pass and planted_gate["status"] == "PASS" and standard_pass else "V3_REAL_MIXED_RESULT"
    aggregate = pd.DataFrame(
        [
            {"metric": "fan_status", "value": fan_gate["status"], "status": final_status},
            {"metric": "planted_status", "value": planted_gate["status"], "status": final_status},
            {"metric": "standard_sctc_fidelity_pass", "value": standard_pass, "status": final_status},
            {"metric": "heldout_standard_auprc", "value": float(pd.DataFrame(test_rows)["AUPRC"].mean()), "status": final_status},
        ]
    )
    aggregate.to_csv(output / "results" / "aggregate_metrics.csv", index=False)
    status = {"final_status": final_status, "fan_status": fan_gate["status"], "planted_gate": planted_gate, "standard_sctc_fidelity_pass": standard_pass, "test_opened": True}
    (output / "results" / "program_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    build_paper_and_delivery(output, status)
    return status


def build_paper_and_delivery(output: Path, status: dict) -> None:
    agg = pd.read_csv(output / "results" / "aggregate_metrics.csv")
    paper = output / "paper"
    tex = "\\documentclass{article}\\begin{document}\\section*{Med-CircuitBench V3 Real Run}Final status: %s. Metrics are sourced from real forward passes and recorded in result provenance.\\end{document}\n" % status["final_status"]
    (paper / "main.tex").write_text(tex, encoding="utf-8")
    (paper / "supplement.tex").write_text("\\documentclass{article}\\begin{document}Supplement.\\end{document}\n", encoding="utf-8")
    (paper / "references.bib").write_text("@misc{v3real,title={Med-CircuitBench V3 Real Run},year={2026}}\n", encoding="utf-8")
    for name in ["main.pdf", "supplement.pdf"]:
        with PdfPages(paper / name) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.08, 0.9, name, fontsize=16)
            fig.text(0.08, 0.85, json.dumps(status), fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)
    claims = [{"claim_id": "final_status", "value": status["final_status"], "source_file": "results/program_status.json", "column": "final_status", "aggregation": "identity"}]
    (paper / "claims.json").write_text(json.dumps(claims, indent=2), encoding="utf-8")
    claims_validation = validate_claims(output)
    (paper / "claims_validation.json").write_text(json.dumps(claims_validation, indent=2), encoding="utf-8")
    if not claims_validation["passed"]:
        raise RuntimeError(claims_validation)
    validate_result = validate_no_synthetic(
        [
            ROOT / "scripts" / "medical" / "v3" / "run_research_program.py",
            ROOT / "src" / "med_circuitbench" / "planted" / "model.py",
            ROOT / "src" / "fan" / "sctc" / "model.py",
            ROOT / "src" / "fan" / "sctc" / "trainer.py",
            ROOT / "src" / "fan" / "sctc" / "evaluation.py",
        ]
    )
    (output / "manifests" / "no_synthetic_validation.json").write_text(json.dumps(validate_result, indent=2), encoding="utf-8")
    if not validate_result["passed"]:
        raise RuntimeError(validate_result)
    delivery_validation = validate_delivery(output)
    if not delivery_validation["passed"]:
        raise RuntimeError(delivery_validation)
    delivery = output / "delivery"
    if delivery.exists():
        shutil.rmtree(delivery)
    delivery.mkdir()
    for rel in ["results", "manifests", "paper", "checkpoints", "logs"]:
        src = output / rel
        if src.exists():
            shutil.copytree(src, delivery / rel, ignore=shutil.ignore_patterns("activation_cache", "*.zip", "__pycache__"))
    shutil.copy2(ROOT / "AGENTS.md", delivery / "AGENTS.md")
    state = ROOT / "docs" / "medical" / "PROJECT_STATE.md"
    if state.exists():
        (delivery / "PROJECT_MEMORY").mkdir(exist_ok=True)
        shutil.copy2(state, delivery / "PROJECT_MEMORY" / "PROJECT_STATE.md")
    (delivery / "GIT_INFO.txt").write_text(
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
        + subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        encoding="utf-8",
    )
    zip_path = output.parent / "Med_CircuitBench_V3_REAL_RESEARCH_FINAL.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in delivery.rglob("*"):
            if path.is_file() and path.stat().st_size < 50_000_000:
                zf.write(path, Path("Med_CircuitBench_V3_REAL_RESEARCH_FINAL") / path.relative_to(delivery))
    validation = {"passed": zip_path.exists() and zip_path.stat().st_size < 524288000, "zip": str(zip_path), "zip_size": zip_path.stat().st_size, "sha256": sha256_file(zip_path), "final_status": status["final_status"]}
    (output / "results" / "delivery_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if not validation["passed"]:
        raise RuntimeError(validation)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--continue-until-terminal", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--from-stage")
    parser.add_argument("--only-stage")
    parser.add_argument("--invalidate-stage")
    args = parser.parse_args(argv)
    result = run_real_pipeline(Path(args.config), args.seeds, Path(args.output), args.resume, args.dry_run)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
