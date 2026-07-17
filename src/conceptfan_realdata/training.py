from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .calibration import Calibrator, fit_calibrators, select_primary_calibrator
from .data import PreparedData, sha256_file
from .metrics import binary_metrics
from .models import (
    CONCEPT_ARMS,
    RealModelOutput,
    build_model,
    exact_decomposition_error,
    masked_concept_loss,
    normalized_contributions,
    parameter_sha256,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_text(arguments: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *arguments], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "UNKNOWN"


def environment_record(device: torch.device) -> dict:
    gpu = None
    if device.type == "cuda":
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "total_memory": int(torch.cuda.get_device_properties(device).total_memory),
            "cuda_version": torch.version.cuda,
        }
    return {
        "created_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "gpu": gpu,
    }


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    concepts: np.ndarray,
    concept_mask: np.ndarray,
    record_ids: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x[indices]).float(),
        torch.from_numpy(y[indices]).float(),
        torch.from_numpy(concepts[indices]).float(),
        torch.from_numpy(concept_mask[indices]).float(),
        torch.from_numpy(record_ids[indices]).long(),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0, pin_memory=True)


@dataclass
class Evaluation:
    record_ids: np.ndarray
    target: np.ndarray
    logits: np.ndarray
    probability: np.ndarray
    concept_trajectories: np.ndarray | None
    concept_summaries: np.ndarray | None
    contributions: np.ndarray | None
    bias: float
    decomposition_error: np.ndarray


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Evaluation:
    model.eval()
    record_ids: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    trajectories: list[np.ndarray] = []
    summaries: list[np.ndarray] = []
    contributions: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    bias = float("nan")
    with torch.no_grad():
        for x, y, _, _, ids in loader:
            x = x.to(device, non_blocking=True)
            output: RealModelOutput = model(x)
            record_ids.append(ids.numpy())
            targets.append(y.numpy())
            logits.append(output.logit.detach().cpu().numpy())
            probabilities.append(output.probability.detach().cpu().numpy())
            errors.append(exact_decomposition_error(output).detach().cpu().numpy())
            bias = float(output.bias.detach().cpu())
            if output.concept_trajectories is not None:
                trajectories.append(output.concept_trajectories.detach().cpu().numpy())
            if output.concept_summaries is not None:
                summaries.append(output.concept_summaries.detach().cpu().numpy())
            if output.contributions is not None:
                contributions.append(output.contributions.detach().cpu().numpy())
    return Evaluation(
        record_ids=np.concatenate(record_ids),
        target=np.concatenate(targets).astype(np.int8),
        logits=np.concatenate(logits),
        probability=np.concatenate(probabilities),
        concept_trajectories=np.concatenate(trajectories) if trajectories else None,
        concept_summaries=np.concatenate(summaries) if summaries else None,
        contributions=np.concatenate(contributions) if contributions else None,
        bias=bias,
        decomposition_error=np.concatenate(errors),
    )


def _save_split_outputs(run_dir: Path, split_name: str, evaluation: Evaluation, primary: Calibrator) -> None:
    probability_primary = primary.predict(evaluation.logits)
    logits = pd.DataFrame(
        {
            "RecordID": evaluation.record_ids,
            "target": evaluation.target,
            "logit_raw": evaluation.logits,
            "probability_raw": evaluation.probability,
            "probability_primary_calibrated": probability_primary,
        }
    )
    logits.to_parquet(run_dir / f"logits_{split_name}.parquet", index=False, compression="zstd")
    if evaluation.concept_summaries is not None and evaluation.concept_trajectories is not None:
        concept_rows: dict[str, object] = {"RecordID": evaluation.record_ids}
        for concept in range(evaluation.concept_summaries.shape[1]):
            concept_rows[f"concept_{concept}_summary"] = evaluation.concept_summaries[:, concept]
            concept_rows[f"concept_{concept}_trajectory"] = [row.astype(float).tolist() for row in evaluation.concept_trajectories[:, :, concept]]
        pd.DataFrame(concept_rows).to_parquet(run_dir / f"concepts_{split_name}.parquet", index=False, compression="zstd")
    else:
        pd.DataFrame({"RecordID": evaluation.record_ids, "not_applicable": True}).to_parquet(
            run_dir / f"concepts_{split_name}.parquet", index=False, compression="zstd"
        )
    if evaluation.contributions is not None:
        normalized = evaluation.contributions / (np.abs(evaluation.contributions).sum(axis=1, keepdims=True) + 1e-8)
        rows: dict[str, object] = {
            "RecordID": evaluation.record_ids,
            "bias": evaluation.bias,
            "decomposition_error": evaluation.decomposition_error,
        }
        for concept in range(evaluation.contributions.shape[1]):
            rows[f"contribution_{concept}"] = evaluation.contributions[:, concept]
            rows[f"normalized_contribution_{concept}"] = normalized[:, concept]
        pd.DataFrame(rows).to_parquet(run_dir / f"contributions_{split_name}.parquet", index=False, compression="zstd")
    else:
        pd.DataFrame({"RecordID": evaluation.record_ids, "not_applicable": True}).to_parquet(
            run_dir / f"contributions_{split_name}.parquet", index=False, compression="zstd"
        )


def _loss_for_model(
    arm: str,
    output: RealModelOutput,
    target: torch.Tensor,
    concepts: torch.Tensor,
    concept_mask: torch.Tensor,
    positive_weight: torch.Tensor,
    lambda_concept: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    task = F.binary_cross_entropy_with_logits(output.logit, target, pos_weight=positive_weight)
    if arm in CONCEPT_ARMS and output.concept_trajectories is not None:
        concept = masked_concept_loss(output.concept_trajectories, concepts, concept_mask)
    else:
        concept = task.new_zeros(())
    total = task + lambda_concept * concept
    return total, {"task_loss": float(task.detach()), "concept_loss": float(concept.detach())}


def _train_standard(
    arm: str,
    model: nn.Module,
    loaders: dict[str, DataLoader],
    config: dict,
    device: torch.device,
    resume_path: Path,
    log_path: Path,
    max_epochs_override: int | None = None,
) -> tuple[nn.Module, list[dict], dict]:
    training = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    positive = loaders["train"].dataset.tensors[1]
    positive_weight = torch.tensor(float((len(positive) - positive.sum()) / positive.sum().clamp_min(1)), device=device)
    start_epoch = 0
    best_score = -float("inf")
    best_state = None
    best_epoch = 0
    history: list[dict] = []
    stale = 0
    if resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        best_score = float(resume["best_score"])
        best_state = resume["best_state_dict"]
        best_epoch = int(resume["best_epoch"])
        history = list(resume["history"])
        stale = int(resume["stale"])
    max_epochs = int(max_epochs_override or training["max_epochs"])
    for epoch in range(start_epoch, max_epochs):
        model.train()
        batch_metrics: list[dict[str, float]] = []
        for x, y, concepts, concept_mask, _ in loaders["train"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            concepts = concepts.to(device, non_blocking=True)
            concept_mask = concept_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output: RealModelOutput = model(x)
            loss, parts = _loss_for_model(
                arm,
                output,
                y,
                concepts,
                concept_mask,
                positive_weight,
                float(training["lambda_concept"]),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {arm}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            optimizer.step()
            batch_metrics.append({"loss": float(loss.detach()), **parts})
        validation = evaluate_model(model, loaders["validation"], device)
        if not np.isfinite(validation.logits).all():
            raise FloatingPointError(f"Non-finite validation logits in {arm}")
        validation_metrics = binary_metrics(validation.target, validation.probability, validation.logits)
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean([item["loss"] for item in batch_metrics])),
            "task_loss": float(np.mean([item["task_loss"] for item in batch_metrics])),
            "concept_loss": float(np.mean([item["concept_loss"] for item in batch_metrics])),
            "validation_AUPRC": validation_metrics["AUPRC"],
            "validation_AUROC": validation_metrics["AUROC"],
        }
        history.append(row)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        if row["validation_AUPRC"] > best_score + 1e-8:
            best_score = row["validation_AUPRC"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_state_dict": best_state,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "history": history,
                "stale": stale,
            },
            resume_path,
        )
        if max_epochs_override is None and epoch + 1 >= int(training["min_epochs"]) and stale >= int(training["patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"No valid checkpoint produced for {arm}")
    model.load_state_dict(best_state)
    return model, history, {"best_epoch": best_epoch, "best_validation_AUPRC": best_score, "optimizer_state_dict": optimizer.state_dict()}


def _stability_weight(epoch: int, target: float, config: dict) -> float:
    stability = config["stability_regularization"]
    warmup = int(stability["warmup_epochs"])
    ramp = int(stability["ramp_epochs"])
    if epoch < warmup:
        return 0.0
    return float(target) * min(1.0, (epoch - warmup + 1) / max(ramp, 1))


def _train_stability_pair(
    model_a: nn.Module,
    model_b: nn.Module,
    loaders: dict[str, DataLoader],
    config: dict,
    device: torch.device,
    lambda_stab: float,
    lambda_logit: float,
    resume_path: Path,
    log_path: Path,
    max_epochs_override: int | None = None,
) -> tuple[nn.Module, nn.Module, list[dict], dict]:
    training = config["training"]
    parameters = list(model_a.parameters()) + list(model_b.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    positive = loaders["train"].dataset.tensors[1]
    positive_weight = torch.tensor(float((len(positive) - positive.sum()) / positive.sum().clamp_min(1)), device=device)
    start_epoch = 0
    best_score = -float("inf")
    best_a = None
    best_b = None
    best_epoch = 0
    history: list[dict] = []
    stale = 0
    if resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        model_a.load_state_dict(resume["model_a_state_dict"])
        model_b.load_state_dict(resume["model_b_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        best_score = float(resume["best_score"])
        best_a = resume["best_a_state_dict"]
        best_b = resume["best_b_state_dict"]
        best_epoch = int(resume["best_epoch"])
        history = list(resume["history"])
        stale = int(resume["stale"])
    max_epochs = int(max_epochs_override or training["max_epochs"])
    for epoch in range(start_epoch, max_epochs):
        model_a.train()
        model_b.train()
        losses: list[dict[str, float]] = []
        current_stab = _stability_weight(epoch, lambda_stab, config)
        for x, y, concepts, concept_mask, _ in loaders["train"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            concepts = concepts.to(device, non_blocking=True)
            concept_mask = concept_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output_a: RealModelOutput = model_a(x)
            output_b: RealModelOutput = model_b(x)
            loss_a, parts_a = _loss_for_model(
                "ConceptFAN-StabilityReg", output_a, y, concepts, concept_mask, positive_weight, float(training["lambda_concept"])
            )
            loss_b, parts_b = _loss_for_model(
                "ConceptFAN-StabilityReg", output_b, y, concepts, concept_mask, positive_weight, float(training["lambda_concept"])
            )
            normalized_a = normalized_contributions(output_a.contributions)
            normalized_b = normalized_contributions(output_b.contributions)
            contribution_consistency = F.mse_loss(normalized_a, normalized_b)
            logit_consistency = F.mse_loss(output_a.logit, output_b.logit)
            total = loss_a + loss_b + current_stab * contribution_consistency + float(lambda_logit) * logit_consistency
            if not torch.isfinite(total):
                raise FloatingPointError("Non-finite StabilityReg loss")
            total.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(training["gradient_clip"]))
            optimizer.step()
            losses.append(
                {
                    "loss": float(total.detach()),
                    "task_loss": 0.5 * (parts_a["task_loss"] + parts_b["task_loss"]),
                    "concept_loss": 0.5 * (parts_a["concept_loss"] + parts_b["concept_loss"]),
                    "contribution_consistency": float(contribution_consistency.detach()),
                    "logit_consistency": float(logit_consistency.detach()),
                }
            )
        validation_a = evaluate_model(model_a, loaders["validation"], device)
        validation_b = evaluate_model(model_b, loaders["validation"], device)
        score_a = binary_metrics(validation_a.target, validation_a.probability, validation_a.logits)["AUPRC"]
        score_b = binary_metrics(validation_b.target, validation_b.probability, validation_b.logits)["AUPRC"]
        score = 0.5 * (score_a + score_b)
        row = {
            "epoch": epoch + 1,
            "lambda_stab_effective": current_stab,
            "lambda_logit": float(lambda_logit),
            "loss": float(np.mean([item["loss"] for item in losses])),
            "task_loss": float(np.mean([item["task_loss"] for item in losses])),
            "concept_loss": float(np.mean([item["concept_loss"] for item in losses])),
            "contribution_consistency": float(np.mean([item["contribution_consistency"] for item in losses])),
            "logit_consistency": float(np.mean([item["logit_consistency"] for item in losses])),
            "validation_AUPRC_replica_a": score_a,
            "validation_AUPRC_replica_b": score_b,
            "validation_AUPRC": score,
        }
        history.append(row)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        if score > best_score + 1e-8:
            best_score = score
            best_a = {name: value.detach().cpu().clone() for name, value in model_a.state_dict().items()}
            best_b = {name: value.detach().cpu().clone() for name, value in model_b.state_dict().items()}
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        torch.save(
            {
                "epoch": epoch,
                "model_a_state_dict": model_a.state_dict(),
                "model_b_state_dict": model_b.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_a_state_dict": best_a,
                "best_b_state_dict": best_b,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "history": history,
                "stale": stale,
            },
            resume_path,
        )
        if max_epochs_override is None and epoch + 1 >= int(training["min_epochs"]) and stale >= int(training["patience"]):
            break
    if best_a is None or best_b is None:
        raise RuntimeError("No valid StabilityReg checkpoint produced")
    model_a.load_state_dict(best_a)
    model_b.load_state_dict(best_b)
    return model_a, model_b, history, {
        "best_epoch": best_epoch,
        "best_validation_AUPRC": best_score,
        "optimizer_state_dict": optimizer.state_dict(),
    }


def build_loaders(
    data: PreparedData,
    channels: str,
    batch_size: int,
    data_order_seed: int,
    splits: tuple[str, ...] = ("train", "validation", "calibration", "test"),
) -> tuple[np.ndarray, dict[str, DataLoader]]:
    x = data.inputs(channels)
    loaders = {
        split: make_loader(
            x,
            data.y,
            data.concepts,
            data.concept_mask,
            data.record_ids,
            data.indices(split),
            batch_size,
            split == "train",
            data_order_seed,
        )
        for split in splits
    }
    return x, loaders


def _metrics_for_calibrators(evaluation: Evaluation, calibrators: dict[str, Calibrator]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for method, calibrator in calibrators.items():
        probability = calibrator.predict(evaluation.logits)
        transformed = calibrator.transform_logits(evaluation.logits) if method != "isotonic" else None
        rows[method] = binary_metrics(evaluation.target, probability, transformed)
    return rows


def train_run(
    arm: str,
    data: PreparedData,
    config: dict,
    run_dir: Path,
    init_seed: int,
    data_order_seed: int,
    channels: str,
    device: torch.device,
    batch_size: int,
    selected_stability: dict[str, float] | None = None,
    max_epochs_override: int | None = None,
    validation_only: bool = False,
) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_status = "VALIDATION_SELECTION_RUN_COMPLETE" if validation_only else "RUN_COMPLETE"
        if existing.get("status") == expected_status:
            return existing
    run_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(init_seed)
    allowed_splits = ("train", "validation") if validation_only else ("train", "validation", "calibration", "test")
    _, loaders = build_loaders(data, channels, batch_size, data_order_seed, allowed_splits)
    input_dim = int(data.metadata["input_dims"][channels])
    train_targets = data.y[data.indices("train")]
    prevalence = float(train_targets.mean())
    model = build_model(arm, config, input_dim, prevalence).to(device)
    run_config = {
        "model_arm": arm,
        "channels": channels,
        "init_seed": init_seed,
        "data_order_seed": data_order_seed,
        "input_dim": input_dim,
        "canonical_model": config["model"],
        "training": config["training"],
        "selected_stability": selected_stability,
        "evaluation_scope": "train_validation_only" if validation_only else "final_all_frozen_splits",
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(environment_record(device), indent=2), encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(git_text(["rev-parse", "HEAD"]) + "\n", encoding="utf-8")
    (run_dir / "split_sha256.txt").write_text(str(data.metadata["split_sha256"]) + "\n", encoding="utf-8")
    (run_dir / "preprocessing_sha256.txt").write_text(str(data.metadata["preprocessing_sha256"]) + "\n", encoding="utf-8")
    resume_path = run_dir / "checkpoint_resume.pt"
    log_path = run_dir / "train.log"
    started = time.perf_counter()
    replica_b = None
    if arm == "ConceptFAN-StabilityReg":
        if selected_stability is None:
            raise ValueError("StabilityReg requires frozen validation-selected coefficients")
        set_all_seeds(init_seed + 1_000_003)
        replica_b = build_model(arm, config, input_dim, prevalence).to(device)
        set_all_seeds(init_seed)
        model, replica_b, history, fit = _train_stability_pair(
            model,
            replica_b,
            loaders,
            config,
            device,
            float(selected_stability["lambda_stab"]),
            float(selected_stability["lambda_logit"]),
            resume_path,
            log_path,
            max_epochs_override=max_epochs_override,
        )
    else:
        model, history, fit = _train_standard(
            arm,
            model,
            loaders,
            config,
            device,
            resume_path,
            log_path,
            max_epochs_override=max_epochs_override,
        )
    validation = evaluate_model(model, loaders["validation"], device)
    train = evaluate_model(model, loaders["train"], device)
    if validation_only:
        identity = Calibrator("none", {})
        evaluations = {"train": train, "validation": validation}
        for split_name, evaluation in evaluations.items():
            _save_split_outputs(run_dir, split_name, evaluation, identity)
        validation_metrics = {"none": binary_metrics(validation.target, validation.probability, validation.logits)}
        (run_dir / "metrics_val.json").write_text(json.dumps(validation_metrics, indent=2), encoding="utf-8")
        primary_name = "none"
        primary = identity
    else:
        calibration = evaluate_model(model, loaders["calibration"], device)
        calibrators = fit_calibrators(calibration.logits, calibration.target)
        primary_name, calibration_losses = select_primary_calibrator(calibrators, calibration.logits, calibration.target)
        primary = calibrators[primary_name]
        test = evaluate_model(model, loaders["test"], device)
        evaluations = {"train": train, "validation": validation, "calibration": calibration, "test": test}
        for split_name, evaluation in evaluations.items():
            _save_split_outputs(run_dir, split_name, evaluation, primary)
        validation_metrics = _metrics_for_calibrators(validation, calibrators)
        calibration_metrics = _metrics_for_calibrators(calibration, calibrators)
        test_metrics = _metrics_for_calibrators(test, calibrators)
        (run_dir / "metrics_val.json").write_text(json.dumps(validation_metrics, indent=2), encoding="utf-8")
        (run_dir / "metrics_calibration.json").write_text(
            json.dumps(
                {
                    "selected_method": primary_name,
                    "selection_scope": "calibration_only",
                    "selection_nll": calibration_losses,
                    "metrics": calibration_metrics,
                    "calibrators": {name: calibrator.to_dict() for name, calibrator in calibrators.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_dir / "metrics_test.json").write_text(
            json.dumps({"primary_method": primary_name, "metrics": test_metrics}, indent=2), encoding="utf-8"
        )
    checkpoint = {
        "status": "BEST_CHECKPOINT",
        "model_arm": arm,
        "channels": channels,
        "input_dim": input_dim,
        "init_seed": init_seed,
        "data_order_seed": data_order_seed,
        "model_state_dict": model.state_dict(),
        "replica_b_state_dict": replica_b.state_dict() if replica_b is not None else None,
        "selected_calibrator": primary.to_dict(),
        "selected_stability": selected_stability,
        "model_config": config["model"],
        "parameter_sha256": parameter_sha256(model),
        "fit": fit,
    }
    checkpoint_path = run_dir / "checkpoint_best.pt"
    torch.save(checkpoint, checkpoint_path)
    max_decomposition_error = float(max(np.max(item.decomposition_error) for item in evaluations.values()))
    if arm != "PlainTransformer" and max_decomposition_error >= 1e-5:
        raise AssertionError(f"Exact additive decomposition failed: {max_decomposition_error}")
    if resume_path.exists():
        resume_path.unlink()
    manifest = {
        "status": "VALIDATION_SELECTION_RUN_COMPLETE" if validation_only else "RUN_COMPLETE",
        "created_utc": utc_now(),
        "model_arm": arm,
        "channels": channels,
        "init_seed": init_seed,
        "data_order_seed": data_order_seed,
        "best_epoch": int(fit["best_epoch"]),
        "epochs_completed": len(history),
        "best_validation_AUPRC": float(fit["best_validation_AUPRC"]),
        "selected_calibrator": primary_name,
        "selected_stability": selected_stability,
        "parameter_sha256": checkpoint["parameter_sha256"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "max_decomposition_error": max_decomposition_error,
        "elapsed_seconds": time.perf_counter() - started,
        "evaluation_scope": "train_validation_only" if validation_only else "final_all_frozen_splits",
        "test_read_after_training_and_calibration_selection": not validation_only,
        "test_used_for_selection": False,
        "config_sha256": sha256_file(run_dir / "config.yaml"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_model_from_checkpoint(path: Path, config: dict, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_arm"], config, int(checkpoint["input_dim"]), 0.5)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def select_stability_coefficients(
    data: PreparedData,
    config: dict,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    selection_path = output_dir / "stability_selection.json"
    if selection_path.exists():
        return json.loads(selection_path.read_text(encoding="utf-8"))["selected"]
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = output_dir / "baseline"
    baseline = train_run(
        "ConceptFAN-NoAlpha",
        data,
        config,
        baseline_dir,
        init_seed=11,
        data_order_seed=1001,
        channels="V+M+D",
        device=device,
        batch_size=batch_size,
        max_epochs_override=10,
        validation_only=True,
    )
    baseline_auprc = float(baseline["best_validation_AUPRC"])
    rows: list[dict] = []
    for lambda_stab in config["stability_regularization"]["lambda_stab_grid"]:
        for lambda_logit in config["stability_regularization"]["lambda_logit_grid"]:
            candidate = {"lambda_stab": float(lambda_stab), "lambda_logit": float(lambda_logit)}
            candidate_dir = output_dir / f"stab_{lambda_stab}_logit_{lambda_logit}"
            manifest = train_run(
                "ConceptFAN-StabilityReg",
                data,
                config,
                candidate_dir,
                init_seed=11,
                data_order_seed=1001,
                channels="V+M+D",
                device=device,
                batch_size=batch_size,
                selected_stability=candidate,
                max_epochs_override=10,
                validation_only=True,
            )
            contrib_a = pd.read_parquet(candidate_dir / "contributions_validation.parquet")
            checkpoint = torch.load(candidate_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)
            model_b = build_model("ConceptFAN-StabilityReg", config, int(checkpoint["input_dim"]), 0.5).to(device)
            model_b.load_state_dict(checkpoint["replica_b_state_dict"])
            _, loaders = build_loaders(data, "V+M+D", batch_size, 1001, ("validation",))
            eval_b = evaluate_model(model_b, loaders["validation"], device)
            columns = [column for column in contrib_a if column.startswith("contribution_") and not column.startswith("contribution_error")]
            a = contrib_a[columns].to_numpy()
            b = eval_b.contributions
            correlations = []
            for left, right in zip(a, b):
                if np.std(left) < 1e-12 or np.std(right) < 1e-12:
                    correlations.append(0.0)
                else:
                    correlations.append(float(pd.Series(left).corr(pd.Series(right), method="spearman")))
            rows.append(
                {
                    **candidate,
                    "validation_AUPRC": float(manifest["best_validation_AUPRC"]),
                    "validation_AUPRC_delta_vs_baseline": float(manifest["best_validation_AUPRC"] - baseline_auprc),
                    "replica_contribution_spearman": float(np.nanmean(correlations)),
                }
            )
    frame = pd.DataFrame(rows)
    margin = float(config["stability_regularization"]["auprc_noninferiority_margin"])
    eligible = frame.loc[frame["validation_AUPRC_delta_vs_baseline"] >= -margin]
    pool = eligible if len(eligible) else frame
    chosen = pool.sort_values(["replica_contribution_spearman", "validation_AUPRC"], ascending=False).iloc[0]
    selected = {"lambda_stab": float(chosen["lambda_stab"]), "lambda_logit": float(chosen["lambda_logit"])}
    frame.to_csv(output_dir / "stability_selection_grid.csv", index=False)
    selection_path.write_text(
        json.dumps(
            {
                "status": "STABILITY_COEFFICIENTS_FROZEN_ON_VALIDATION",
                "baseline_validation_AUPRC": baseline_auprc,
                "noninferiority_margin": margin,
                "selected": selected,
                "test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return selected
