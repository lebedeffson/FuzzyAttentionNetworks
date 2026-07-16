#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.medical.v3_1.run_q1_neural_final import (
    ARMS,
    DEVICE,
    build_arrays,
    checkpoint_paths,
    config_contract,
    eval_model,
    load_checkpoint,
    sha256_file,
    state_dict_sha256,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_release(path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if path.is_dir():
        return path, None
    tmp = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"ZIP integrity failed at {bad}")
        zf.extractall(tmp.name)
    candidates = list(Path(tmp.name).glob("Q1_NEURAL_FINAL"))
    if not candidates:
        raise FileNotFoundError("ZIP does not contain Q1_NEURAL_FINAL root")
    return candidates[0], tmp


def check_release(release: Path, cfg: dict, output_json: Path | None = None) -> dict:
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add("release path has no _worktree", "_worktree" not in str(release), str(release))
    manifest_path = release / "MANIFESTS" / "q1_neural_final_manifest.json"
    add("final manifest exists", manifest_path.exists(), str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    add("final status is Q1_EMPIRICAL_EXTENSION_NEURAL_VALIDATED", manifest.get("status") == "Q1_EMPIRICAL_EXTENSION_NEURAL_VALIDATED", str(manifest.get("status")))
    add("canonical config contract", all(config_contract(cfg).values()), json.dumps(config_contract(cfg), sort_keys=True))
    paths = checkpoint_paths(release)
    add("150 model checkpoints present", len(paths) == 150, f"count={len(paths)}")
    arm_counts = {arm: len(list((release / "CHECKPOINTS" / arm).glob("run_*/checkpoint.pt"))) for arm in ARMS}
    add("30 checkpoints per arm", all(v == 30 for v in arm_counts.values()), json.dumps(arm_counts, sort_keys=True))
    shared = list((release / "CHECKPOINTS" / "shared_concept_path").glob("run_*/shared_concept_path.pt"))
    add("30 shared concept path checkpoints", len(shared) == 30, f"count={len(shared)}")

    shas = []
    sample_arrays = build_arrays(cfg, 6262)
    for ckpt_path in paths:
        try:
            model, ckpt = load_checkpoint(ckpt_path)
            sha = state_dict_sha256(model.state_dict())
            shas.append(sha)
            add(f"parameter sha matches {ckpt_path.relative_to(release)}", sha == ckpt.get("parameter_sha256"), sha)
            xb = torch.from_numpy(sample_arrays.x_val[:8]).to(DEVICE)
            with torch.no_grad():
                out = model(xb)
            prob = out["probability"] if isinstance(out, dict) else out.probability
            add(f"forward [B,36,27] {ckpt_path.relative_to(release)}", tuple(xb.shape) == (8, 36, 27) and tuple(prob.shape) == (8,), str(tuple(xb.shape)))
        except Exception as exc:
            add(f"checkpoint loads and forwards {ckpt_path.relative_to(release)}", False, repr(exc))
    add("150 unique model parameter hashes", len(set(shas)) == 150, f"unique={len(set(shas))}")

    metrics_path = release / "TABLES" / "q1_neural_predictive_calibration_metrics.csv"
    fit_path = release / "TABLES" / "q1_neural_checkpoint_fit_metrics.csv"
    local_path = release / "TABLES" / "q1_neural_episode_local_contributions.parquet"
    stability_path = release / "TABLES" / "q1_neural_episode_pairwise_stability.parquet"
    suff_path = release / "TABLES" / "q1_neural_exhaustive_32_mask_sufficiency.parquet"
    leak_path = release / "TABLES" / "q1_neural_heldout_leakage_audit.csv"
    rob_path = release / "TABLES" / "q1_neural_raw_input_robustness.csv"
    ms_path = release / "TABLES" / "q1_neural_membership_sensitivity.csv"
    sanity_path = release / "TABLES" / "q1_neural_model_arm_sanity_gate.csv"
    for path in [metrics_path, fit_path, local_path, stability_path, suff_path, leak_path, rob_path, ms_path]:
        add(f"artifact exists {path.name}", path.exists() and path.stat().st_size > 0, str(path))

    if fit_path.exists():
        fit = pd.read_csv(fit_path)
        add("fit metrics contain five arms", set(fit["model_arm"]) == set(ARMS), ",".join(sorted(fit["model_arm"].unique())))
        add("fit metrics contain 30 runs per arm", fit.groupby("model_arm")["run_id"].nunique().eq(30).all(), str(fit.groupby("model_arm")["run_id"].nunique().to_dict()))
        add("grid ConceptFAN mean AUPRC near 0.82", 0.79 <= float(fit[fit["model_arm"].eq("ConceptFAN-NoAlpha")]["calibrated_AUPRC"].mean()) <= 0.86, str(float(fit[fit["model_arm"].eq("ConceptFAN-NoAlpha")]["calibrated_AUPRC"].mean())))
        add("grid PlainTransformer mean AUPRC near 0.82", 0.79 <= float(fit[fit["model_arm"].eq("PlainTransformer")]["calibrated_AUPRC"].mean()) <= 0.86, str(float(fit[fit["model_arm"].eq("PlainTransformer")]["calibrated_AUPRC"].mean())))

    if sanity_path.exists():
        sanity = pd.read_csv(sanity_path)
        add("sanity gate contains five arms", set(sanity["model_arm"]) == set(ARMS), ",".join(sorted(sanity["model_arm"].unique())))
        fan = sanity[sanity["model_arm"].eq("ConceptFAN-NoAlpha")]
        plain = sanity[sanity["model_arm"].eq("PlainTransformer")]
        add("sanity ConceptFAN min AUPRC >= 0.79", float(fan["calibrated_AUPRC"].min()) >= 0.79, str(float(fan["calibrated_AUPRC"].min()) if not fan.empty else "empty"))
        add("sanity ConceptFAN mean AUPRC near 0.82", 0.79 <= float(fan["calibrated_AUPRC"].mean()) <= 0.86, str(float(fan["calibrated_AUPRC"].mean()) if not fan.empty else "empty"))
        add("sanity PlainTransformer mean AUPRC near 0.82", 0.79 <= float(plain["calibrated_AUPRC"].mean()) <= 0.86, str(float(plain["calibrated_AUPRC"].mean()) if not plain.empty else "empty"))

    if stability_path.exists():
        stability = pd.read_parquet(stability_path)
        add("435 real ConceptFAN checkpoint pairs", stability[["checkpoint_a", "checkpoint_b"]].drop_duplicates().shape[0] == math.comb(30, 2), str(stability[["checkpoint_a", "checkpoint_b"]].drop_duplicates().shape[0]))
        add("stability is episode-level", "episode_pos" in stability.columns and stability["episode_pos"].nunique() > 1, str(stability.columns.tolist()))
        add("stability metrics present", {"spearman", "kendall", "top1_agreement", "top3_jaccard", "sign_agreement"}.issubset(stability.columns), str(stability.columns.tolist()))

    if suff_path.exists():
        suff = pd.read_parquet(suff_path)
        add("exhaustive 32-mask interventions present", set(range(32)).issubset(set(suff["subset_mask_int"].astype(int).unique())), str(sorted(suff["subset_mask_int"].astype(int).unique())[:40]))
        add("M sensitivity in sufficiency 1..5", set(range(1, 6)).issubset(set(suff["M"].astype(int).unique())), str(sorted(suff["M"].astype(int).unique())))
        add("sufficiency controls present", {"random_same_size", "shuffled_ranking", "magnitude_matched"}.issubset(set(suff["control"])), str(sorted(suff["control"].unique())))

    if leak_path.exists():
        leak = pd.read_csv(leak_path)
        add("held-out leakage feature sets present", {"true_concepts", "residuals", "true_concepts_plus_residuals", "permuted_residuals", "matched_random_noise"}.issubset(set(leak["feature_set"])), str(sorted(leak["feature_set"].unique())))

    if rob_path.exists():
        rob = pd.read_csv(rob_path)
        add("raw-input robustness scenarios present", {"gaussian_noise", "mcar", "block_missing", "outliers_spikes", "sensor_bias", "generator_parameter_shift"}.issubset(set(rob["scenario"])), str(sorted(rob["scenario"].unique())))

    if ms_path.exists():
        ms = pd.read_csv(ms_path)
        add("membership sensitivity 2/3/4/5", {2, 3, 4, 5}.issubset(set(ms["n_memberships"].astype(int))), str(sorted(ms["n_memberships"].astype(int).unique())))

    source_text = (ROOT / "scripts" / "medical" / "v3_1" / "run_q1_neural_final.py").read_text(encoding="utf-8")
    for token in ["Q1_NEURAL_FAILED_PILOT", "latent_dim\": 32", "epochs 2"]:
        add(f"forbidden final-source token absent: {token}", token not in source_text, token)

    passed = all(row["passed"] for row in checks)
    report = {
        "status": "Q1_NEURAL_READONLY_VALIDATION_PASS" if passed else "Q1_NEURAL_READONLY_VALIDATION_FAIL",
        "created_utc": now(),
        "release": str(release),
        "passed": passed,
        "checks": checks,
        "failed_checks": [row for row in checks if not row["passed"]],
    }
    if output_json is None:
        output_json = release / "MANIFESTS" / "q1_neural_readonly_validation.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True)
    parser.add_argument("--config", default="configs/medical/v3/full.yaml")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    release, tmp = resolve_release(Path(args.release))
    try:
        cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
        report = check_release(release, cfg, Path(args.output_json) if args.output_json else None)
        print(json.dumps({"status": report["status"], "failed_checks": report["failed_checks"][:10]}, indent=2))
        return 0 if report["passed"] else 2
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
