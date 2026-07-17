#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
SEEDS = [42, 43, 44]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def latest_final_practice() -> Path:
    candidates = sorted((ROOT / "artifacts" / "medical").glob("Med_CircuitBench_SCTC_FINAL_PRACTICE_2026-07-14_*"))
    dirs = [p for p in candidates if p.is_dir()]
    if not dirs:
        raise FileNotFoundError("No Med_CircuitBench_SCTC_FINAL_PRACTICE extracted directory found")
    return dirs[-1]


def validate_config(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checks = {
        "latent_dim_128": int(cfg["model"]["latent_dim"]) == 128,
        "layers_4": int(cfg["model"]["layers"]) == 4 and int(cfg["model"]["transformer_layers"]) == 4,
        "heads_4": int(cfg["model"]["heads"]) == 4 and int(cfg["model"]["transformer_heads"]) == 4,
        "ffn_512": int(cfg["model"]["d_ffn"]) == 512 and int(cfg["model"]["transformer_ffn"]) == 512,
        "max_epochs_50": int(cfg["training"]["max_epochs"]) == 50,
        "seeds_42_43_44": list(map(int, cfg["seeds"])) == SEEDS,
    }
    return {"path": str(config_path), "checks": checks, "passed": all(checks.values())}


def load_transformer_metrics(final_practice: Path) -> pd.DataFrame:
    path = final_practice / "model_metrics_by_seed.csv"
    if not path.exists():
        path = final_practice / "RESULTS" / "aggregate_metrics.csv"
    df = pd.read_csv(path)
    if "validation_auprc" not in df.columns:
        raise ValueError(f"Transformer metrics missing validation_auprc: {path}")
    out = df[df["seed"].astype(int).isin(SEEDS)].copy()
    out = out[["seed", "validation_auprc", "validation_auroc", "best_epoch"]]
    out = out.rename(columns={"validation_auprc": "AUPRC", "validation_auroc": "AUROC"})
    out["model_arm"] = "PlainTransformer"
    out["source"] = str(path.relative_to(ROOT))
    return out


def load_fan_metrics() -> pd.DataFrame:
    path = ROOT / "artifacts" / "medical" / "v3_real_final" / "results" / "fan_validation_metrics.csv"
    df = pd.read_csv(path)
    out = df[df["seed"].astype(int).isin(SEEDS)].copy()
    out = out[["seed", "AUPRC", "AUROC"]]
    out["best_epoch"] = pd.NA
    out["model_arm"] = "ConceptFAN-NoAlpha"
    out["source"] = str(path.relative_to(ROOT))
    return out


def checkpoint_evidence(final_practice: Path) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        transformer = final_practice / "RUNS" / f"seed_{seed}_validation" / "med_circuitbench" / "transformer" / "model.ckpt"
        fan = ROOT / "artifacts" / "medical" / "v3_real" / "checkpoints" / "fan" / f"predicted_noalpha_seed_{seed}.pt"
        for arm, path in [("PlainTransformer", transformer), ("ConceptFAN-NoAlpha", fan)]:
            rows.append(
                {
                    "seed": seed,
                    "model_arm": arm,
                    "checkpoint": str(path.relative_to(ROOT)) if path.exists() else str(path),
                    "exists": path.exists(),
                    "bytes": path.stat().st_size if path.exists() else 0,
                    "sha256": sha256_file(path) if path.exists() else "",
                }
            )
    return pd.DataFrame(rows)


def run_parity(output: Path, *, min_auprc: float, package: bool, zip_output_dir: Path) -> dict:
    if output.exists():
        shutil.rmtree(output)
    for sub in ["TABLES", "MANIFESTS", "REPORTS"]:
        (output / sub).mkdir(parents=True, exist_ok=True)

    final_practice = latest_final_practice()
    cfg_report = validate_config(ROOT / "configs" / "medical" / "v3" / "full.yaml")
    metrics = pd.concat([load_fan_metrics(), load_transformer_metrics(final_practice)], ignore_index=True)
    metrics = metrics.sort_values(["model_arm", "seed"]).reset_index(drop=True)
    metrics.to_csv(output / "TABLES" / "canonical_3_seed_parity_metrics.csv", index=False)
    ckpts = checkpoint_evidence(final_practice)
    ckpts.to_csv(output / "TABLES" / "canonical_3_seed_checkpoint_evidence.csv", index=False)

    grouped = metrics.groupby("model_arm").agg(
        seeds=("seed", lambda s: sorted(map(int, s))),
        mean_AUPRC=("AUPRC", "mean"),
        min_AUPRC=("AUPRC", "min"),
        mean_AUROC=("AUROC", "mean"),
    )
    checks = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add("canonical config", bool(cfg_report["passed"]), json.dumps(cfg_report["checks"], sort_keys=True))
    add("two arms present", set(metrics["model_arm"]) == {"ConceptFAN-NoAlpha", "PlainTransformer"}, ",".join(sorted(metrics["model_arm"].unique())))
    for arm in ["ConceptFAN-NoAlpha", "PlainTransformer"]:
        part = metrics[metrics["model_arm"].eq(arm)]
        add(f"{arm} has seeds 42/43/44", sorted(part["seed"].astype(int).tolist()) == SEEDS, str(sorted(part["seed"].astype(int).tolist())))
        add(f"{arm} min AUPRC >= {min_auprc}", float(part["AUPRC"].min()) >= min_auprc, f"min={float(part['AUPRC'].min()):.6f}")
        add(f"{arm} mean AUPRC near 0.82", 0.79 <= float(part["AUPRC"].mean()) <= 0.86, f"mean={float(part['AUPRC'].mean()):.6f}")
    add("all checkpoints present", bool(ckpts["exists"].all()), ckpts[["model_arm", "seed", "exists"]].to_json(orient="records"))
    add("checkpoint hashes unique", ckpts["sha256"].replace("", pd.NA).nunique() == len(ckpts), f"unique={ckpts['sha256'].replace('', pd.NA).nunique()} total={len(ckpts)}")

    passed = all(row["passed"] for row in checks)
    manifest = {
        "status": "CANONICAL_3_SEED_PARITY_PASS" if passed else "CANONICAL_3_SEED_PARITY_FAIL",
        "created_utc": now(),
        "code_commit": git_text(["rev-parse", "HEAD"]),
        "branch": git_text(["branch", "--show-current"]),
        "git_status_clean": git_text(["status", "--porcelain"]) == "",
        "minimum_seed_auprc": min_auprc,
        "config": cfg_report,
        "checks": checks,
        "failed_checks": [row for row in checks if not row["passed"]],
        "summary": grouped.reset_index().to_dict(orient="records"),
        "next_step_allowed": "Q1_NEURAL_GRID_ALLOWED" if passed else "Q1_NEURAL_GRID_FORBIDDEN",
    }
    (output / "MANIFESTS" / "canonical_3_seed_parity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "REPORTS" / "CANONICAL_3_SEED_PARITY_STATUS.md").write_text(
        f"# Canonical 3-Seed Parity\n\nStatus: `{manifest['status']}`\n\nNext: `{manifest['next_step_allowed']}`\n",
        encoding="utf-8",
    )

    zip_path = None
    if package:
        zip_output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = zip_output_dir / f"Med_CircuitBench_CANONICAL_3_SEED_PARITY_{git_text(['rev-parse', '--short', 'HEAD'])}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    zf.write(path, f"CANONICAL_3_SEED_PARITY/{path.relative_to(output).as_posix()}")
            zf.write(ROOT / "scripts" / "medical" / "v3_1" / "verify_canonical_3_seed_parity.py", "SOURCE/scripts/medical/v3_1/verify_canonical_3_seed_parity.py")
            zf.write(ROOT / "configs" / "medical" / "v3" / "full.yaml", "CONFIGS/medical/v3/full.yaml")
        zip_path.with_suffix(zip_path.suffix + ".sha256").write_text(f"{sha256_file(zip_path)}  {zip_path.name}\n", encoding="utf-8")
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP integrity failed at {bad}")

    report = {
        "status": manifest["status"],
        "output": str(output),
        "zip": str(zip_path) if zip_path else None,
        "zip_sha256": sha256_file(zip_path) if zip_path else None,
    }
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/medical/canonical_3_seed_parity")
    parser.add_argument("--min-auprc", type=float, default=0.79)
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--zip-output-dir", default="artifacts/medical")
    args = parser.parse_args(argv)
    report = run_parity(ROOT / args.output, min_auprc=args.min_auprc, package=args.package, zip_output_dir=ROOT / args.zip_output_dir)
    return 0 if report["status"] == "CANONICAL_3_SEED_PARITY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
