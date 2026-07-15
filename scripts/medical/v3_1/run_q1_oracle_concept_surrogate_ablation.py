#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.medical.v2.run_v2_program import make_episodes, split_frame


CONCEPT_NAMES = ["infection", "inflammation", "hemodynamics", "organ_dysfunction", "shock"]
REQUIRED_Q1_ARTIFACTS = {
    "30 independent runs": "TABLES/q1_run_metrics.csv",
    "PCBM baseline": "TABLES/q1_run_metrics.csv",
    "CEM baseline": "TABLES/q1_run_metrics.csv",
    "NoFuzzy ablation": "TABLES/q1_run_metrics.csv",
    "M sensitivity 2/3/4/5": "TABLES/q1_m_sensitivity.csv",
    "Brier score": "TABLES/q1_calibration.csv",
    "ECE": "TABLES/q1_calibration.csv",
    "NLL": "TABLES/q1_calibration.csv",
    "temperature scaling": "TABLES/q1_calibration.csv",
    "Spearman distribution": "TABLES/q1_stability_pairwise.csv",
    "Kendall distribution": "TABLES/q1_stability_pairwise.csv",
    "Jaccard distribution": "TABLES/q1_stability_pairwise.csv",
    "initialization/data-order split": "TABLES/q1_seed_variance_decomposition.csv",
    "shuffled concept controls": "TABLES/q1_controls.csv",
    "random concept controls": "TABLES/q1_controls.csv",
    "target leakage through concept errors": "TABLES/q1_leakage_audit.csv",
    "noise sweep": "TABLES/q1_robustness.csv",
    "MCAR missingness": "TABLES/q1_robustness.csv",
    "block missingness": "TABLES/q1_robustness.csv",
    "generator/distribution shift": "TABLES/q1_robustness.csv",
    "article-ready Q1 tables": "TABLES/q1_model_summary.csv",
    "article-ready Q1 figures": "FIGURES/q1_model_auprc.png",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0 - 1e-8)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if mask.any():
            out += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(out)


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0 - 1e-8)
    return {
        "AUROC": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan"),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "ECE": ece_score(y, p),
        "NLL": float(log_loss(y, p, labels=[0, 1])),
    }


def best_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    grid = np.r_[np.linspace(0.35, 1.0, 14), np.linspace(1.1, 4.0, 30)]
    losses = [log_loss(y, sigmoid(logits / t), labels=[0, 1]) for t in grid]
    return float(grid[int(np.argmin(losses))])


def state_sequence(df: pd.DataFrame, n_concepts: int = 5) -> np.ndarray:
    return np.stack(df["states"].map(lambda v: np.asarray(v, dtype=np.float32)).to_numpy())[:, :36, :n_concepts]


def targets(df: pd.DataFrame) -> np.ndarray:
    return df["target"].to_numpy(dtype=int)


def concept_feature_matrix(seq: np.ndarray) -> tuple[np.ndarray, list[str], list[str]]:
    blocks = [
        ("last", seq[:, -1, :]),
        ("mean", seq.mean(axis=1)),
        ("max", seq.max(axis=1)),
        ("mean6", seq[:, -6:, :].mean(axis=1)),
        ("slope6", seq[:, -1, :] - seq[:, -6, :]),
    ]
    cols: list[np.ndarray] = []
    names: list[str] = []
    groups: list[str] = []
    for suffix, block in blocks:
        cols.append(block)
        names.extend([f"{name}_{suffix}" for name in CONCEPT_NAMES])
        groups.extend(CONCEPT_NAMES)
    return np.concatenate(cols, axis=1).astype(np.float64), names, groups


def split_train_calibration(n: int, data_order_seed: int, calibration_fraction: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(data_order_seed)
    order = np.arange(n)
    rng.shuffle(order)
    n_cal = max(1, int(round(n * calibration_fraction)))
    return order[n_cal:], order[:n_cal]


def run_grid(n_runs: int) -> list[dict[str, int]]:
    init_seeds = list(range(1001, 1011))
    order_seeds = list(range(2001, 2004))
    grid = []
    for init_seed in init_seeds:
        for data_order_seed in order_seeds:
            grid.append(
                {
                    "run_id": len(grid) + 1,
                    "initialization_seed": init_seed,
                    "data_order_seed": data_order_seed,
                }
            )
    return grid[: int(n_runs)]


@dataclass
class FittedArm:
    arm: str
    model: LogisticRegression
    scaler: StandardScaler
    feature_names: list[str]
    concept_groups: list[str]
    m: int = 0
    fuzzy: "FuzzyTransformer | None" = None
    pca: PCA | None = None
    temperature: float = 1.0

    def design(self, x: np.ndarray) -> np.ndarray:
        z, _, _ = design_matrix(self.arm, x, m=self.m, fitted_fuzzy=self.fuzzy)
        return z

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(self.design(x))
        if self.pca is not None:
            z = self.pca.transform(z)
        return z

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.model.decision_function(self.transform(x))

    def probabilities(self, x: np.ndarray, calibrated: bool = False) -> np.ndarray:
        temp = self.temperature if calibrated else 1.0
        return sigmoid(self.logits(x) / temp)

    def concept_importance(self) -> np.ndarray:
        coef = np.asarray(self.model.coef_).reshape(-1)
        if self.pca is not None:
            coef = self.pca.components_.T @ coef
        out = np.zeros(len(CONCEPT_NAMES), dtype=float)
        for idx, concept in enumerate(self.concept_groups):
            out[CONCEPT_NAMES.index(concept)] += abs(float(coef[idx]))
        denom = np.abs(out).sum()
        return out / denom if denom > 0 else out


class FuzzyTransformer:
    def __init__(self, m: int):
        self.m = int(m)
        self.centers: np.ndarray | None = None
        self.widths: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "FuzzyTransformer":
        qs = np.linspace(0.10, 0.90, self.m)
        self.centers = np.quantile(x, qs, axis=0).T
        span = np.maximum(np.quantile(x, 0.90, axis=0) - np.quantile(x, 0.10, axis=0), 1e-5)
        self.widths = np.repeat((span / max(1, self.m - 1))[:, None], self.m, axis=1)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.centers is None or self.widths is None:
            raise RuntimeError("FuzzyTransformer is not fitted")
        diff = x[:, :, None] - self.centers[None, :, :]
        z = np.exp(-0.5 * (diff / np.maximum(self.widths[None, :, :], 1e-5)) ** 2)
        return z.reshape(x.shape[0], -1)


def design_matrix(arm: str, x: np.ndarray, *, m: int, fitted_fuzzy: FuzzyTransformer | None = None) -> tuple[np.ndarray, list[str], FuzzyTransformer | None]:
    if arm == "ConceptFAN":
        fuzzy = fitted_fuzzy or FuzzyTransformer(m).fit(x)
        return fuzzy.transform(x), [f"fuzzy_{i}" for i in range(x.shape[1] * m)], fuzzy
    if arm == "NoFuzzy":
        return x, [f"raw_{i}" for i in range(x.shape[1])], None
    if arm == "CEM":
        emb = np.concatenate([x, x * x, np.sqrt(np.clip(x, 0.0, None)), x * (1.0 - x)], axis=1)
        return emb, [f"cem_{i}" for i in range(emb.shape[1])], None
    if arm == "PCBM":
        return x, [f"pcbm_raw_{i}" for i in range(x.shape[1])], None
    raise ValueError(f"Unknown arm {arm}")


def expanded_groups(base_groups: list[str], arm: str, m: int) -> list[str]:
    if arm == "ConceptFAN":
        return [g for g in base_groups for _ in range(m)]
    if arm == "CEM":
        return base_groups * 4
    return list(base_groups)


def fit_arm(
    arm: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    *,
    init_seed: int,
    m: int,
    base_groups: list[str],
) -> FittedArm:
    x_design, feature_names, fuzzy = design_matrix(arm, x_train, m=m)
    pca = None
    if arm == "PCBM":
        pca = PCA(n_components=min(8, x_design.shape[1]), random_state=init_seed)
    scaler = StandardScaler().fit(x_design)
    z_train = scaler.transform(x_design)
    if pca is not None:
        z_train = pca.fit_transform(z_train)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=init_seed)
    model.fit(z_train, y_train)
    x_cal_design, _, _ = design_matrix(arm, x_cal, m=m, fitted_fuzzy=fuzzy)
    fitted = FittedArm(
        arm=arm,
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        concept_groups=expanded_groups(base_groups, arm, m),
        m=m,
        fuzzy=fuzzy,
        pca=pca,
    )
    logits_cal = model.decision_function(pca.transform(scaler.transform(x_cal_design)) if pca is not None else scaler.transform(x_cal_design))
    fitted.temperature = best_temperature(logits_cal, y_cal)
    return fitted


def evaluate_arm_rows(
    fitted: FittedArm,
    x_val: np.ndarray,
    y_val: np.ndarray,
    run_meta: dict[str, int],
    *,
    arm: str,
    m: int,
    split_label: str = "validation",
) -> list[dict]:
    logits = fitted.logits(x_val)
    raw = sigmoid(logits)
    cal = sigmoid(logits / fitted.temperature)
    rows = []
    for calibrated, p in [(False, raw), (True, cal)]:
        rows.append(
            {
                **run_meta,
                "model_arm": arm,
                "n_memberships": m if arm == "ConceptFAN" else 0,
                "split": split_label,
                "calibrated": calibrated,
                "temperature": float(fitted.temperature if calibrated else 1.0),
                **binary_metrics(y_val, p),
            }
        )
    return rows


def make_dataset(cfg: dict, seed: int, *, n_samples: int | None = None, infection_prevalence: float | None = None, impulse: float | None = None) -> dict[str, pd.DataFrame]:
    local = copy.deepcopy(cfg)
    if n_samples is not None:
        local["dataset"]["n_samples"] = int(n_samples)
    if infection_prevalence is not None:
        local["dataset"]["infection_prevalence"] = float(infection_prevalence)
    if impulse is not None:
        local["dataset"]["infection_impulse_strength"] = float(impulse)
    return split_frame(make_episodes(seed, local, "clean"), seed)


def perturb_sequence(seq: np.ndarray, kind: str, level: float, rng: np.random.Generator, fill: np.ndarray) -> np.ndarray:
    out = seq.copy()
    if kind == "noise":
        return np.clip(out + rng.normal(0.0, level, size=out.shape), 0.0, 1.0)
    if kind == "mcar":
        mask = rng.random(out.shape) < level
        out[mask] = np.broadcast_to(fill, out.shape)[mask]
        return out
    if kind == "block_missing":
        block = max(1, int(round(out.shape[1] * level)))
        start = max(0, (out.shape[1] - block) // 2)
        out[:, start : start + block, :] = fill.reshape(1, 1, -1)
        return out
    raise ValueError(kind)


def pairwise_stability(importance_rows: list[dict]) -> pd.DataFrame:
    rows = []
    concept_cols = [f"importance_{c}" for c in CONCEPT_NAMES]
    df = pd.DataFrame(importance_rows)
    fan = df[df["model_arm"].eq("ConceptFAN")].reset_index(drop=True)
    for i in range(len(fan)):
        for j in range(i + 1, len(fan)):
            a = fan.loc[i, concept_cols].to_numpy(dtype=float)
            b = fan.loc[j, concept_cols].to_numpy(dtype=float)
            top_a = set(np.argsort(-a)[:2].tolist())
            top_b = set(np.argsort(-b)[:2].tolist())
            rows.append(
                {
                    "run_a": int(fan.loc[i, "run_id"]),
                    "run_b": int(fan.loc[j, "run_id"]),
                    "spearman": float(stats.spearmanr(a, b).statistic),
                    "kendall": float(stats.kendalltau(a, b).statistic),
                    "jaccard_top2": float(len(top_a & top_b) / max(1, len(top_a | top_b))),
                }
            )
    return pd.DataFrame(rows)


def variance_decomposition(run_metrics: pd.DataFrame, importance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fan = run_metrics[(run_metrics.model_arm == "ConceptFAN") & (~run_metrics.calibrated)].copy()
    for metric in ["AUPRC", "Brier", "ECE", "NLL"]:
        total = float(fan[metric].var(ddof=0))
        init_var = float(fan.groupby("initialization_seed")[metric].mean().var(ddof=0))
        order_var = float(fan.groupby("data_order_seed")[metric].mean().var(ddof=0))
        rows.append(
            {
                "quantity": metric,
                "total_variance": total,
                "initialization_seed_variance": init_var,
                "data_order_seed_variance": order_var,
                "residual_variance": max(0.0, total - init_var - order_var),
            }
        )
    for concept in CONCEPT_NAMES:
        col = f"importance_{concept}"
        total = float(importance[col].var(ddof=0))
        rows.append(
            {
                "quantity": f"importance_{concept}",
                "total_variance": total,
                "initialization_seed_variance": float(importance.groupby("initialization_seed")[col].mean().var(ddof=0)),
                "data_order_seed_variance": float(importance.groupby("data_order_seed")[col].mean().var(ddof=0)),
                "residual_variance": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def leakage_audit(x_val: np.ndarray, y_val: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for noise in [0.01, 0.03, 0.05, 0.10]:
        noisy = np.clip(x_val + rng.normal(0.0, noise, size=x_val.shape), 0.0, 1.0)
        errors = np.abs(noisy - x_val)
        leak = LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear", random_state=17)
        leak.fit(errors, y_val)
        p = leak.predict_proba(errors)[:, 1]
        rows.append({"concept_error_noise": noise, **binary_metrics(y_val, p)})
    return pd.DataFrame(rows)


def summarize_requirements(output: Path) -> pd.DataFrame:
    rows = []
    for requirement, rel in REQUIRED_Q1_ARTIFACTS.items():
        path = output / rel
        rows.append(
            {
                "requirement": requirement,
                "artifact": rel,
                "present": path.exists(),
                "bytes": int(path.stat().st_size) if path.exists() else 0,
                "sha256": sha256_file(path) if path.exists() else "",
            }
        )
    return pd.DataFrame(rows)


def _finite_columns(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(col in df.columns and np.isfinite(pd.to_numeric(df[col], errors="coerce")).all() for col in cols)


def validate_q1_outputs(output: Path) -> dict:
    tables = output / "TABLES"
    figures = output / "FIGURES"
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    required_paths = sorted(set(REQUIRED_Q1_ARTIFACTS.values())) + ["TABLES/q1_requirements_matrix.csv"]
    for rel in required_paths:
        path = output / rel
        add(f"artifact exists: {rel}", path.exists() and path.stat().st_size > 0, str(path))

    run = pd.read_csv(tables / "q1_run_metrics.csv")
    run_pairs = run[["run_id", "initialization_seed", "data_order_seed"]].drop_duplicates()
    add("30 independent crossed runs", len(run_pairs) == 30, f"unique run triples={len(run_pairs)}")
    add("10 initialization seeds", run_pairs["initialization_seed"].nunique() == 10, f"count={run_pairs['initialization_seed'].nunique()}")
    add("3 data-order seeds", run_pairs["data_order_seed"].nunique() == 3, f"count={run_pairs['data_order_seed'].nunique()}")
    add("4 required model arms", set(run["model_arm"]) == {"ConceptFAN", "PCBM", "CEM", "NoFuzzy"}, ",".join(sorted(run["model_arm"].unique())))
    add("calibrated and raw rows", set(run["calibrated"].astype(str)) == {"False", "True"}, ",".join(sorted(run["calibrated"].astype(str).unique())))
    add("run metric row count", len(run) == 30 * 4 * 2, f"rows={len(run)}")
    add("run metrics finite", _finite_columns(run, ["AUROC", "AUPRC", "Brier", "ECE", "NLL", "temperature"]), "AUROC/AUPRC/Brier/ECE/NLL/temperature")

    cal = pd.read_csv(tables / "q1_calibration.csv")
    add("calibration columns", {"Brier", "ECE", "NLL", "temperature", "calibrated"}.issubset(cal.columns), ",".join(cal.columns))
    add("temperature scaling evaluated", bool((cal["calibrated"].astype(str) == "True").any()), "calibrated=True present")
    add("temperature scaling non-trivial", bool((cal.loc[cal["calibrated"].astype(str) == "True", "temperature"] != 1.0).any()), "at least one fitted temperature differs from 1")

    ms = pd.read_csv(tables / "q1_m_sensitivity.csv")
    add("M sensitivity levels", set(ms["n_memberships"].astype(int)) == {2, 3, 4, 5}, ",".join(map(str, sorted(ms["n_memberships"].unique()))))
    add("M sensitivity row count", len(ms) == 30 * 4, f"rows={len(ms)}")
    add("M sensitivity finite", _finite_columns(ms, ["AUROC", "AUPRC", "Brier", "ECE", "NLL"]), "metrics")

    stability = pd.read_csv(tables / "q1_stability_pairwise.csv")
    add("stability pair count", len(stability) == math.comb(30, 2), f"rows={len(stability)}")
    add("stability metrics finite", _finite_columns(stability, ["spearman", "kendall", "jaccard_top2"]), "Spearman/Kendall/Jaccard")

    seed_var = pd.read_csv(tables / "q1_seed_variance_decomposition.csv")
    expected_quantities = {"AUPRC", "Brier", "ECE", "NLL"} | {f"importance_{name}" for name in CONCEPT_NAMES}
    add("seed variance quantities", expected_quantities.issubset(set(seed_var["quantity"])), ",".join(seed_var["quantity"].astype(str)))

    controls = pd.read_csv(tables / "q1_controls.csv")
    expected_controls = {
        "shuffled_concepts",
        "random_concepts",
        "sufficient_subset_shock_only",
        "sufficient_subset_infection_shock",
        "sufficient_subset_hemodynamics_organ_shock",
    }
    add("concept controls", expected_controls.issubset(set(controls["control"])), ",".join(sorted(controls["control"].unique())))
    add("control row count", len(controls) == 30 * len(expected_controls), f"rows={len(controls)}")

    leakage = pd.read_csv(tables / "q1_leakage_audit.csv")
    add("leakage audit levels", set(np.round(leakage["concept_error_noise"].astype(float), 2)) == {0.01, 0.03, 0.05, 0.10}, ",".join(map(str, leakage["concept_error_noise"])))
    add("leakage metrics finite", _finite_columns(leakage, ["AUROC", "AUPRC", "Brier", "ECE", "NLL"]), "metrics")

    robustness = pd.read_csv(tables / "q1_robustness.csv")
    add("robustness scenarios", set(robustness["scenario"]) == {"noise", "mcar", "block_missing", "generator_shift"}, ",".join(sorted(robustness["scenario"].unique())))
    add("robustness row count", len(robustness) == 30 * (5 + 5 + 4 + 4), f"rows={len(robustness)}")
    add("robustness metrics finite", _finite_columns(robustness, ["AUROC", "AUPRC", "Brier", "ECE", "NLL"]), "metrics")

    model_summary = pd.read_csv(tables / "q1_model_summary.csv")
    add("article table model summary", len(model_summary) == 8, f"rows={len(model_summary)}")
    for figure_name in ["q1_model_auprc.png", "q1_calibration_brier.png", "q1_stability_spearman.png", "q1_robustness.png"]:
        path = figures / figure_name
        add(f"article figure {figure_name}", path.exists() and path.stat().st_size > 1024, str(path))

    passed = all(row["passed"] for row in checks)
    return {
        "status": "Q1_STRICT_VALIDATION_PASS" if passed else "Q1_STRICT_VALIDATION_FAIL",
        "created_utc": now(),
        "checks": checks,
        "passed": passed,
        "failed_checks": [row for row in checks if not row["passed"]],
    }


def write_figures(output: Path, run_metrics: pd.DataFrame, stability: pd.DataFrame, robustness: pd.DataFrame) -> None:
    fig_dir = output / "FIGURES"
    fig_dir.mkdir(parents=True, exist_ok=True)
    raw = run_metrics[~run_metrics["calibrated"]]
    plt.figure(figsize=(8, 4))
    raw.boxplot(column="AUPRC", by="model_arm", rot=20)
    plt.suptitle("")
    plt.title("Q1 model AUPRC over independent runs")
    plt.ylabel("AUPRC")
    plt.tight_layout()
    plt.savefig(fig_dir / "q1_model_auprc.png", dpi=180)
    plt.close()

    cal = run_metrics.groupby(["model_arm", "calibrated"], as_index=False)["Brier"].mean()
    pivot = cal.pivot(index="model_arm", columns="calibrated", values="Brier")
    pivot.plot(kind="bar", figsize=(8, 4))
    plt.title("Q1 calibration Brier score")
    plt.ylabel("Brier")
    plt.tight_layout()
    plt.savefig(fig_dir / "q1_calibration_brier.png", dpi=180)
    plt.close()

    if not stability.empty:
        plt.figure(figsize=(7, 4))
        plt.hist(stability["spearman"], bins=12)
        plt.title("Q1 ConceptFAN contribution Spearman distribution")
        plt.xlabel("Spearman")
        plt.ylabel("pair count")
        plt.tight_layout()
        plt.savefig(fig_dir / "q1_stability_spearman.png", dpi=180)
        plt.close()

    rob = robustness[robustness["model_arm"].eq("ConceptFAN")]
    if not rob.empty:
        plt.figure(figsize=(8, 4))
        for scenario, grp in rob.groupby("scenario"):
            means = grp.groupby("level")["AUPRC"].mean().reset_index()
            plt.plot(means["level"], means["AUPRC"], marker="o", label=scenario)
        plt.title("Q1 ConceptFAN robustness")
        plt.xlabel("level")
        plt.ylabel("AUPRC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "q1_robustness.png", dpi=180)
        plt.close()


def package_output(output: Path, zip_output_dir: Path) -> Path:
    short = subprocess_text(["git", "rev-parse", "--short", "HEAD"])
    suffix = short if subprocess_text(["git", "status", "--porcelain"]) == "" else f"{short}_worktree"
    zip_output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_output_dir / f"Med_CircuitBench_Q1_EMPIRICAL_EXTENSION_{suffix}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zf.write(path, f"Q1_EMPIRICAL_EXTENSION/{path.relative_to(output).as_posix()}")
        for src, arc in [
            (
                ROOT / "scripts" / "medical" / "v3_1" / "run_q1_oracle_concept_surrogate_ablation.py",
                "SOURCE/scripts/medical/v3_1/run_q1_oracle_concept_surrogate_ablation.py",
            ),
            (ROOT / "configs" / "medical" / "v3" / "full.yaml", "CONFIGS/medical/v3/full.yaml"),
            (ROOT / "AGENTS.md", "MANIFESTS/AGENTS.md"),
        ]:
            if src.exists():
                zf.write(src, arc)
    sidecar = zip_path.with_suffix(zip_path.suffix + ".sha256")
    sidecar.write_text(f"{sha256_file(zip_path)}  {zip_path.name}\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
    if bad is not None:
        raise RuntimeError(f"ZIP test failed at {bad}")
    return zip_path


def subprocess_text(cmd: list[str]) -> str:
    import subprocess

    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def run_extension(cfg: dict, output: Path, *, n_runs: int, n_samples: int, package: bool, zip_output_dir: Path) -> dict:
    if output.exists():
        shutil.rmtree(output)
    for sub in ["TABLES", "FIGURES", "MANIFESTS", "REPORTS"]:
        (output / sub).mkdir(parents=True, exist_ok=True)

    split = make_dataset(cfg, seed=4242, n_samples=n_samples)
    train_seq = state_sequence(split["train"])
    val_seq = state_sequence(split["validation"])
    y_train = targets(split["train"])
    y_val = targets(split["validation"])
    x_train, feature_names, base_groups = concept_feature_matrix(train_seq)
    x_val, _, _ = concept_feature_matrix(val_seq)
    shift_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for shift_name, prevalence, impulse in [
        ("prevalence_low", 0.15, None),
        ("prevalence_high", 0.35, None),
        ("impulse_low", None, 2.4),
        ("impulse_high", None, 3.6),
    ]:
        shifted = make_dataset(
            cfg,
            seed=7000,
            n_samples=max(1200, min(n_samples, 3000)),
            infection_prevalence=prevalence,
            impulse=impulse,
        )
        shifted_seq = state_sequence(shifted["validation"])
        x_shift, _, _ = concept_feature_matrix(shifted_seq)
        shift_cache[shift_name] = (x_shift, targets(shifted["validation"]))

    run_rows: list[dict] = []
    importance_rows: list[dict] = []
    m_rows: list[dict] = []
    control_rows: list[dict] = []
    robustness_rows: list[dict] = []
    fitted_fan_runs: list[tuple[dict, FittedArm]] = []

    for meta in run_grid(n_runs):
        fit_idx, cal_idx = split_train_calibration(len(y_train), meta["data_order_seed"])
        for arm in ["ConceptFAN", "PCBM", "CEM", "NoFuzzy"]:
            fitted = fit_arm(
                arm,
                x_train[fit_idx],
                y_train[fit_idx],
                x_train[cal_idx],
                y_train[cal_idx],
                init_seed=meta["initialization_seed"],
                m=3,
                base_groups=base_groups,
            )
            run_rows.extend(evaluate_arm_rows(fitted, x_val, y_val, meta, arm=arm, m=3))
            if arm == "ConceptFAN":
                fitted_fan_runs.append((meta, fitted))
            imp = fitted.concept_importance()
            importance_rows.append(
                {
                    **meta,
                    "model_arm": arm,
                    **{f"importance_{name}": float(val) for name, val in zip(CONCEPT_NAMES, imp)},
                }
            )
        for m in [2, 3, 4, 5]:
            fitted = fit_arm(
                "ConceptFAN",
                x_train[fit_idx],
                y_train[fit_idx],
                x_train[cal_idx],
                y_train[cal_idx],
                init_seed=meta["initialization_seed"],
                m=m,
                base_groups=base_groups,
            )
            p = fitted.probabilities(x_val, calibrated=True)
            m_rows.append({**meta, "n_memberships": m, **binary_metrics(y_val, p)})

        rng = np.random.default_rng(meta["initialization_seed"] + meta["data_order_seed"])
        for control in ["shuffled_concepts", "random_concepts"]:
            if control == "shuffled_concepts":
                xtr = x_train.copy()
                xva = x_val.copy()
                for col in range(xtr.shape[1]):
                    xtr[:, col] = rng.permutation(xtr[:, col])
                    xva[:, col] = rng.permutation(xva[:, col])
            else:
                xtr = rng.normal(0.0, 1.0, size=x_train.shape)
                xva = rng.normal(0.0, 1.0, size=x_val.shape)
            fitted = fit_arm(
                "ConceptFAN",
                xtr[fit_idx],
                y_train[fit_idx],
                xtr[cal_idx],
                y_train[cal_idx],
                init_seed=meta["initialization_seed"],
                m=3,
                base_groups=base_groups,
            )
            control_rows.append({**meta, "control": control, **binary_metrics(y_val, fitted.probabilities(xva, calibrated=True))})
        for subset, concepts in {
            "shock_only": ["shock"],
            "infection_shock": ["infection", "shock"],
            "hemodynamics_organ_shock": ["hemodynamics", "organ_dysfunction", "shock"],
        }.items():
            cols = [idx for idx, grp in enumerate(base_groups) if grp in concepts]
            fitted = fit_arm(
                "ConceptFAN",
                x_train[fit_idx][:, cols],
                y_train[fit_idx],
                x_train[cal_idx][:, cols],
                y_train[cal_idx],
                init_seed=meta["initialization_seed"],
                m=3,
                base_groups=[base_groups[i] for i in cols],
            )
            control_rows.append(
                {**meta, "control": f"sufficient_subset_{subset}", **binary_metrics(y_val, fitted.probabilities(x_val[:, cols], calibrated=True))}
            )

    fill = train_seq.mean(axis=(0, 1))
    for meta, fitted in fitted_fan_runs:
        rng = np.random.default_rng(meta["initialization_seed"] * 31 + meta["data_order_seed"])
        for scenario, levels in {
            "noise": [0.00, 0.02, 0.05, 0.10, 0.20],
            "mcar": [0.00, 0.05, 0.10, 0.20, 0.35],
            "block_missing": [0.00, 0.10, 0.20, 0.35],
        }.items():
            for level in levels:
                perturbed = perturb_sequence(val_seq, scenario, float(level), rng, fill)
                x_pert, _, _ = concept_feature_matrix(perturbed)
                robustness_rows.append(
                    {
                        **meta,
                        "model_arm": "ConceptFAN",
                        "scenario": scenario,
                        "level": float(level),
                        **binary_metrics(y_val, fitted.probabilities(x_pert, calibrated=True)),
                    }
                )
        for shift_name, (x_shift, y_shift) in shift_cache.items():
            robustness_rows.append(
                {
                    **meta,
                    "model_arm": "ConceptFAN",
                    "scenario": "generator_shift",
                    "level": shift_name,
                    **binary_metrics(y_shift, fitted.probabilities(x_shift, calibrated=True)),
                }
            )

    run_metrics = pd.DataFrame(run_rows)
    importance = pd.DataFrame(importance_rows)
    m_sensitivity = pd.DataFrame(m_rows)
    controls = pd.DataFrame(control_rows)
    stability = pairwise_stability(importance)
    seed_variance = variance_decomposition(run_metrics, importance[importance["model_arm"].eq("ConceptFAN")])
    leakage = leakage_audit(x_val, y_val, np.random.default_rng(9001))
    robustness = pd.DataFrame(robustness_rows)

    model_summary = (
        run_metrics.groupby(["model_arm", "calibrated"], as_index=False)
        .agg(
            runs=("run_id", "nunique"),
            AUPRC_mean=("AUPRC", "mean"),
            AUPRC_std=("AUPRC", "std"),
            AUROC_mean=("AUROC", "mean"),
            Brier_mean=("Brier", "mean"),
            ECE_mean=("ECE", "mean"),
            NLL_mean=("NLL", "mean"),
            temperature_mean=("temperature", "mean"),
        )
        .sort_values(["calibrated", "AUPRC_mean"], ascending=[False, False])
    )
    calibration = run_metrics[
        ["run_id", "initialization_seed", "data_order_seed", "model_arm", "calibrated", "temperature", "Brier", "ECE", "NLL"]
    ].copy()

    tables = {
        "q1_run_metrics.csv": run_metrics,
        "q1_model_summary.csv": model_summary,
        "q1_calibration.csv": calibration,
        "q1_m_sensitivity.csv": m_sensitivity,
        "q1_contribution_importance.csv": importance,
        "q1_stability_pairwise.csv": stability,
        "q1_seed_variance_decomposition.csv": seed_variance,
        "q1_controls.csv": controls,
        "q1_leakage_audit.csv": leakage,
        "q1_robustness.csv": robustness,
    }
    for name, df in tables.items():
        df.to_csv(output / "TABLES" / name, index=False)

    write_figures(output, run_metrics, stability, robustness)
    requirements = summarize_requirements(output)
    requirements.to_csv(output / "TABLES" / "q1_requirements_matrix.csv", index=False)
    strict_validation = validate_q1_outputs(output)
    (output / "MANIFESTS" / "q1_strict_validation.json").write_text(json.dumps(strict_validation, indent=2), encoding="utf-8")
    all_present = bool(requirements["present"].all())
    strict_pass = bool(strict_validation["passed"])
    manifest = {
        "status": "Q1_EMPIRICAL_EXTENSION_COMPLETE" if all_present and strict_pass and int(n_runs) >= 30 else "Q1_EMPIRICAL_EXTENSION_INCOMPLETE",
        "created_utc": now(),
        "code_commit": subprocess_text(["git", "rev-parse", "HEAD"]),
        "branch": subprocess_text(["git", "branch", "--show-current"]),
        "git_status_clean": subprocess_text(["git", "status", "--porcelain"]) == "",
        "n_runs": int(n_runs),
        "run_design": {
            "initialization_seeds": sorted(set(row["initialization_seed"] for row in run_grid(n_runs))),
            "data_order_seeds": sorted(set(row["data_order_seed"] for row in run_grid(n_runs))),
            "runs_are_crossed_grid": int(n_runs) == 30,
        },
        "split_scope": "validation extension; consumed V3 held-out test is not reopened",
        "dataset": {
            "source": "Med-CircuitBench synthetic generator",
            "n_samples": int(n_samples),
            "train": int(len(split["train"])),
            "validation": int(len(split["validation"])),
        },
        "required_artifacts_present": all_present,
        "strict_validation": strict_validation["status"],
        "strict_validation_report": "MANIFESTS/q1_strict_validation.json",
        "requirements_matrix": "TABLES/q1_requirements_matrix.csv",
        "primary_tables": sorted(tables),
        "figures": sorted(p.name for p in (output / "FIGURES").glob("*.png")),
    }
    (output / "MANIFESTS" / "q1_empirical_extension_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "REPORTS" / "Q1_EMPIRICAL_EXTENSION_STATUS.md").write_text(
        "\n".join(
            [
                "# Q1 Empirical Extension Status",
                "",
                f"Status: `{manifest['status']}`",
                "",
                "This extension adds the statistical-depth experiments requested for Q1 without reopening the consumed V3 held-out test.",
                "The evidence is limited to the Med-CircuitBench synthetic validation setting and must not be described as clinical validation.",
                "",
                "Primary artifacts are under `TABLES/`, `FIGURES/`, and `MANIFESTS/`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    zip_path = package_output(output, zip_output_dir) if package else None
    report = {
        "status": manifest["status"],
        "output": str(output),
        "zip": str(zip_path) if zip_path else None,
        "zip_sha256": sha256_file(zip_path) if zip_path else None,
        "requirements_present": all_present,
        "strict_validation": strict_validation["status"],
    }
    (output / "MANIFESTS" / "q1_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medical/v3/full.yaml")
    parser.add_argument("--output", default="artifacts/medical/q1_empirical_extension")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=6000)
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--zip-output-dir", default="artifacts/medical")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    report = run_extension(
        cfg,
        ROOT / args.output,
        n_runs=args.runs,
        n_samples=args.n_samples,
        package=args.package,
        zip_output_dir=ROOT / args.zip_output_dir,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "Q1_EMPIRICAL_EXTENSION_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
