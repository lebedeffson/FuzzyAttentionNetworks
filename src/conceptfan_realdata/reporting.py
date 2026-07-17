from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .data import PreparedData, sha256_file


COLORS = {
    "ConceptFAN-NoAlpha": "#006D77",
    "PureNoFuzzy": "#E29578",
    "PlainTransformer": "#3D405B",
    "ConceptFAN-StabilityReg": "#7A5195",
    "TemporalCEM": "#D4A017",
}


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    frame.to_parquet(path.with_suffix(".parquet"), index=False, compression="zstd")


def _save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ["svg", "pdf", "png"]:
        output_path = stem.with_suffix(f".{suffix}")
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        if suffix == "svg":
            lines = output_path.read_text(encoding="utf-8").splitlines()
            output_path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(figure)


def write_report_hash_index(report_dir: Path) -> None:
    hash_rows = []
    for path in sorted(report_dir.rglob("*")):
        if path.is_file() and path.name not in {
            "artifacts_sha256.txt",
            "archive_validation.json",
            "final_audit_lite.json",
        }:
            hash_rows.append(f"{sha256_file(path)}  {path.relative_to(report_dir)}")
    manifest_dir = report_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "artifacts_sha256.txt").write_text("\n".join(hash_rows) + "\n", encoding="utf-8")


def _metric_value(summary: pd.DataFrame, arm: str, state: str, metric: str, column: str = "mean") -> float:
    row = summary.loc[
        summary["model_arm"].eq(arm) & summary["reporting_state"].eq(state) & summary["metric"].eq(metric)
    ]
    return float(row.iloc[0][column]) if len(row) else float("nan")


def _primary_calibration_table(predictive: pd.DataFrame) -> pd.DataFrame:
    raw = predictive.loc[predictive["calibration_method"].eq("none")].copy()
    primary = predictive.loc[predictive["is_primary"]].copy()
    raw["state"] = "raw"
    primary["state"] = "primary_calibrated"
    frame = pd.concat([raw, primary], ignore_index=True).drop_duplicates(["model_arm", "run_id", "state"])
    return frame.groupby(["model_arm", "state"], as_index=False).agg(
        runs=("run_id", "nunique"),
        Brier_mean=("Brier", "mean"),
        Brier_std=("Brier", "std"),
        NLL_mean=("NLL", "mean"),
        NLL_std=("NLL", "std"),
        ECE15_mean=("ECE15", "mean"),
        ECE15_std=("ECE15", "std"),
        calibration_slope_mean=("calibration_slope", "mean"),
        calibration_intercept_mean=("calibration_intercept", "mean"),
    )


def _figure_pipeline(data: PreparedData, report_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    labels = [
        ("PhysioNet 2012 Set A\n4,000 ICU stays", "#DCEAF0"),
        ("Patient split\n2400 / 600 /\n400 / 600", "#E8E8E8"),
        ("48-hour V / M / D\ntrain-only\npreprocessing", "#F5E6CC"),
        ("5 proxy concepts\nsoft trajectories", "#DDEAD1"),
        ("5 model arms\n30 runs per arm", "#E8DDF0"),
        ("Calibration, stability,\nrobustness and\nsufficiency", "#F4D6D0"),
    ]
    for index, (label, color) in enumerate(labels):
        x = 0.01 + index * 0.165
        ax.add_patch(plt.Rectangle((x, 0.34), 0.145, 0.32, facecolor=color, edgecolor="#333333", linewidth=1))
        ax.text(x + 0.0725, 0.5, label, ha="center", va="center", fontsize=8.5)
        if index < len(labels) - 1:
            ax.annotate("", xy=(x + 0.163, 0.5), xytext=(x + 0.147, 0.5), arrowprops={"arrowstyle": "->"})
    ax.text(0.02, 0.14, f"Temporal variables: {len(data.variables)} | Proxy concept observability: {data.concept_mask.mean():.1%}", fontsize=10)
    _save_figure(fig, report_dir / "figures" / "fig_real_pipeline")


def _reliability_curve(
    target: np.ndarray,
    probability: np.ndarray,
    seed: int,
    bins: int = 10,
    bootstrap_repetitions: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if target.ndim != 2 or probability.shape != target.shape:
        raise ValueError("Reliability inputs must have shape [runs, patients]")
    edges = np.linspace(0.0, 1.0, bins + 1)
    target_flat = target.ravel()
    probability_flat = probability.ravel()
    positions = np.clip(np.digitize(probability_flat, edges[1:-1]), 0, bins - 1)
    predicted: list[float] = []
    observed: list[float] = []
    rng = np.random.default_rng(seed)
    bootstrap_observed = np.full((bootstrap_repetitions, bins), np.nan, dtype=np.float64)
    for repetition in range(bootstrap_repetitions):
        sampled_patients = rng.integers(0, target.shape[1], size=target.shape[1])
        sampled_target = target[:, sampled_patients].ravel()
        sampled_probability = probability[:, sampled_patients].ravel()
        sampled_positions = np.clip(np.digitize(sampled_probability, edges[1:-1]), 0, bins - 1)
        counts = np.bincount(sampled_positions, minlength=bins)
        sums = np.bincount(sampled_positions, weights=sampled_target, minlength=bins)
        valid = counts > 0
        bootstrap_observed[repetition, valid] = sums[valid] / counts[valid]
    active_bins: list[int] = []
    for index in range(bins):
        selected = positions == index
        if selected.any():
            active_bins.append(index)
            predicted.append(float(probability_flat[selected].mean()))
            observed.append(float(target_flat[selected].mean()))
    intervals = np.nanquantile(bootstrap_observed[:, active_bins], [0.025, 0.975], axis=0)
    return np.asarray(predicted), np.asarray(observed), intervals[0], intervals[1]


def _figure_performance(predictive: pd.DataFrame, artifacts_root: Path, report_dir: Path) -> None:
    raw = predictive.loc[predictive["calibration_method"].eq("none")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    arms = list(COLORS)
    x = np.arange(len(arms))
    for position, metric in enumerate(["AUPRC", "AUROC"]):
        means = [raw.loc[raw["model_arm"].eq(arm), metric].mean() for arm in arms]
        stds = [raw.loc[raw["model_arm"].eq(arm), metric].std() for arm in arms]
        axes[position].bar(x, means, yerr=stds, color=[COLORS[arm] for arm in arms], capsize=3)
        axes[position].set_xticks(x, [arm.replace("ConceptFAN-", "CF-") for arm in arms], rotation=25, ha="right")
        axes[position].set_ylabel(metric)
        axes[position].grid(axis="y", alpha=0.25)
    run_paths = sorted((artifacts_root / "runs" / "ConceptFAN-NoAlpha").glob("run_*/logits_test.parquet"))
    if len(run_paths) != 30:
        raise ValueError(f"Expected 30 ConceptFAN test prediction files, found {len(run_paths)}")
    logits = [
        pd.read_parquet(
            path,
            columns=["RecordID", "target", "probability_raw", "probability_primary_calibrated"],
        ).sort_values("RecordID")
        for path in run_paths
    ]
    record_ids = logits[0]["RecordID"].to_numpy()
    if any(not np.array_equal(frame["RecordID"].to_numpy(), record_ids) for frame in logits[1:]):
        raise ValueError("ConceptFAN runs do not share the same held-out patient order")
    target = np.stack([frame["target"].to_numpy(dtype=np.float64) for frame in logits])
    for seed, column, label, color in [
        (20260717, "probability_raw", "Raw", COLORS["ConceptFAN-NoAlpha"]),
        (20260718, "probability_primary_calibrated", "Primary calibrated", COLORS["PureNoFuzzy"]),
    ]:
        predicted, observed, low, high = _reliability_curve(
            target,
            np.stack([frame[column].to_numpy(dtype=np.float64) for frame in logits]),
            seed=seed,
        )
        axes[2].plot(predicted, observed, marker="o", linewidth=1.5, label=label, color=color)
        axes[2].fill_between(predicted, low, high, alpha=0.15, color=color)
    axes[2].plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#555555", label="Ideal")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Observed event rate")
    axes[2].set_title("ConceptFAN reliability (patient bootstrap CI)")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(alpha=0.25)
    fig.suptitle("Held-out predictive performance and calibration across 30 runs")
    _save_figure(fig, report_dir / "figures" / "fig_calibration")


def _figure_stability(episode_path: Path, report_dir: Path) -> None:
    frame = pd.read_parquet(episode_path, columns=["model_arm", "representation", "spearman"])
    frame = frame.loc[frame["representation"].eq("signed")]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins = np.linspace(-1, 1, 60)
    for arm in ["ConceptFAN-NoAlpha", "PureNoFuzzy", "ConceptFAN-StabilityReg"]:
        values = frame.loc[frame["model_arm"].eq(arm), "spearman"].to_numpy()
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.6, label=arm, color=COLORS[arm])
    ax.set_xlabel("Episode-level Spearman across retrainings")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    _save_figure(fig, report_dir / "figures" / "fig_stability_distribution")


def _figure_robustness(robustness: pd.DataFrame, report_dir: Path) -> None:
    subset = robustness.loc[robustness["scenario"].isin(["gaussian_noise", "mcar", "block_missing"])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, scenario in zip(axes, ["gaussian_noise", "mcar", "block_missing"]):
        group = subset.loc[subset["scenario"].eq(scenario)]
        for arm in ["ConceptFAN-NoAlpha", "PureNoFuzzy"]:
            arm_group = group.loc[group["model_arm"].eq(arm)].sort_values("level")
            axis.plot(arm_group["level"], arm_group["AUPRC_degradation_mean"], marker="o", label=arm, color=COLORS[arm])
        axis.set_title(scenario)
        axis.set_xlabel("Perturbation level")
        axis.set_ylabel("AUPRC degradation")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    _save_figure(fig, report_dir / "figures" / "fig_fuzzy_robustness")


def _figure_leakage(leakage: pd.DataFrame, report_dir: Path) -> None:
    subset = leakage.loc[leakage["diagnostic_model"].eq("l2_logistic_regression")]
    order = ["residuals", "matched_gaussian_noise", "permuted_residual_vectors", "missingness_only", "delta_time_only", "raw_value_only", "shuffled_labels"]
    pivot = subset.pivot(index="control", columns="channels", values="AUPRC_mean").reindex(order)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, color=["#006D77", "#83C5BE", "#E29578", "#3D405B"])
    ax.set_ylabel("Residual diagnostic test AUPRC")
    ax.set_xlabel("")
    ax.legend(title="Encoder channels", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, report_dir / "figures" / "fig_leakage_localization")


def _figure_pareto(performance: pd.DataFrame, stability: pd.DataFrame, report_dir: Path) -> None:
    raw_auprc = performance.loc[
        performance["reporting_state"].eq("raw") & performance["metric"].eq("AUPRC")
    ].set_index("model_arm")["mean"]
    signed = stability.loc[stability["representation"].eq("signed") & stability["metric"].eq("spearman")].set_index("model_arm")["mean"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for arm in ["ConceptFAN-NoAlpha", "PureNoFuzzy", "ConceptFAN-StabilityReg"]:
        if arm in raw_auprc and arm in signed:
            ax.scatter(raw_auprc[arm], signed[arm], s=80, color=COLORS[arm], label=arm)
            ax.annotate(arm, (raw_auprc[arm], signed[arm]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean held-out AUPRC")
    ax.set_ylabel("Mean signed contribution Spearman")
    ax.grid(alpha=0.25)
    _save_figure(fig, report_dir / "figures" / "fig_stability_pareto")


def _write_article_package(report_dir: Path, numbers: dict[str, float], claims: dict[str, bool]) -> None:
    article = report_dir / "article_update"
    article.mkdir(parents=True, exist_ok=True)
    auprc = numbers["conceptfan_auprc"]
    auroc = numbers["conceptfan_auroc"]
    spearman = numbers["conceptfan_spearman"]
    stab_spearman = numbers["stabilityreg_spearman"]
    robustness_difference = numbers["mcar20_difference"]
    calibration_delta = numbers["calibration_nll_delta"]
    ru_result = (
        f"На внешней эмпирической выборке PhysioNet 2012 (n=4000) ConceptFAN-NoAlpha достиг среднего test AUPRC {auprc:.3f} "
        f"и AUROC {auroc:.3f} по 30 запускам. Средняя episode-level воспроизводимость вкладов по Spearman составила {spearman:.3f}; "
        f"для StabilityReg она составила {stab_spearman:.3f}. При MCAR 20% разница деградации AUPRC ConceptFAN минус PureNoFuzzy была "
        f"{robustness_difference:+.3f}. Выбранная на calibration split посткалибровка изменила средний NLL на {calibration_delta:+.3f}."
    )
    en_result = (
        f"On the external empirical PhysioNet 2012 cohort (n=4,000), ConceptFAN-NoAlpha obtained a mean test AUPRC of {auprc:.3f} "
        f"and AUROC of {auroc:.3f} across 30 runs. Mean episode-level contribution stability was {spearman:.3f} by Spearman correlation, "
        f"compared with {stab_spearman:.3f} for StabilityReg. Under 20% MCAR, the ConceptFAN minus PureNoFuzzy AUPRC degradation "
        f"difference was {robustness_difference:+.3f}. Calibration-split-selected post-calibration changed mean NLL by {calibration_delta:+.3f}."
    )
    files = {
        "ABSTRACT_PATCH_RU.md": "# Патч аннотации\n\n" + ru_result + "\n",
        "ABSTRACT_PATCH_EN.md": "# Abstract patch\n\n" + en_result + "\n",
        "METHODS_REAL_DATA_RU.md": "# Методы: реальные данные\n\nИспользованы 4000 госпитализаций Set A PhysioNet/CinC 2012, первые 48 часов и patient-level split 2400/600/400/600. Все статистики препроцессинга вычислены только на train. Пять мягких траекторий являются proxy-концептами органной дисфункции, а не диагнозами.\n",
        "RESULTS_REAL_DATA_RU.md": "# Результаты: реальные данные\n\n" + ru_result + "\n",
        "DISCUSSION_PATCH_RU.md": "# Патч обсуждения\n\nРезультаты оценивают внешнюю эмпирическую переносимость архитектуры и устойчивость объяснений. Они не устанавливают причинный источник остаточного сигнала и не являются клинической валидацией.\n",
        "LIMITATIONS_PATCH_RU.md": "# Ограничения\n\nКонцепты заданы эвристическими proxy-траекториями; Set A является ретроспективной benchmark-выборкой; множественные запуски не заменяют внешнюю проспективную проверку; residual classifier локализует ассоциации, но не причинность.\n",
        "CONCLUSION_PATCH_RU.md": "# Патч заключения\n\n" + ru_result + " Результаты следует трактовать как внешнюю эмпирическую проверку, а не как подтверждение готовности к применению у постели пациента.\n",
        "TABLE_AND_FIGURE_MAP.md": """# Table and figure map

Retain the synthetic benchmark results as mechanistic evidence and add the real-data items below as external empirical evidence.

| Manuscript item | New source | Recommended placement | Proposed caption or update |
|---|---|---|---|
| Real-data cohort | `../tables/table_cohort.csv` | Methods, after dataset description | Patient-level frozen split and mortality characteristics for PhysioNet 2012 Set A. |
| Predictive results | `../tables/table_performance.csv` | Main results, new real-data subsection | Held-out predictive metrics across 30 crossed initialization and data-order seeds. |
| Calibration | `../tables/table_calibration.csv` | Main results or supplement | Calibration metrics before and after the calibration-split-selected transformation. |
| Explanation stability | `../tables/table_stability.csv` | Interpretability results | Episode-level cross-retraining agreement with hierarchical bootstrap intervals. |
| Robustness | `../tables/table_robustness.csv` | Robustness subsection | Deterministic perturbation sensitivity for ConceptFAN and the pure non-fuzzy ablation. |
| Residual signal | `../tables/table_leakage.csv` | Diagnostic supplement | Associative residual-signal localization across V, V+M, V+D and V+M+D inputs and controls. |
| StabilityReg comparison | `../tables/table_stability_reg.csv` | Interpretability results | Pre-specified AUPRC non-inferiority and contribution-stability comparison. |
| Figure 1 | `../figures/fig_real_pipeline.svg` | Methods | Real-data cohort, frozen split, temporal channels and proxy-concept pipeline. |
| Figure 2 | `../figures/fig_calibration.svg` | Main results | Held-out AUPRC/AUROC and patient-bootstrap reliability curves. |
| Figure 3 | `../figures/fig_stability_distribution.svg` | Interpretability results | Distribution of episode-level Spearman agreement across retraining pairs. |
| Figure 4 | `../figures/fig_fuzzy_robustness.svg` | Robustness subsection | AUPRC degradation under value noise and missingness perturbations. |
| Figure 5 | `../figures/fig_leakage_localization.svg` | Diagnostic supplement | Residual diagnostic AUPRC by input-channel ablation and negative control. |
| Figure 6 | `../figures/fig_stability_pareto.svg` | Interpretability results | Held-out AUPRC versus signed contribution stability for concept-mediated arms. |
""",
        "CLAIMS_ALLOWED.md": "# Claims allowed\n\n- Report observed metrics, uncertainty, and paired comparisons.\n- Describe concepts as proxy trajectories.\n- Describe residual results as associative localization.\n- Describe this study as external empirical benchmark validation.\n",
        "CLAIMS_FORBIDDEN.md": "# Claims forbidden\n\n- Clinical readiness or bedside utility.\n- Causal identification of leakage sources.\n- Fuzzy superiority without supported paired statistics.\n- StabilityReg as a solution unless both stability and non-inferiority criteria pass.\n",
    }
    if claims["stabilityreg_pareto"]:
        files["CLAIMS_ALLOWED.md"] += "- StabilityReg met the pre-specified stability/non-inferiority Pareto criterion in this benchmark.\n"
    for name, content in files.items():
        (article / name).write_text(content, encoding="utf-8")


def build_report(data: PreparedData, artifacts_root: Path, report_dir: Path) -> dict:
    tables_root = artifacts_root / "TABLES"
    predictive = pd.read_parquet(tables_root / "predictive_metrics.parquet")
    performance = pd.read_parquet(tables_root / "performance_summary.parquet")
    stability = pd.read_parquet(tables_root / "stability_aggregate.parquet")
    robustness = pd.read_parquet(tables_root / "robustness_aggregate.parquet")
    leakage = pd.read_parquet(tables_root / "leakage_aggregate.parquet")
    cohort = pd.read_parquet(tables_root / "cohort.parquet")
    paired = pd.read_parquet(tables_root / "paired_model_comparisons.parquet")
    calibration = _primary_calibration_table(predictive)
    stability_reg = pd.concat(
        [
            performance.loc[performance["model_arm"].isin(["ConceptFAN-NoAlpha", "ConceptFAN-StabilityReg"])],
            stability.loc[stability["model_arm"].isin(["ConceptFAN-NoAlpha", "ConceptFAN-StabilityReg"])],
        ],
        ignore_index=True,
        sort=False,
    )
    _write_table(cohort, report_dir / "tables" / "table_cohort.csv")
    _write_table(performance, report_dir / "tables" / "table_performance.csv")
    _write_table(calibration, report_dir / "tables" / "table_calibration.csv")
    _write_table(stability, report_dir / "tables" / "table_stability.csv")
    _write_table(robustness, report_dir / "tables" / "table_robustness.csv")
    _write_table(leakage, report_dir / "tables" / "table_leakage.csv")
    _write_table(stability_reg, report_dir / "tables" / "table_stability_reg.csv")
    _write_table(paired, report_dir / "tables" / "table_paired_statistics.csv")
    _figure_pipeline(data, report_dir)
    _figure_performance(predictive, artifacts_root, report_dir)
    _figure_stability(tables_root / "episode_pairwise_stability.parquet", report_dir)
    _figure_robustness(robustness, report_dir)
    _figure_leakage(leakage, report_dir)
    _figure_pareto(performance, stability, report_dir)
    cf_auprc = _metric_value(performance, "ConceptFAN-NoAlpha", "raw", "AUPRC")
    cf_auroc = _metric_value(performance, "ConceptFAN-NoAlpha", "raw", "AUROC")
    signed_spearman = stability.loc[stability["representation"].eq("signed") & stability["metric"].eq("spearman")]
    cf_spearman = float(signed_spearman.loc[signed_spearman["model_arm"].eq("ConceptFAN-NoAlpha"), "mean"].iloc[0])
    stab_spearman = float(signed_spearman.loc[signed_spearman["model_arm"].eq("ConceptFAN-StabilityReg"), "mean"].iloc[0])
    mcar = robustness.loc[robustness["scenario"].eq("mcar") & robustness["level"].eq(0.2)]
    mcar_difference = float(
        mcar.loc[mcar["model_arm"].eq("ConceptFAN-NoAlpha"), "AUPRC_degradation_mean"].iloc[0]
        - mcar.loc[mcar["model_arm"].eq("PureNoFuzzy"), "AUPRC_degradation_mean"].iloc[0]
    )
    cf_cal = calibration.loc[calibration["model_arm"].eq("ConceptFAN-NoAlpha")].set_index("state")
    nll_delta = float(cf_cal.loc["primary_calibrated", "NLL_mean"] - cf_cal.loc["raw", "NLL_mean"])
    cf_auprc_low = _metric_value(performance, "ConceptFAN-NoAlpha", "raw", "AUPRC", "ci95_low")
    cf_auprc_high = _metric_value(performance, "ConceptFAN-NoAlpha", "raw", "AUPRC", "ci95_high")
    stab_auprc = _metric_value(performance, "ConceptFAN-StabilityReg", "raw", "AUPRC")
    stabilityreg_pareto = stab_spearman > cf_spearman and stab_auprc >= cf_auprc - 0.01
    numbers = {
        "conceptfan_auprc": cf_auprc,
        "conceptfan_auroc": cf_auroc,
        "conceptfan_spearman": cf_spearman,
        "stabilityreg_spearman": stab_spearman,
        "mcar20_difference": mcar_difference,
        "calibration_nll_delta": nll_delta,
    }
    claims = {"stabilityreg_pareto": stabilityreg_pareto}
    _write_article_package(report_dir, numbers, claims)
    result_paragraph = (
        f"ConceptFAN-NoAlpha mean held-out AUPRC was {cf_auprc:.3f} (run-level bootstrap 95% CI {cf_auprc_low:.3f}-{cf_auprc_high:.3f}) "
        f"and AUROC was {cf_auroc:.3f}. Signed episode-level cross-retraining Spearman was {cf_spearman:.3f}; StabilityReg was {stab_spearman:.3f}. "
        f"The MCAR 20% degradation difference versus PureNoFuzzy was {mcar_difference:+.3f}. Primary calibration changed mean NLL by {nll_delta:+.3f}."
    )
    docs = {
        "EXECUTIVE_SUMMARY.md": "# Executive summary\n\n" + result_paragraph + "\n\nThese are benchmark results, not clinical readiness evidence.\n",
        "METHODS_FOR_PAPER.md": "# Methods for paper\n\nWe used the 4,000-patient PhysioNet/CinC 2012 Set A cohort with a frozen patient-level train/validation/calibration/test split of 2400/600/400/600. Models received the first 48 hours represented as values, observation masks, and elapsed-time channels. All normalization and proxy-concept thresholds were estimated on training data only. Five canonical architectures were evaluated across 30 crossed initialization/data-order seeds.\n",
        "RESULTS_FOR_PAPER.md": "# Results for paper\n\n" + result_paragraph + "\n",
        "LIMITATIONS_FOR_PAPER.md": "# Limitations for paper\n\nThe five concepts are heuristic soft proxies rather than diagnoses. Set A is retrospective. Residual classification identifies associations, not causes. Multiple retrainings characterize algorithmic variability but do not establish prospective clinical utility.\n",
        "REPRODUCE.md": """# Reproduce from raw files

Run from the repository root at commit `e60d9a3da40fc05b66874f89ad61015d6e07d33c` plus the PhysioNet last-run implementation on this branch:

```bash
bash scripts/run_physionet2012_last_run.sh \\
  --set-a-zip data/physionet2012/raw/set-a.zip \\
  --outcomes data/physionet2012/raw/Outcomes-a.txt \\
  --artifacts-dir artifacts/physionet2012_last_run \\
  --device cuda \\
  --resume
```

If `Outcomes-a.txt` is absent, the pipeline downloads it from the configured official PhysioNet endpoint before the raw-data audit. Raw patient files are never added to the report archive.
""",
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, content in docs.items():
        (report_dir / name).write_text(content, encoding="utf-8")
    manifests = report_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    fit = pd.read_parquet(tables_root / "checkpoint_fit_metrics.parquet")
    fit.to_csv(manifests / "runs_manifest.csv", index=False)
    raw_audit = json.loads((artifacts_root / "raw_audit" / "audit.json").read_text(encoding="utf-8"))
    (manifests / "data_sha256.txt").write_text(
        "\n".join(f"{digest}  {name}" for name, digest in raw_audit["hashes"].items()) + "\n", encoding="utf-8"
    )
    (manifests / "environment_lock.txt").write_text(
        f"python={sys.version}\nplatform={platform.platform()}\ntorch={torch.__version__}\nnumpy={np.__version__}\n", encoding="utf-8"
    )
    report_manifest = {
        "status": "PHYSIONET2012_REPORT_COMPLETE",
        "numbers": numbers,
        "claims": claims,
        "report_files": len([path for path in report_dir.rglob("*") if path.is_file()]),
    }
    (report_dir / "report_manifest.json").write_text(json.dumps(report_manifest, indent=2), encoding="utf-8")
    write_report_hash_index(report_dir)
    return report_manifest
