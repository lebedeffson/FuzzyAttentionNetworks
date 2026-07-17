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
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=180, bbox_inches="tight")
    plt.close(figure)


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
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")
    labels = [
        ("PhysioNet 2012 Set A\n4,000 ICU stays", "#DCEAF0"),
        ("Patient split\n2400 / 600 / 400 / 600", "#E8E8E8"),
        ("48-hour V / M / D\ntrain-only preprocessing", "#F5E6CC"),
        ("5 proxy concepts\nsoft trajectories", "#DDEAD1"),
        ("5 model arms\n30 runs each", "#E8DDF0"),
        ("Calibration, stability,\nrobustness, sufficiency", "#F4D6D0"),
    ]
    for index, (label, color) in enumerate(labels):
        x = 0.02 + index * 0.162
        ax.add_patch(plt.Rectangle((x, 0.35), 0.135, 0.3, facecolor=color, edgecolor="#333333", linewidth=1))
        ax.text(x + 0.0675, 0.5, label, ha="center", va="center", fontsize=9)
        if index < len(labels) - 1:
            ax.annotate("", xy=(x + 0.158, 0.5), xytext=(x + 0.137, 0.5), arrowprops={"arrowstyle": "->"})
    ax.text(0.02, 0.14, f"Temporal variables: {len(data.variables)} | Proxy concept observability: {data.concept_mask.mean():.1%}", fontsize=10)
    _save_figure(fig, report_dir / "figures" / "fig_real_pipeline")


def _figure_performance(predictive: pd.DataFrame, report_dir: Path) -> None:
    raw = predictive.loc[predictive["calibration_method"].eq("none")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    arms = list(COLORS)
    x = np.arange(len(arms))
    for position, metric in enumerate(["AUPRC", "AUROC"]):
        means = [raw.loc[raw["model_arm"].eq(arm), metric].mean() for arm in arms]
        stds = [raw.loc[raw["model_arm"].eq(arm), metric].std() for arm in arms]
        axes[position].bar(x, means, yerr=stds, color=[COLORS[arm] for arm in arms], capsize=3)
        axes[position].set_xticks(x, [arm.replace("ConceptFAN-", "CF-") for arm in arms], rotation=25, ha="right")
        axes[position].set_ylabel(metric)
        axes[position].grid(axis="y", alpha=0.25)
    fig.suptitle("Held-out predictive performance across 30 runs")
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
        "TABLE_AND_FIGURE_MAP.md": "# Table and figure map\n\n| Manuscript item | New source | Placement |\n|---|---|---|\n| Real-data cohort | `../tables/table_cohort.csv` | Methods / cohort |\n| Predictive results | `../tables/table_performance.csv` | Main results |\n| Calibration | `../tables/table_calibration.csv` | Main results or supplement |\n| Explanation stability | `../tables/table_stability.csv` | Interpretability results |\n| Robustness | `../tables/table_robustness.csv` | Robustness section |\n| Residual signal | `../tables/table_leakage.csv` | Diagnostic supplement |\n| Figures 1-6 | `../figures/fig_*.svg` | Replace surrogate-only plots |\n",
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
    _figure_performance(predictive, report_dir)
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
    hash_rows = []
    for path in sorted(report_dir.rglob("*")):
        if path.is_file() and path.name != "artifacts_sha256.txt":
            hash_rows.append(f"{sha256_file(path)}  {path.relative_to(report_dir)}")
    (manifests / "artifacts_sha256.txt").write_text("\n".join(hash_rows) + "\n", encoding="utf-8")
    report_manifest = {
        "status": "PHYSIONET2012_REPORT_COMPLETE",
        "numbers": numbers,
        "claims": claims,
        "report_files": len([path for path in report_dir.rglob("*") if path.is_file()]),
    }
    (report_dir / "report_manifest.json").write_text(json.dumps(report_manifest, indent=2), encoding="utf-8")
    return report_manifest
