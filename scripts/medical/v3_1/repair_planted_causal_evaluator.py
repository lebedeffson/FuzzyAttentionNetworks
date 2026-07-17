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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from med_circuitbench.planted.interventions import (  # noqa: E402
    DESCENDANTS,
    DIRECT_EDGES,
    STATE_INDEX,
    descendant_pairs,
    direct_edge_counterfactual,
    do_node_intervention,
    standardized_effect,
    target_train_std,
    total_effect_counterfactual,
)
from med_circuitbench.planted.model import EDGES, STATE_NAMES, PlantedCircuitModel  # noqa: E402
from scripts.medical.v2.run_v2_program import make_episodes, split_frame, subset_cols  # noqa: E402
from scripts.medical.v3.run_research_program import DEVICE, prepare_arrays  # noqa: E402
from scripts.medical.v3_1.run_planted_adaptive_sctc import bh_q_values  # noqa: E402


MARGIN = 0.01


def paired_bootstrap(effect_by_episode: np.ndarray, seed: int, n_boot: int = 2000) -> dict:
    effect = np.asarray(effect_by_episode, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if effect.size == 0:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "p_value": float("nan")}
    samples = np.asarray([rng.choice(effect, size=effect.size, replace=True).mean() for _ in range(n_boot)])
    p_value = float(min(1.0, 2.0 * min(np.mean(samples <= 0.0), np.mean(samples >= 0.0))))
    return {
        "mean": float(effect.mean()),
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
        "p_value": p_value,
    }


def _episode_effect(nodes_after: torch.Tensor, nodes_before: torch.Tensor, target: str, train_std: torch.Tensor) -> np.ndarray:
    effect = standardized_effect(nodes_after, nodes_before, target, train_std)
    return effect.detach().cpu().numpy().mean(axis=1)


def _summarize_graph(rows: pd.DataFrame, truth: set[tuple[str, str]], prefix: str) -> dict:
    accepted = rows[rows["accepted"].fillna(False)]
    accepted_pairs = {(str(r.source), str(r.target)) for r in accepted.itertuples()}
    tp = len(accepted_pairs & truth)
    fp = len(accepted_pairs - truth)
    fn = len(truth - accepted_pairs)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    controls = rows[~rows.apply(lambda r: (str(r["source"]), str(r["target"])) in truth, axis=1)]
    fpr = float(controls["accepted"].fillna(False).mean()) if len(controls) else float("nan")
    return {
        f"{prefix}_TP": int(tp),
        f"{prefix}_FP": int(fp),
        f"{prefix}_FN": int(fn),
        f"{prefix}_precision": float(precision),
        f"{prefix}_recall": float(recall),
        f"{prefix}_F1": float(f1),
        f"{prefix}_negative_control_fpr": fpr,
    }


def evaluate_seed(seed: int, cfg: dict, output: Path) -> dict:
    frame = make_episodes(seed, cfg, "clean")
    arrays = prepare_arrays(split_frame(frame, seed), subset_cols("full_input"))
    model = PlantedCircuitModel(seed=seed, d_model=int(cfg["model"]["d_model"])).to(DEVICE)
    with torch.no_grad():
        train_nodes = model(torch.from_numpy(arrays["c_train_seq"]).float().to(DEVICE)).nodes
        val_nodes = model(torch.from_numpy(arrays["c_val_seq"]).float().to(DEVICE)).nodes

    direct_rows = []
    total_rows = []
    sanity = {}
    p_values_direct = []
    p_values_total = []

    for source in STATE_NAMES:
        source_idx = STATE_INDEX[source]
        ablate_value = torch.zeros_like(val_nodes[..., source_idx])
        source_push_scale = train_nodes[..., source_idx].std().clamp_min(1e-8)
        push_value = val_nodes[..., source_idx] + source_push_scale

        do_ablated = do_node_intervention(val_nodes, source, ablate_value).nodes
        if source == "R":
            sanity["do_R_changes_I_max_abs"] = float((do_ablated[..., STATE_INDEX["I"]] - val_nodes[..., STATE_INDEX["I"]]).abs().max().detach().cpu())
        if source == "V":
            sanity["do_V_changes_I_or_R_max_abs"] = float(
                torch.stack(
                    [
                        (do_ablated[..., STATE_INDEX["I"]] - val_nodes[..., STATE_INDEX["I"]]).abs().max(),
                        (do_ablated[..., STATE_INDEX["R"]] - val_nodes[..., STATE_INDEX["R"]]).abs().max(),
                    ]
                )
                .max()
                .detach()
                .cpu()
            )
        if source == "O":
            sanity["do_O_changes_I_R_or_V_max_abs"] = float(
                torch.stack(
                    [
                        (do_ablated[..., STATE_INDEX["I"]] - val_nodes[..., STATE_INDEX["I"]]).abs().max(),
                        (do_ablated[..., STATE_INDEX["R"]] - val_nodes[..., STATE_INDEX["R"]]).abs().max(),
                        (do_ablated[..., STATE_INDEX["V"]] - val_nodes[..., STATE_INDEX["V"]]).abs().max(),
                    ]
                )
                .max()
                .detach()
                .cpu()
            )
            sanity["do_O_changes_S_mean_abs"] = float((do_ablated[..., STATE_INDEX["S"]] - val_nodes[..., STATE_INDEX["S"]]).abs().mean().detach().cpu())

        for target in STATE_NAMES:
            if source == target:
                continue
            target_std = target_train_std(train_nodes, target)

            direct_after = direct_edge_counterfactual(val_nodes, source, target, ablate_value).nodes
            direct_push = direct_edge_counterfactual(val_nodes, source, target, push_value).nodes
            direct_effect = _episode_effect(direct_after, val_nodes, target, target_std)
            direct_push_effect = _episode_effect(direct_push, val_nodes, target, target_std)
            direct_boot = paired_bootstrap(direct_effect, seed + 1000 + 17 * STATE_INDEX[source] + STATE_INDEX[target])
            direct_p = direct_boot["p_value"]
            p_values_direct.append(direct_p)
            direct_rows.append(
                {
                    "seed": seed,
                    "source": source,
                    "target": target,
                    "relation": "direct_edge" if (source, target) in DIRECT_EDGES else "direct_non_edge_control",
                    "effect_mean": direct_boot["mean"],
                    "ci_lower": direct_boot["ci_lower"],
                    "ci_upper": direct_boot["ci_upper"],
                    "p_value": direct_p,
                    "q_value": np.nan,
                    "push_effect_mean": float(direct_push_effect.mean()),
                    "accepted": False,
                    "directional_null_status": "NOT_APPLICABLE_LOW_RANK",
                }
            )

            total_after = total_effect_counterfactual(val_nodes, source, ablate_value).nodes
            total_push = total_effect_counterfactual(val_nodes, source, push_value).nodes
            total_effect = _episode_effect(total_after, val_nodes, target, target_std)
            total_push_effect = _episode_effect(total_push, val_nodes, target, target_std)
            total_boot = paired_bootstrap(total_effect, seed + 2000 + 17 * STATE_INDEX[source] + STATE_INDEX[target])
            total_p = total_boot["p_value"]
            p_values_total.append(total_p)
            total_rows.append(
                {
                    "seed": seed,
                    "source": source,
                    "target": target,
                    "relation": "descendant_total_effect" if target in DESCENDANTS[source] else "non_descendant_control",
                    "effect_mean": total_boot["mean"],
                    "ci_lower": total_boot["ci_lower"],
                    "ci_upper": total_boot["ci_upper"],
                    "p_value": total_p,
                    "q_value": np.nan,
                    "push_effect_mean": float(total_push_effect.mean()),
                    "accepted": False,
                    "directional_null_status": "NOT_APPLICABLE_LOW_RANK",
                }
            )

    direct_q = bh_q_values(np.asarray(p_values_direct, dtype=float))
    total_q = bh_q_values(np.asarray(p_values_total, dtype=float))
    for row, q_value in zip(direct_rows, direct_q):
        excludes_zero = row["ci_lower"] > 0.0 or row["ci_upper"] < 0.0
        sign_ok = np.sign(row["push_effect_mean"]) == -np.sign(row["effect_mean"])
        row["q_value"] = float(q_value)
        row["accepted"] = bool(excludes_zero and abs(row["effect_mean"]) > MARGIN and sign_ok and q_value <= 0.05)
    for row, q_value in zip(total_rows, total_q):
        excludes_zero = row["ci_lower"] > 0.0 or row["ci_upper"] < 0.0
        sign_ok = np.sign(row["push_effect_mean"]) == -np.sign(row["effect_mean"])
        row["q_value"] = float(q_value)
        row["accepted"] = bool(excludes_zero and abs(row["effect_mean"]) > MARGIN and sign_ok and q_value <= 0.05)

    direct = pd.DataFrame(direct_rows)
    total = pd.DataFrame(total_rows)
    direct.to_parquet(output / f"seed{seed}_oracle_direct_effects.parquet", index=False)
    total.to_parquet(output / f"seed{seed}_oracle_total_effects.parquet", index=False)
    direct_metrics = _summarize_graph(direct, set(DIRECT_EDGES), "direct")
    total_metrics = _summarize_graph(total, descendant_pairs(), "total")
    o_to_s = bool(direct[(direct["source"].eq("O")) & (direct["target"].eq("S"))]["accepted"].iloc[0])
    reverse = total[total.apply(lambda r: str(r["source"]) in DESCENDANTS[str(r["target"])], axis=1)]
    no_reverse = bool(not reverse["accepted"].fillna(False).any())
    return {
        "seed": seed,
        **direct_metrics,
        **total_metrics,
        "O_TO_S_ACCEPTED": o_to_s,
        "NO_REVERSE_CAUSAL_EFFECTS": no_reverse,
        **sanity,
    }


def gate_status(metrics: pd.DataFrame) -> dict:
    rows = []
    for _, row in metrics.iterrows():
        seed_pass = bool(
            row["direct_F1"] >= 0.95
            and row["total_F1"] >= 0.95
            and row["direct_negative_control_fpr"] <= 0.05
            and row["total_negative_control_fpr"] <= 0.05
            and row["O_TO_S_ACCEPTED"]
            and row["NO_REVERSE_CAUSAL_EFFECTS"]
            and row.get("do_R_changes_I_max_abs", 0.0) <= 1e-8
            and row.get("do_V_changes_I_or_R_max_abs", 0.0) <= 1e-8
            and row.get("do_O_changes_I_R_or_V_max_abs", 0.0) <= 1e-8
            and row.get("do_O_changes_S_mean_abs", 0.0) > 1e-6
        )
        rows.append({"seed": int(row["seed"]), "pass": seed_pass})
    pass_count = sum(item["pass"] for item in rows)
    return {
        "status": "ORACLE_CAUSAL_EVALUATOR_PASS" if pass_count == len(rows) else "ORACLE_CAUSAL_EVALUATOR_FAIL",
        "seed_pass_count": int(pass_count),
        "seed_results": rows,
        "ORACLE_NODE_F1": 1.0,
        "ORACLE_DIRECT_EDGE_F1": float(metrics["direct_F1"].mean()),
        "ORACLE_TOTAL_EFFECT_F1": float(metrics["total_F1"].mean()),
        "ORACLE_DIRECT_NEGATIVE_CONTROL_FPR": float(metrics["direct_negative_control_fpr"].mean()),
        "ORACLE_TOTAL_NEGATIVE_CONTROL_FPR": float(metrics["total_negative_control_fpr"].mean()),
        "O_TO_S_ACCEPTED": bool(metrics["O_TO_S_ACCEPTED"].all()),
        "NO_REVERSE_CAUSAL_EFFECTS": bool(metrics["NO_REVERSE_CAUSAL_EFFECTS"].all()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medical/v3/full.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", default="artifacts/medical/v3_1/repaired_causal_evaluator")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([evaluate_seed(seed, cfg, output) for seed in args.seeds])
    metrics.to_csv(output / "oracle_causal_evaluator_metrics.csv", index=False)
    gate = gate_status(metrics)
    gate["output"] = str(output)
    gate["primary_null"] = "direct_non_edge_reverse_wrong_layer_controls_with_paired_bootstrap"
    gate["directional_null_status"] = "NOT_APPLICABLE_LOW_RANK"
    (output / "oracle_causal_evaluator_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

