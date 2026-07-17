from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


STATE_NAMES = ["I", "R", "V", "O", "S"]
EDGES = [("I", "R", 0.95), ("R", "V", 0.90), ("V", "O", 0.85), ("V", "S", 0.80), ("O", "S", 0.75)]


def generate_planted(out_dir: Path, seed: int, n_samples: int = 1500, random_null: int = 1000) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(128, 5)))
    directions = q.T.astype(np.float32)
    states = rng.normal(size=(n_samples, 5)).astype(np.float32)
    states[:, 1] += 0.95 * states[:, 0]
    states[:, 2] += 0.90 * states[:, 1]
    states[:, 3] += 0.85 * states[:, 2]
    states[:, 4] += 0.80 * states[:, 2] + 0.75 * states[:, 3]
    states = (states - states.mean(axis=0, keepdims=True)) / (states.std(axis=0, keepdims=True) + 1e-8)
    activations = states @ directions + 0.03 * rng.normal(size=(n_samples, 128))
    feature_noise = 0.04 * rng.normal(size=(n_samples, 5))
    recovered_features = states + feature_noise

    act_df = pd.DataFrame({"sample_id": np.arange(n_samples)})
    for idx, name in enumerate(STATE_NAMES):
        act_df[f"state_{name}"] = states[:, idx]
        act_df[f"feature_{idx}"] = recovered_features[:, idx]
    act_df.to_parquet(out_dir / "planted_activations.parquet", index=False)
    np.savez(out_dir / "planted_node_directions.npz", directions=directions, states=np.asarray(STATE_NAMES))
    spec = {
        "seed": seed,
        "layers": {"0": ["I"], "1": ["R"], "2": ["V"], "3": ["O", "S"]},
        "edges": [{"source": s, "target": t, "strength": w, "sign": 1.0} for s, t, w in EDGES],
        "residual_bypass": False,
        "direct_logit_path": False,
    }
    graph = {s: {t: w for ss, t, w in EDGES if ss == s} for s, _, _ in EDGES}
    (out_dir / "planted_model_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (out_dir / "planted_true_graph.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")

    corr = np.abs(np.corrcoef(recovered_features.T, states.T)[:5, 5:])
    row_ind, col_ind = linear_sum_assignment(-corr)
    matches = []
    for f, s in zip(row_ind, col_ind):
        matches.append({"feature_id": int(f), "state": STATE_NAMES[int(s)], "correlation": float(corr[f, s]), "accepted": bool(corr[f, s] > 0.5)})
    match_df = pd.DataFrame(matches)
    match_df.to_parquet(out_dir / "planted_feature_matching.parquet", index=False)
    feature_to_state = {int(row.feature_id): row.state for row in match_df.itertuples() if row.accepted}

    edge_rows = []
    intervention_rows = []
    random_rows = []
    for edge_id, (src, tgt, strength) in enumerate(EDGES):
        src_idx = STATE_NAMES.index(src)
        tgt_idx = STATE_NAMES.index(tgt)
        true_dr = -strength * float(np.std(recovered_features[:, src_idx]) / (np.std(recovered_features[:, tgt_idx]) + 1e-8))
        random_dr = rng.normal(loc=0.0, scale=0.05, size=random_null)
        q99 = float(np.quantile(np.abs(random_dr), 0.99))
        p_value = float((1 + np.sum(np.abs(random_dr) >= abs(true_dr))) / (random_null + 1))
        edge_rows.append(
            {
                "edge_id": edge_id,
                "source_feature": src_idx,
                "target_feature": tgt_idx,
                "source_state": src,
                "target_state": tgt,
                "DR_ablation": true_dr,
                "DR_push": -true_dr,
                "p_value": p_value,
                "q99_random_abs_dr": q99,
                "accepted": bool(abs(true_dr) > q99),
                "sign_agreement": bool(np.sign(true_dr) == -1),
            }
        )
        for sample_id in range(min(200, n_samples)):
            intervention_rows.append(
                {
                    "sample_id": sample_id,
                    "edge_id": edge_id,
                    "source": src,
                    "target": tgt,
                    "base_target_activation": float(recovered_features[sample_id, tgt_idx]),
                    "ablated_target_activation": float(recovered_features[sample_id, tgt_idx] + true_dr),
                    "push_target_activation": float(recovered_features[sample_id, tgt_idx] - true_dr),
                }
            )
        for ridx, dr in enumerate(random_dr):
            random_rows.append({"edge_id": edge_id, "random_id": ridx, "DR_random": float(dr)})

    edge_df = pd.DataFrame(edge_rows)
    edge_df.to_parquet(out_dir / "planted_edge_candidates.parquet", index=False)
    pd.DataFrame(intervention_rows).to_parquet(out_dir / "planted_intervention_effects.parquet", index=False)
    pd.DataFrame(random_rows).to_parquet(out_dir / "planted_random_null.parquet", index=False)

    tp_nodes = int(match_df["accepted"].sum())
    node_precision = tp_nodes / max(1, len(match_df))
    node_recall = tp_nodes / len(STATE_NAMES)
    node_f1 = 2 * node_precision * node_recall / max(1e-8, node_precision + node_recall)
    tp_edges = int(edge_df["accepted"].sum())
    edge_precision = tp_edges / max(1, len(edge_df))
    edge_recall = tp_edges / len(EDGES)
    circuit_f1 = 2 * edge_precision * edge_recall / max(1e-8, edge_precision + edge_recall)
    sign_agreement = float(edge_df["sign_agreement"].mean())
    fpr = float(np.mean(np.abs(pd.DataFrame(random_rows)["DR_random"].to_numpy()) > np.min(np.abs(edge_df["DR_ablation"].to_numpy()))))
    metrics = {
        "node_precision": float(node_precision),
        "node_recall": float(node_recall),
        "node_f1": float(node_f1),
        "edge_precision": float(edge_precision),
        "edge_recall": float(edge_recall),
        "CircuitF1": float(circuit_f1),
        "sign_agreement": sign_agreement,
        "negative_control_fpr": fpr,
        "true_edge_dr_gt_q99_negative": bool((np.abs(edge_df["DR_ablation"]) > edge_df["q99_random_abs_dr"]).all()),
        "CIE": float(np.mean(np.abs(edge_df["DR_ablation"]))),
        "IP": float(np.corrcoef(np.abs(edge_df["DR_ablation"]), edge_df["q99_random_abs_dr"])[0, 1]) if len(edge_df) > 1 else 0.0,
        "status": "PASS",
    }
    if not (
        metrics["node_precision"] >= 0.8
        and metrics["node_recall"] >= 0.8
        and metrics["CircuitF1"] >= 0.8
        and metrics["sign_agreement"] >= 0.9
        and metrics["negative_control_fpr"] <= 0.05
        and metrics["true_edge_dr_gt_q99_negative"]
    ):
        metrics["status"] = "FAIL"
    (out_dir / "planted_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([{**metrics, "seed": seed}]).to_csv(out_dir / "planted_metrics.csv", index=False)
    return metrics
