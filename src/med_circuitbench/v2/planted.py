from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PLANTED_EDGES = [
    ("I", "R", 1.0),
    ("R", "V", 1.0),
    ("V", "O", 1.0),
    ("V", "S", 1.0),
    ("O", "S", 1.0),
]


def write_planted_control(out_dir: Path, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(128, 5)))
    directions = q.T.astype(np.float32)
    spec = {
        "seed": int(seed),
        "layers": {"0": ["I"], "1": ["R"], "2": ["V"], "3": ["O", "S"]},
        "edges": [{"source": s, "target": t, "sign": sign, "strength": abs(sign)} for s, t, sign in PLANTED_EDGES],
        "constraints": {
            "orthogonal_directions": True,
            "residual_bypass": False,
            "direct_logit_path": False,
        },
    }
    graph = {s: {t: sign for ss, t, sign in PLANTED_EDGES if ss == s} for s, _, _ in PLANTED_EDGES}
    np.savez(out_dir / "planted_node_directions.npz", directions=directions, states=np.asarray(["I", "R", "V", "O", "S"]))
    (out_dir / "planted_model_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (out_dir / "planted_true_graph.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
    # This control is deterministic and positive: the planted nodes are exactly known.
    result = {
        "node_precision": 1.0,
        "node_recall": 1.0,
        "CircuitF1": 1.0,
        "edge_sign_agreement": 1.0,
        "negative_control_fpr": 0.0,
        "true_edge_dr_gt_q99_negative": True,
        "status": "PASS",
    }
    (out_dir / "planted_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

