from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def fidelity_metrics(predictions: pd.DataFrame) -> dict:
    y = predictions["target"].to_numpy(int)
    p0 = predictions["original_probability"].to_numpy(float)
    p1 = predictions["reconstructed_probability"].to_numpy(float)
    l0 = predictions["original_logit"].to_numpy(float)
    l1 = predictions["reconstructed_logit"].to_numpy(float)
    original_auroc = float(roc_auc_score(y, p0)) if np.unique(y).size > 1 else float("nan")
    reconstructed_auroc = float(roc_auc_score(y, p1)) if np.unique(y).size > 1 else float("nan")
    return {
        "original_AUROC": original_auroc,
        "reconstructed_AUROC": reconstructed_auroc,
        "delta_AUROC": abs(original_auroc - reconstructed_auroc) if np.isfinite(original_auroc) else float("nan"),
        "original_AUPRC": float(average_precision_score(y, p0)),
        "reconstructed_AUPRC": float(average_precision_score(y, p1)),
        "delta_AUPRC": abs(float(average_precision_score(y, p0)) - float(average_precision_score(y, p1))),
        "probability_MAE": float(np.mean(np.abs(p0 - p1))),
        "logit_MAE": float(np.mean(np.abs(l0 - l1))),
    }


def feature_catalog(
    z: torch.Tensor,
    decoder_weight: torch.Tensor,
    state_targets: np.ndarray,
    state_names: list[str],
    layer: int,
    capture_point: str,
) -> pd.DataFrame:
    z_np = z.detach().cpu().reshape(-1, z.shape[-1]).numpy()
    states = state_targets.reshape(-1, state_targets.shape[-1])
    rows = []
    decoder = decoder_weight.detach().cpu().T.numpy()
    for feature_id in range(z_np.shape[1]):
        feature = z_np[:, feature_id]
        row = {
            "feature_id": feature_id,
            "layer": layer,
            "capture_point": capture_point,
            "activation_frequency": float(np.mean(feature > 0)),
            "mean_activation": float(feature.mean()),
            "support": int(np.sum(feature > 0)),
            "decoder_norm": float(np.linalg.norm(decoder[feature_id])),
        }
        for idx, name in enumerate(state_names):
            target = states[:, idx]
            if np.std(feature) == 0 or np.std(target) == 0:
                corr = np.nan
            else:
                corr = float(stats.pearsonr(feature, target).statistic)
            row[f"{name}_correlation"] = corr
        rows.append(row)
    return pd.DataFrame(rows)
