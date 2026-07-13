from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil
from typing import Dict, Iterable

import numpy as np
import torch


def collect_ffn_activations(model, loader, device: str = "cpu") -> Dict[str, np.ndarray]:
    model.eval()
    logits, probs, targets, h_layers, a_layers = [], [], [], None, None
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch["x"], batch["y"]
            out = model(x.to(device), return_activations=True)
            logits.append(out["logit"].detach().cpu().numpy())
            probs.append(out["probability"].detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
            if h_layers is None:
                h_layers = [[] for _ in out["h_ffn"]]
                a_layers = [[] for _ in out["a_ffn"]]
            for i, h in enumerate(out["h_ffn"]):
                h_layers[i].append(h.detach().cpu().numpy())
            for i, a in enumerate(out["a_ffn"]):
                a_layers[i].append(a.detach().cpu().numpy())
    return {
        "logit": np.concatenate(logits),
        "probability": np.concatenate(probs),
        "target": np.concatenate(targets),
        "h_ffn": np.asarray([np.concatenate(v) for v in h_layers]),
        "a_ffn": np.asarray([np.concatenate(v) for v in a_layers]),
    }


def max_prediction_delta_with_hooks(model, x: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        plain = model(x, return_activations=False)["logit"]
        hooked = model(x, return_activations=True)["logit"]
    return float((plain - hooked).abs().max().item())


def write_activation_store_zarr(store: Dict[str, np.ndarray], out_dir: Path) -> None:
    import zarr

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"arrays": {}}
    for key, value in store.items():
        arr = np.asarray(value)
        zarr.save_array(str(out_dir / key), arr)
        manifest["arrays"][key] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }
    (out_dir / "activation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
