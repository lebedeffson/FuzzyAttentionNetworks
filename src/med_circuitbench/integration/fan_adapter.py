from __future__ import annotations

from typing import Any, Dict

import torch


class FANAdapter:
    """Read-only adapter for existing FAN models.

    The adapter intentionally avoids subclassing or editing existing FAN classes.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
        if isinstance(out, dict):
            return out
        return {"output": out}
