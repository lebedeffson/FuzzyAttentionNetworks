from __future__ import annotations

import numpy as np


class TrainNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype("float32")
        self.std = np.maximum(std.astype("float32"), 1e-6)

    @classmethod
    def fit(cls, x_train: np.ndarray, mask_train: np.ndarray | None = None):
        if mask_train is None:
            mask_train = np.isfinite(x_train)
        safe = np.where(mask_train, x_train, np.nan)
        mean = np.nanmean(safe, axis=(0, 1))
        std = np.nanstd(safe, axis=(0, 1))
        return cls(np.nan_to_num(mean), np.nan_to_num(std, nan=1.0))

    def transform(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        z = (np.nan_to_num(x, nan=0.0) - self.mean) / self.std
        if mask is not None:
            z = np.where(mask, z, 0.0)
        return z.astype("float32")
