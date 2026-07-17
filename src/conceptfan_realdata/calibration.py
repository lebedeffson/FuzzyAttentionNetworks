from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from .metrics import sigmoid


@dataclass
class Calibrator:
    method: str
    parameters: dict[str, float | list[float]]

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        if self.method == "none":
            return logits
        if self.method == "temperature":
            return logits / float(self.parameters["temperature"])
        if self.method == "platt":
            return float(self.parameters["slope"]) * logits + float(self.parameters["intercept"])
        raise ValueError(f"{self.method} does not define transformed logits")

    def predict(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        if self.method != "isotonic":
            return sigmoid(self.transform_logits(logits))
        x = np.asarray(self.parameters["x_thresholds"], dtype=np.float64)
        y = np.asarray(self.parameters["y_thresholds"], dtype=np.float64)
        return np.clip(np.interp(logits, x, y, left=y[0], right=y[-1]), 1e-8, 1.0 - 1e-8)

    def to_dict(self) -> dict:
        return {"method": self.method, "parameters": self.parameters}


def fit_calibrators(logits: np.ndarray, y: np.ndarray) -> dict[str, Calibrator]:
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    result: dict[str, Calibrator] = {"none": Calibrator("none", {})}

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        return float(log_loss(y, sigmoid(logits / temperature), labels=[0, 1]))

    optimum = minimize_scalar(objective, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    result["temperature"] = Calibrator("temperature", {"temperature": float(np.exp(optimum.x))})
    platt = LogisticRegression(C=1e8, solver="lbfgs", max_iter=2000)
    platt.fit(logits.reshape(-1, 1), y)
    slope = float(platt.coef_[0, 0])
    if slope <= 0:
        slope = 1.0
        intercept = 0.0
    else:
        intercept = float(platt.intercept_[0])
    result["platt"] = Calibrator("platt", {"slope": slope, "intercept": intercept})
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(logits, y)
    result["isotonic"] = Calibrator(
        "isotonic",
        {
            "x_thresholds": isotonic.X_thresholds_.astype(float).tolist(),
            "y_thresholds": isotonic.y_thresholds_.astype(float).tolist(),
        },
    )
    return result


def select_primary_calibrator(calibrators: dict[str, Calibrator], logits: np.ndarray, y: np.ndarray) -> tuple[str, dict[str, float]]:
    losses = {
        method: float(log_loss(y, calibrator.predict(logits), labels=[0, 1]))
        for method, calibrator in calibrators.items()
        if method in {"none", "temperature", "platt"}
    }
    selected = min(losses, key=losses.get)
    return selected, losses
