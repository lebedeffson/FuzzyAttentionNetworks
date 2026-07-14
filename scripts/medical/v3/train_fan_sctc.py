from __future__ import annotations


def fan_sctc_status(fan_gate_passed: bool) -> dict:
    if fan_gate_passed:
        return {"stage_status": "COMPLETED"}
    return {"stage_status": "SKIPPED_BY_GATE", "reason": "FAN Foundation Gate failed"}


__all__ = ["fan_sctc_status"]
