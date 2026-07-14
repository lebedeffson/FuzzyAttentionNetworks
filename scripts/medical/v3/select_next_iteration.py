from __future__ import annotations


def select_next_registered_config(diagnosis: dict) -> dict | None:
    if diagnosis.get("terminal"):
        return None
    return diagnosis.get("next_config")


__all__ = ["select_next_registered_config"]
