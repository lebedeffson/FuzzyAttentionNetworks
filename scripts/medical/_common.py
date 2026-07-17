from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def add_config_arg() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    add_run_context_args(parser)
    return parser


def add_run_context_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", choices=["validation", "test"], default=None)
    return parser


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_manifest(run_dir: Path, config_path: Path, payload: Dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "config_path": str(config_path),
        **payload,
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def write_environment(path: Path) -> None:
    try:
        import numpy as np
        import pandas as pd
        import torch
        import sklearn
    except Exception:  # pragma: no cover - best effort diagnostic
        np = pd = torch = sklearn = None
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": getattr(np, "__version__", None),
        "pandas": getattr(pd, "__version__", None),
        "torch": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()) if torch is not None else None,
        "cuda_version": getattr(torch.version, "cuda", None) if torch is not None else None,
        "sklearn": getattr(sklearn, "__version__", None),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
