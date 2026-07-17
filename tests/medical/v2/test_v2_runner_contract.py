from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_v2_runner_dry_run(tmp_path: Path):
    cmd = [
        sys.executable,
        "scripts/medical/v2/run_v2_program.py",
        "--config",
        "configs/medical/v2/program.yaml",
        "--seeds",
        "42",
        "--mode",
        "full",
        "--output",
        str(tmp_path / "v2"),
        "--dry-run",
    ]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert "med_circuitbench_v2" in (tmp_path / "v2" / "MANIFESTS" / "preflight.json").read_text()

