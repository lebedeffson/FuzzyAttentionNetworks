from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .io import MimicSource


REQUIRED_RELATIVE_FILES = [
    "hosp/patients.csv.gz",
    "hosp/admissions.csv.gz",
    "hosp/labevents.csv.gz",
    "icu/icustays.csv.gz",
    "icu/chartevents.csv.gz",
]


@dataclass(frozen=True)
class MimicAccessStatus:
    status: str
    root: str | None
    missing_files: list[str]
    dataset_kind: str = "unknown"


def verify_mimic_access(root: str | None = None) -> MimicAccessStatus:
    root = root or os.environ.get("MIMIC_IV_ROOT") or os.environ.get("MIMIC_IV_DEMO_ZIP")
    if not root and Path("mimic-iv-clinical-database-demo-2.2.zip").exists():
        root = "mimic-iv-clinical-database-demo-2.2.zip"
    if not root:
        return MimicAccessStatus("BLOCKED_DATA_ACCESS", None, REQUIRED_RELATIVE_FILES)
    source = MimicSource.open(root)
    missing = [rel for rel in REQUIRED_RELATIVE_FILES if not source.exists(rel)]
    status = "OK_DEMO" if source.kind == "demo_zip" and not missing else "OK" if not missing else "BLOCKED_DATA_ACCESS"
    return MimicAccessStatus(status, str(Path(root)), missing, source.kind)
