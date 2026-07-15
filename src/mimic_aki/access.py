from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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


def verify_mimic_access(root: str | None = None) -> MimicAccessStatus:
    root = root or os.environ.get("MIMIC_IV_ROOT")
    if not root:
        return MimicAccessStatus("BLOCKED_DATA_ACCESS", None, REQUIRED_RELATIVE_FILES)
    base = Path(root)
    missing = [rel for rel in REQUIRED_RELATIVE_FILES if not (base / rel).exists()]
    return MimicAccessStatus("OK" if not missing else "BLOCKED_DATA_ACCESS", str(base), missing)
