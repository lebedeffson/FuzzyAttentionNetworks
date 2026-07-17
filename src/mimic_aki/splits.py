from __future__ import annotations

import numpy as np
import pandas as pd


def split_subjects(subject_ids, seed: int = 20260715, train: float = 0.70, validation: float = 0.15) -> pd.DataFrame:
    subjects = np.asarray(sorted(set(map(int, subject_ids))))
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(round(train * n))
    n_val = int(round(validation * n))
    rows = []
    for idx, subject_id in enumerate(subjects):
        split = "train" if idx < n_train else "validation" if idx < n_train + n_val else "test"
        rows.append({"subject_id": int(subject_id), "split": split})
    return pd.DataFrame(rows)
