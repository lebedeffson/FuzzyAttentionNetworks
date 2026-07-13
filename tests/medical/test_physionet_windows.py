import numpy as np

from src.med_circuitbench.data.windows import NormalizationStats, assert_patient_disjoint, delta_time_channels, normalize_with_masks


def test_patient_splits_must_be_disjoint():
    assert_patient_disjoint({"train": ["a", "b"], "validation": ["c"], "test": ["d"]})
    try:
        assert_patient_disjoint({"train": ["a"], "test": ["a"]})
    except ValueError:
        return
    raise AssertionError("expected overlap to fail")


def test_no_nan_inf_after_normalization():
    values = np.asarray([[[1.0, np.nan], [2.0, 4.0]]])
    masks = np.asarray([[[1.0, 0.0], [1.0, 1.0]]])
    values = np.nan_to_num(values, nan=0.0)
    out = normalize_with_masks(values, masks, NormalizationStats(mean=np.asarray([1.0, 4.0]), std=np.asarray([1.0, 2.0])))
    assert np.isfinite(out).all()
    assert out[0, 0, 1] == 0.0


def test_delta_time_channels():
    masks = np.asarray([[[0.0], [0.0], [1.0], [0.0]]])
    out = delta_time_channels(masks)
    assert np.allclose(out[:, :, 0], [[np.log1p(36), np.log1p(36), 0.0, np.log1p(1)]])
