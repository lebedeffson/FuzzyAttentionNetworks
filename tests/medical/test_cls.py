import json

import numpy as np

from src.med_circuitbench.sctc.layer_screening import compute_cls


def test_cls_fixture_values():
    expected = json.load(open("tests/medical/fixtures/test_cls_expected.json"))
    normalized = np.asarray(
        [
            [0.0, 1.0, 1.0, 0.96666667],
            [1.0, 0.0, 0.94166667, 1.0],
            [0.89285714, 1.0, 1.0, 0.0],
            [0.94285714, 1.0, 0.0, 1.0],
        ]
    )
    auc = ((normalized[:, 0] + 1.0) / 2.0)[:, None]
    result = compute_cls(auc, normalized[:, 1], normalized[:, 2], normalized[:, 3])
    assert np.allclose(result.cls, expected["CLS"], atol=1e-7)
    assert result.ranking_report == expected["ranking"]
    assert result.selected_report == expected["selected"]
