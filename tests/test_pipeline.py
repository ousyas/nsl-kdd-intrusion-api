from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import FEATURE_COLUMNS
from src.pipeline import NetworkFeatureEngineer
from src.train import select_f1_threshold


def test_feature_engineering_preserves_schema_and_adds_features() -> None:
    row = {column: 0 for column in FEATURE_COLUMNS}
    row.update(
        {
            "protocol_type": "tcp",
            "service": "http",
            "flag": "SF",
            "src_bytes": 100,
            "dst_bytes": 49,
            "duration": 9,
            "serror_rate": 0.2,
            "rerror_rate": 0.1,
        }
    )
    transformed = NetworkFeatureEngineer().fit_transform(pd.DataFrame([row]))

    assert transformed.loc[0, "bytes_ratio"] == 2.0
    assert transformed.loc[0, "error_ratio"] > 1.9
    assert transformed.loc[0, "log_src_bytes"] > 0
    assert set(FEATURE_COLUMNS).issubset(transformed.columns)


def test_threshold_is_selected_from_validation_scores() -> None:
    labels = np.array([0, 1, 1])
    scores = np.array([0.1, 0.4, 0.9])

    assert select_f1_threshold(labels, scores) == 0.4
