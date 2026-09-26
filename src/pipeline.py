from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import (
    CATEGORICAL_COLUMNS,
    ENGINEERED_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
)


class NetworkFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add the five engineered features used in the original notebook."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "NetworkFeatureEngineer":
        self._validate(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate(X)
        result = X.copy()

        src = pd.to_numeric(result["src_bytes"], errors="coerce").fillna(0.0)
        dst = pd.to_numeric(result["dst_bytes"], errors="coerce").fillna(0.0)
        duration = pd.to_numeric(result["duration"], errors="coerce").fillna(0.0)
        serror = pd.to_numeric(result["serror_rate"], errors="coerce").fillna(0.0)
        rerror = pd.to_numeric(result["rerror_rate"], errors="coerce").fillna(0.0)

        result["bytes_ratio"] = src / (dst + 1.0)
        result["error_ratio"] = serror / (rerror + 0.001)
        result["log_src_bytes"] = np.log1p(np.clip(src, 0.0, None))
        result["log_dst_bytes"] = np.log1p(np.clip(dst, 0.0, None))
        result["log_duration"] = np.log1p(np.clip(duration, 0.0, None))
        return result

    @staticmethod
    def _validate(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("NetworkFeatureEngineer expects a pandas DataFrame")
        missing = sorted(set(FEATURE_COLUMNS) - set(X.columns))
        if missing:
            raise ValueError(f"Missing required NSL-KDD features: {missing}")


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Create a leak-free preprocessing and XGBoost classification pipeline."""
    numeric_with_engineering = [*NUMERIC_COLUMNS, *ENGINEERED_COLUMNS]
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_COLUMNS,
            ),
            ("numeric", StandardScaler(), numeric_with_engineering),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    classifier = XGBClassifier(
        n_estimators=180,
        learning_rate=0.03,
        max_depth=3,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("feature_engineering", NetworkFeatureEngineer()),
            ("preprocessing", preprocessing),
            ("variance_filter", VarianceThreshold()),
            ("feature_selection", SelectKBest(score_func=f_classif, k=28)),
            ("classifier", classifier),
        ]
    )

