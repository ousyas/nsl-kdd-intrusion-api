from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data import FEATURE_COLUMNS

from .schemas import ConnectionFeatures, PredictionResponse


class IntrusionDetectionService:
    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Trained model not found: {path}")

        bundle = joblib.load(path)
        required_keys = {
            "pipeline",
            "threshold",
            "feature_columns",
            "model_version",
            "trained_at",
            "validation_metrics",
            "test_metrics",
        }
        missing = required_keys - set(bundle)
        if missing:
            raise ValueError(f"Invalid model bundle, missing keys: {sorted(missing)}")
        if list(bundle["feature_columns"]) != FEATURE_COLUMNS:
            raise ValueError("Model feature schema does not match the API schema")

        self.pipeline = bundle["pipeline"]
        self.threshold = float(bundle["threshold"])
        self.model_version = str(bundle["model_version"])
        self.trained_at = str(bundle["trained_at"])
        self.validation_metrics = dict(bundle["validation_metrics"])
        self.test_metrics = dict(bundle["test_metrics"])

    def predict(self, records: list[ConnectionFeatures]) -> list[PredictionResponse]:
        frame = pd.DataFrame([record.model_dump() for record in records])
        frame = frame[FEATURE_COLUMNS]
        probabilities = self.pipeline.predict_proba(frame)[:, 1]
        return [
            PredictionResponse(
                prediction="attack" if probability >= self.threshold else "normal",
                attack_probability=round(float(probability), 6),
                threshold=round(self.threshold, 6),
                model_version=self.model_version,
            )
            for probability in probabilities
        ]

