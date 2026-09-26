from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from app.main import app
from src.data import FEATURE_COLUMNS


@pytest.fixture
def connection() -> dict:
    record = {column: 0 for column in FEATURE_COLUMNS}
    record.update({"protocol_type": "tcp", "service": "http", "flag": "SF"})
    return record


@pytest.fixture
def model_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, connection: dict) -> Path:
    # A tiny deterministic model exercises the HTTP contract without training NSL-KDD.
    frame = pd.DataFrame([connection, connection])[FEATURE_COLUMNS]
    classifier = DummyClassifier(strategy="prior").fit(frame, [0, 1])
    path = tmp_path / "test_model.joblib"
    joblib.dump(
        {
            "pipeline": classifier,
            "threshold": 0.4,
            "feature_columns": FEATURE_COLUMNS,
            "model_version": "test",
            "trained_at": "2026-01-01T00:00:00+00:00",
            "validation_metrics": {},
            "test_metrics": {},
        },
        path,
    )
    monkeypatch.setenv("MODEL_PATH", str(path))
    return path


def test_health_and_model_info(model_path: Path) -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        info = client.get("/model-info")

    assert health.status_code == 200
    assert health.json() == {"status": "ready", "model_version": "test"}
    assert info.status_code == 200
    assert info.json()["threshold"] == 0.4


def test_single_and_batch_prediction_match(model_path: Path, connection: dict) -> None:
    with TestClient(app) as client:
        single = client.post("/predict", json=connection)
        batch = client.post("/predict-batch", json={"records": [connection, connection]})

    assert single.status_code == 200
    assert single.json()["prediction"] == "attack"
    assert single.json()["attack_probability"] == 0.5
    assert batch.status_code == 200
    assert batch.json()["predictions"] == [single.json(), single.json()]


@pytest.mark.parametrize("invalid_field", ["missing", "negative", "extra"])
def test_invalid_connection_returns_422(
    model_path: Path, connection: dict, invalid_field: str
) -> None:
    if invalid_field == "missing":
        connection.pop("duration")
    elif invalid_field == "negative":
        connection["duration"] = -1
    else:
        connection["unexpected"] = 1

    with TestClient(app) as client:
        response = client.post("/predict", json=connection)

    assert response.status_code == 422


def test_missing_model_returns_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, connection: dict
) -> None:
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))

    with TestClient(app) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json=connection)

    assert health.status_code == 503
    assert health.json()["status"] == "not_ready"
    assert prediction.status_code == 503
