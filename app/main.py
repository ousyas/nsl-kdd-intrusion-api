from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, status

from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ConnectionFeatures,
    ModelInfoResponse,
    PredictionResponse,
)
from .service import IntrusionDetectionService


DEFAULT_MODEL_PATH = Path("models/nsl_kdd_xgb.joblib")


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    try:
        app.state.predictor = IntrusionDetectionService(model_path)
        app.state.load_error = None
    except (FileNotFoundError, ValueError) as exc:
        app.state.predictor = None
        app.state.load_error = str(exc)
    yield


app = FastAPI(
    title="NSL-KDD Intrusion Detection API",
    description="Binary network intrusion detection using a validated XGBoost pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_predictor(request: Request) -> IntrusionDetectionService:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=request.app.state.load_error or "Model is not available",
        )
    return predictor


@app.get("/health")
def health(request: Request, response: Response) -> dict:
    predictor = request.app.state.predictor
    if predictor is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if predictor is not None else "not_ready",
        "model_version": predictor.model_version if predictor is not None else None,
    }


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    predictor = get_predictor(request)
    return ModelInfoResponse(
        model_version=predictor.model_version,
        trained_at=predictor.trained_at,
        threshold=predictor.threshold,
        validation_metrics=predictor.validation_metrics,
        test_metrics=predictor.test_metrics,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(record: ConnectionFeatures, request: Request) -> PredictionResponse:
    predictor = get_predictor(request)
    return predictor.predict([record])[0]


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(
    payload: BatchPredictionRequest, request: Request
) -> BatchPredictionResponse:
    predictor = get_predictor(request)
    return BatchPredictionResponse(predictions=predictor.predict(payload.records))
