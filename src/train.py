from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .data import FEATURE_COLUMNS, load_nsl_kdd
from .pipeline import build_pipeline


def select_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose a threshold on validation data by maximizing attack-class F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1_values))])


def evaluate(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype("int8")
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_attack": float(precision_score(y_true, predictions)),
        "recall_attack": float(recall_score(y_true, predictions)),
        "f1_attack": float(f1_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def train(data_dir: Path, output_path: Path, metrics_path: Path) -> dict:
    X_train_full, y_train_full = load_nsl_kdd(data_dir / "KDDTrain+.txt")
    X_test, y_test = load_nsl_kdd(data_dir / "KDDTest+.txt")

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full,
    )

    pipeline = build_pipeline(random_state=42)
    pipeline.fit(X_train, y_train)

    validation_probabilities = pipeline.predict_proba(X_validation)[:, 1]
    threshold = select_f1_threshold(y_validation.to_numpy(), validation_probabilities)
    validation_metrics = evaluate(
        y_validation.to_numpy(), validation_probabilities, threshold
    )

    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test.to_numpy(), test_probabilities, threshold)

    trained_at = datetime.now(UTC).isoformat()
    bundle = {
        "pipeline": pipeline,
        "threshold": threshold,
        "feature_columns": FEATURE_COLUMNS,
        "model_version": "1.0.0",
        "trained_at": trained_at,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)

    report = {
        "model_version": bundle["model_version"],
        "trained_at": trained_at,
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_validation)),
        "test_samples": int(len(X_test)),
        "validation": validation_metrics,
        "test": test_metrics,
    }
    metrics_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NSL-KDD XGBoost pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path("models/nsl_kdd_xgb.joblib")
    )
    parser.add_argument(
        "--metrics", type=Path, default=Path("models/metrics.json")
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train(args.data_dir, args.output, args.metrics)
    print(json.dumps(result, indent=2, ensure_ascii=False))

