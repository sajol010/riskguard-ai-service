import json
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
)

from preprocessing import build_pipeline, get_all_feature_columns
from config import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

MODEL_NAME = "order_risk"
TARGET_COL = "is_fraud"


def train(data_path: str):
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    feature_cols = get_all_feature_columns(MODEL_NAME)
    X = df[feature_cols]
    y = df[TARGET_COL]

    logger.info(f"Dataset: {len(df)} rows, positive rate: {y.mean():.4f}")

    # 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    pipeline = build_pipeline(MODEL_NAME)
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_proba)
    logger.info(f"Validation AUC: {val_auc:.4f}")

    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba)
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="binary"
    )

    logger.info(f"Test AUC: {test_auc:.4f}")
    logger.info(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logger.info("\n" + classification_report(y_test, y_test_pred))

    # Save
    model_dir = settings.MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = model_dir / f"{MODEL_NAME}_pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
    logger.info(f"Pipeline saved to {pipeline_path}")

    (model_dir / f"{MODEL_NAME}_version.txt").write_text(settings.MODEL_VERSION)
    (model_dir / f"{MODEL_NAME}_training_date.txt").write_text(
        datetime.now(timezone.utc).isoformat()
    )

    metrics = {
        "model": MODEL_NAME,
        "version": settings.MODEL_VERSION,
        "dataset_size": len(df),
        "positive_rate": float(y.mean()),
        "validation_auc": float(val_auc),
        "test_auc": float(test_auc),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
    }
    (model_dir / f"{MODEL_NAME}_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    logger.info(f"Training complete for {MODEL_NAME}.")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/order_fraud_data.csv")
    args = parser.parse_args()
    train(args.data)
