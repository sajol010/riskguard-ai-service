# AI Refund, Fraud & Return Intelligence Platform

## Full Technical Document

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Configuration](#4-configuration)
5. [Data Pipeline](#5-data-pipeline)
6. [Feature Engineering](#6-feature-engineering)
7. [ML Models](#7-ml-models)
8. [Preprocessing Pipeline](#8-preprocessing-pipeline)
9. [Training Scripts](#9-training-scripts)
10. [FastAPI Inference Service](#10-fastapi-inference-service)
11. [API Reference](#11-api-reference)
12. [Authentication & Security](#12-authentication--security)
13. [Laravel Integration Guide](#13-laravel-integration-guide)
14. [Model Retraining](#14-model-retraining)
15. [Logging & Monitoring](#15-logging--monitoring)
16. [Deployment](#16-deployment)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. System Overview

### Purpose

The ML system provides three binary classification models that predict:

| Model | Prediction | Target Column | Output Key |
|-------|-----------|---------------|------------|
| Order Fraud Risk | Probability an order is fraudulent | `is_fraud` | `risk_score` |
| Refund Abuse Risk | Probability a refund request is abusive | `is_abuse` | `abuse_score` |
| Return Probability | Probability an order will be returned | `is_returned` | `return_score` |

Each model outputs a probability score between `0.0` and `1.0`. The score is consumed by a Laravel backend decision engine.

### Design Principles

- **Stateless**: No UI, no session state. Pure API service.
- **API-Callable**: All predictions via REST endpoints.
- **Disk-Loaded Models**: Trained pipelines serialized as `.pkl` files, loaded at startup.
- **Low Latency**: Inference completes in <5ms (well under the 500ms requirement).
- **No PII in Logs**: Only order amounts and scores are logged.

### Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Web Framework | FastAPI | 0.128.8 |
| ASGI Server | Uvicorn | 0.39.0 |
| ML Framework | scikit-learn | >=1.3, <1.6 |
| Gradient Boosting | XGBoost | >=1.7, <2.3 |
| Data Processing | pandas | >=1.5, <2.3 |
| Numerical Computing | numpy | >=1.24, <2.0 |
| Model Serialization | joblib | >=1.3 |
| Validation | Pydantic v2 | 2.12.5 |
| Configuration | python-dotenv | >=1.0 |
| Runtime | Python | 3.9+ |

---

## 2. Architecture

### High-Level Flow

```
PostgreSQL (Production Data)
        |
        v
[ generate_data.py ]  ← (dev: synthetic)  OR  [ SQL Export ] ← (prod: real data)
        |
        v
    CSV files in data/
        |
        v
[ train_order_model.py  ]
[ train_refund_model.py ]  →  .pkl pipeline files in models/
[ train_return_model.py ]
        |
        v
[ FastAPI Service (app.py) ]  ← loads .pkl at startup
        |
        v
[ Laravel Backend ]  ← calls API, receives scores, applies thresholds
```

### Two Operational Modes

| Mode | Purpose | When |
|------|---------|------|
| **Training** (offline) | Build/retrain models from data | Weekly/monthly, or on-demand |
| **Prediction** (online) | Serve predictions via API | Always running |

### Request Flow (Prediction Mode)

```
Laravel → HTTP POST with JSON body + Bearer token
    → FastAPI validates token (HTTPBearer)
    → FastAPI validates request body (Pydantic schema)
    → Converts to single-row DataFrame
    → sklearn Pipeline: preprocess → XGBClassifier.predict_proba
    → Returns probability score as JSON
    → Logs: model name, score, latency, order_amount (no PII)
```

---

## 3. Project Structure

```
ai-refund-return-intelligence/
├── .gitignore                          # Ignores venv, .env, .pkl, .csv
├── fastapi-env/                        # Python 3.9 virtual environment
│
└── ai-service/                         # Main application directory
    ├── app.py                          # FastAPI inference service (4 endpoints)
    ├── config.py                       # Settings loaded from .env
    ├── schemas.py                      # Pydantic request/response models
    ├── preprocessing.py                # Shared feature definitions + pipeline builders
    ├── generate_data.py                # Synthetic data generator (dev/testing)
    ├── train_order_model.py            # Training script: order fraud model
    ├── train_refund_model.py           # Training script: refund abuse model
    ├── train_return_model.py           # Training script: return probability model
    ├── requirements.txt                # Python dependencies
    ├── .env                            # Environment variables (gitignored)
    ├── .env.example                    # Template for .env
    │
    ├── models/                         # Trained model artifacts (gitignored)
    │   ├── order_risk_pipeline.pkl     # Serialized order fraud pipeline
    │   ├── refund_risk_pipeline.pkl    # Serialized refund abuse pipeline
    │   ├── return_risk_pipeline.pkl    # Serialized return probability pipeline
    │   ├── order_risk_metrics.json     # Training metrics
    │   ├── refund_risk_metrics.json
    │   ├── return_risk_metrics.json
    │   ├── order_risk_version.txt      # Model version string
    │   ├── refund_risk_version.txt
    │   ├── return_risk_version.txt
    │   ├── order_risk_training_date.txt # ISO timestamp of training
    │   ├── refund_risk_training_date.txt
    │   └── return_risk_training_date.txt
    │
    └── data/                           # Training data CSVs (gitignored)
        ├── order_fraud_data.csv
        ├── refund_abuse_data.csv
        └── return_data.csv
```

### File Responsibilities

| File | Role | Dependencies |
|------|------|-------------|
| `config.py` | Loads `.env`, exposes `settings` singleton | python-dotenv |
| `schemas.py` | Pydantic v2 request/response models with validation | pydantic |
| `preprocessing.py` | Feature column constants, `build_pipeline()`, `build_preprocessor()` | scikit-learn, xgboost |
| `generate_data.py` | CLI tool to create synthetic training CSVs | numpy, pandas |
| `train_*.py` | Train a model, evaluate metrics, save `.pkl` + metadata | preprocessing, config |
| `app.py` | FastAPI app with auth, prediction endpoints, health check | all above |

---

## 4. Configuration

### Environment Variables

Configuration is managed via `.env` file in the `ai-service/` directory, loaded by `config.py`.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `API_TOKEN` | string | *(required)* | Bearer token for authenticating requests from Laravel |
| `MODEL_DIR` | string | `models` | Directory containing `.pkl` model files (relative to ai-service/) |
| `LOG_LEVEL` | string | `INFO` | Python logging level (DEBUG, INFO, WARNING, ERROR) |
| `MODEL_VERSION` | string | `0.1.0` | Version string returned in every prediction response |

### .env.example

```env
API_TOKEN=changeme-generate-a-real-token
MODEL_DIR=models
LOG_LEVEL=INFO
MODEL_VERSION=0.1.0
```

### config.py Implementation

```python
import os
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

class Settings:
    def __init__(self):
        self.API_TOKEN: str = os.getenv("API_TOKEN", "")
        self.MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", "models"))
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.MODEL_VERSION: str = os.getenv("MODEL_VERSION", "0.1.0")

        if not self.API_TOKEN:
            raise ValueError("API_TOKEN must be set in environment or .env file")

        if not self.MODEL_DIR.is_absolute():
            self.MODEL_DIR = Path(__file__).resolve().parent / self.MODEL_DIR

settings = Settings()
```

The `Settings` class:
- Fails fast on startup if `API_TOKEN` is missing
- Resolves `MODEL_DIR` to an absolute path relative to `ai-service/`
- Is instantiated once as a module-level singleton

---

## 5. Data Pipeline

### Production Data Source

In production, data comes from PostgreSQL tables:

```sql
SELECT
    o.amount AS order_amount,
    o.currency,
    o.billing_country,
    o.shipping_country,
    (o.billing_country != o.shipping_country) AS country_mismatch,
    c.total_orders AS customer_total_orders,
    c.refund_rate AS customer_refund_rate,
    c.dispute_rate AS customer_dispute_rate,
    o.device_reuse_count,
    o.ip_reuse_count,
    o.order_velocity_24h,
    r.reason AS refund_reason,
    EXTRACT(DAY FROM r.created_at - o.created_at) AS days_since_purchase,
    o.delivery_confirmed,
    o.item_category,
    o.is_fraud,
    r.is_abuse,
    o.is_returned
FROM orders o
LEFT JOIN refunds r ON r.order_id = o.id
JOIN customers c ON c.id = o.customer_id;
```

Export to CSV and place in `ai-service/data/`.

### Synthetic Data Generator (Development)

For development and testing, `generate_data.py` creates realistic synthetic data.

**Usage:**
```bash
python generate_data.py --samples 10000 --fraud-rate 0.05 --abuse-rate 0.08 --return-rate 0.15 --seed 42
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--samples` | 10000 | Number of rows per dataset |
| `--fraud-rate` | 0.05 | Target fraud positive rate (~5%) |
| `--abuse-rate` | 0.08 | Target abuse positive rate (~8%) |
| `--return-rate` | 0.15 | Target return positive rate (~15%) |
| `--seed` | 42 | Random seed for reproducibility |

**Outputs:**
- `data/order_fraud_data.csv` (11 features + `is_fraud`)
- `data/refund_abuse_data.csv` (15 features + `is_abuse`)
- `data/return_data.csv` (11 features + `is_returned`)

**Statistical Distributions Used:**

| Feature | Distribution | Parameters | Rationale |
|---------|-------------|------------|-----------|
| `order_amount` | Log-normal | mean=4.0, sigma=1.0 | Right-skewed: most orders small, some large |
| `customer_refund_rate` | Beta | alpha=2, beta=20 | Clustered near 0, long tail (most customers rarely refund) |
| `customer_dispute_rate` | Beta | alpha=1, beta=50 | Very low values (disputes are rare) |
| `customer_total_orders` | Poisson | lambda=8 | Discrete count, average ~8 orders |
| `device_reuse_count` | Poisson | lambda=2 | Low device reuse on average |
| `ip_reuse_count` | Poisson | lambda=1 | Low IP reuse |
| `order_velocity_24h` | Poisson | lambda=1.5 | ~1-2 orders per 24h |
| `days_since_purchase` | Poisson | lambda=14 | Average ~2 weeks |
| `country_mismatch` | Derived | billing != shipping | Correlated with countries |

**Label Generation (Correlated with Features):**

Fraud labels are correlated with:
- Country mismatch (weight: 0.30)
- High order velocity (weight: 0.20)
- High order amount (weight: 0.15)
- Device reuse (weight: 0.15)
- IP reuse (weight: 0.10)
- Dispute rate (weight: 0.10)

Abuse labels are correlated with:
- High refund rate (weight: 0.25)
- "not_as_described" reason (weight: 0.20)
- Delivery not confirmed (weight: 0.20)
- Days since purchase > 25 (weight: 0.15)
- High order amount (weight: 0.10)
- High dispute rate (weight: 0.10)

Return labels are correlated with:
- High order amount (weight: 0.25)
- Country mismatch (weight: 0.20)
- High refund rate (weight: 0.20)
- High velocity (weight: 0.15)
- Low total orders (weight: 0.20)

---

## 6. Feature Engineering

### Common Features (All 3 Models)

| Feature | Type | Preprocessing | Description |
|---------|------|--------------|-------------|
| `order_amount` | float | Median impute → StandardScaler | Order total in currency units |
| `customer_total_orders` | int | Median impute → StandardScaler | Lifetime order count |
| `customer_refund_rate` | float | Median impute → StandardScaler | Ratio of refunded orders (0.0-1.0) |
| `customer_dispute_rate` | float | Median impute → StandardScaler | Ratio of disputed orders (0.0-1.0) |
| `device_reuse_count` | int | Median impute → StandardScaler | Times this device ID was seen |
| `ip_reuse_count` | int | Median impute → StandardScaler | Times this IP was seen |
| `order_velocity_24h` | int | Median impute → StandardScaler | Orders from this customer in last 24h |
| `currency` | categorical | "unknown" impute → OneHotEncoder | 3-letter currency code (USD, EUR, etc.) |
| `billing_country` | categorical | "unknown" impute → OneHotEncoder | 2-letter billing country code |
| `shipping_country` | categorical | "unknown" impute → OneHotEncoder | 2-letter shipping country code |
| `country_mismatch` | boolean | Passthrough | billing_country != shipping_country |

### Refund Model Extra Features (Refund Abuse Model Only)

| Feature | Type | Preprocessing | Description |
|---------|------|--------------|-------------|
| `refund_reason` | categorical | "unknown" impute → OneHotEncoder | Reason for refund request |
| `item_category` | categorical | "unknown" impute → OneHotEncoder | Product category |
| `days_since_purchase` | int | Median impute → StandardScaler | Days between order and refund request |
| `delivery_confirmed` | boolean | Passthrough | Whether delivery was confirmed |

### Feature Sets by Model

| Model | Numeric (7+) | Categorical (3+) | Boolean (1+) | Total |
|-------|-------------|------------------|-------------|-------|
| Order Risk | 7 common | 3 common | 1 common | **11** |
| Refund Risk | 7 common + 1 extra | 3 common + 2 extra | 1 common + 1 extra | **15** |
| Return Risk | 7 common | 3 common | 1 common | **11** |

### Feature Column Definitions (preprocessing.py)

```python
NUMERIC_COMMON = [
    "order_amount", "customer_total_orders", "customer_refund_rate",
    "customer_dispute_rate", "device_reuse_count", "ip_reuse_count",
    "order_velocity_24h",
]

CATEGORICAL_COMMON = ["currency", "billing_country", "shipping_country"]

BOOLEAN_COMMON = ["country_mismatch"]

# Refund extras
NUMERIC_REFUND_EXTRA = ["days_since_purchase"]
CATEGORICAL_REFUND_EXTRA = ["refund_reason", "item_category"]
BOOLEAN_REFUND_EXTRA = ["delivery_confirmed"]

FEATURE_SETS = {
    "order_risk":  {"numeric": NUMERIC_COMMON, "categorical": CATEGORICAL_COMMON, "boolean": BOOLEAN_COMMON},
    "refund_risk": {"numeric": NUMERIC_COMMON + NUMERIC_REFUND_EXTRA, "categorical": CATEGORICAL_COMMON + CATEGORICAL_REFUND_EXTRA, "boolean": BOOLEAN_COMMON + BOOLEAN_REFUND_EXTRA},
    "return_risk": {"numeric": NUMERIC_COMMON, "categorical": CATEGORICAL_COMMON, "boolean": BOOLEAN_COMMON},
}
```

---

## 7. ML Models

### Model Selection: XGBClassifier

All three models use `XGBClassifier` (XGBoost gradient-boosted decision trees) wrapped in an sklearn `Pipeline`.

**Why XGBoost:**
- Handles mixed data types (numeric + encoded categorical)
- Explainable via feature importance
- Fast inference (<5ms per prediction)
- Handles class imbalance reasonably well
- No deep learning infrastructure required

### Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_estimators` | 200 | Number of boosting rounds |
| `max_depth` | 6 | Maximum tree depth |
| `learning_rate` | 0.1 | Step size shrinkage |
| `eval_metric` | logloss | Evaluation metric for binary classification |
| `random_state` | 42 | Reproducibility seed |

These can be overridden per-model by passing `xgb_params` to `build_pipeline()`:
```python
pipeline = build_pipeline("order_risk", xgb_params={"n_estimators": 500, "max_depth": 8})
```

### Pipeline Architecture

Each model is a single sklearn `Pipeline` object containing:

```
Pipeline
├── Step 1: "preprocessor" (ColumnTransformer)
│   ├── "num": SimpleImputer(median) → StandardScaler       [numeric columns]
│   ├── "cat": SimpleImputer("unknown") → OneHotEncoder     [categorical columns]
│   └── "bool": passthrough                                  [boolean columns]
│
└── Step 2: "classifier" (XGBClassifier)
```

**Key design: The entire pipeline is serialized to a single `.pkl` file.** This means:
- At inference time, one call to `pipeline.predict_proba(df)` handles all preprocessing + prediction
- No train/serve skew is possible — the exact same transformations are applied
- `OneHotEncoder(handle_unknown="ignore")` ensures unseen categories at inference produce all-zeros (safe degradation)
- `sparse_output=False` avoids sparse matrix conversion overhead

---

## 8. Preprocessing Pipeline

### Preprocessing Steps

**Step 1: Missing Value Imputation**

| Feature Type | Strategy | Rationale |
|-------------|----------|-----------|
| Numeric | Median imputation | Robust to outliers (unlike mean) |
| Categorical | Constant "unknown" | Preserves information that the value was missing |
| Boolean | Passthrough | Boolean fields are required in the API schema |

**Step 2: Feature Transformation**

| Feature Type | Transformer | Rationale |
|-------------|------------|-----------|
| Numeric | StandardScaler (zero mean, unit variance) | XGBoost doesn't strictly need scaling, but it helps with regularization and interpretability |
| Categorical | OneHotEncoder | Converts categories to binary columns. `handle_unknown="ignore"` prevents crashes on unseen categories |
| Boolean | Passthrough | Already binary (0/1). Scaling would be counterproductive |

**Step 3: Data Split**

| Split | Percentage | Usage |
|-------|-----------|-------|
| Train | 70% | Model fitting |
| Validation | 15% | AUC evaluation during development |
| Test | 15% | Final metrics (precision, recall, F1, AUC) |

Split is **stratified** to preserve the positive label ratio in each split.

### build_preprocessor() Function

```python
def build_preprocessor(feature_set: str) -> ColumnTransformer:
    fs = FEATURE_SETS[feature_set]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, fs["numeric"]),
        ("cat", categorical_transformer, fs["categorical"]),
        ("bool", "passthrough", fs["boolean"]),
    ])
```

### build_pipeline() Function

```python
def build_pipeline(feature_set: str, xgb_params: dict = None) -> Pipeline:
    default_params = {
        "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
        "eval_metric": "logloss", "random_state": 42,
    }
    if xgb_params:
        default_params.update(xgb_params)

    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(feature_set)),
        ("classifier", XGBClassifier(**default_params)),
    ])
```

---

## 9. Training Scripts

### Common Training Flow

All three training scripts follow identical logic:

```
1. Load CSV from data/
2. Extract features (X) and target (y) using get_all_feature_columns()
3. Split: 70% train / 15% validation / 15% test (stratified)
4. Build pipeline via build_pipeline(MODEL_NAME)
5. Fit pipeline on training data
6. Evaluate on validation set (AUC)
7. Evaluate on test set (AUC, precision, recall, F1, classification report)
8. Save to models/:
   - {MODEL_NAME}_pipeline.pkl      (serialized pipeline)
   - {MODEL_NAME}_version.txt       (version string)
   - {MODEL_NAME}_training_date.txt (UTC ISO timestamp)
   - {MODEL_NAME}_metrics.json      (training metrics)
```

### Training Scripts Summary

| Script | MODEL_NAME | TARGET_COL | Default Data |
|--------|-----------|-----------|-------------|
| `train_order_model.py` | `order_risk` | `is_fraud` | `data/order_fraud_data.csv` |
| `train_refund_model.py` | `refund_risk` | `is_abuse` | `data/refund_abuse_data.csv` |
| `train_return_model.py` | `return_risk` | `is_returned` | `data/return_data.csv` |

### Usage

```bash
# Default data path
python train_order_model.py

# Custom data path
python train_order_model.py --data /path/to/custom_data.csv
```

### Output Artifacts

Each training run produces 4 files in `models/`:

**1. Pipeline file (`*_pipeline.pkl`)**
- Complete sklearn Pipeline (preprocessor + classifier)
- Loaded by `app.py` at startup via `joblib.load()`

**2. Version file (`*_version.txt`)**
- Contains the `MODEL_VERSION` from `.env` (e.g., `0.1.0`)

**3. Training date (`*_training_date.txt`)**
- UTC ISO 8601 timestamp (e.g., `2026-02-26T05:18:50.123456+00:00`)

**4. Metrics file (`*_metrics.json`)**
```json
{
  "model": "order_risk",
  "version": "0.1.0",
  "dataset_size": 10000,
  "positive_rate": 0.0393,
  "validation_auc": 0.5835,
  "test_auc": 0.4750,
  "test_precision": 0.0,
  "test_recall": 0.0,
  "test_f1": 0.0
}
```

### Evaluation Metrics

| Metric | Description | Target (Production) |
|--------|------------|-------------------|
| ROC AUC | Area under ROC curve | > 0.85 |
| Precision | True positives / predicted positives | > 80% |
| Recall | True positives / actual positives | > 70% |
| F1 Score | Harmonic mean of precision and recall | Derived |
| False Positive Rate | False positives / actual negatives | < 5% |

> **Note:** Current metrics are low because models are trained on synthetic data. Real production data with stronger fraud/abuse signals will yield significantly better metrics.

---

## 10. FastAPI Inference Service

### Application Lifecycle

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Load all model pipelines from disk
    for name, filename in MODEL_FILES.items():
        path = settings.MODEL_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)
        else:
            logger.warning(f"Model file not found: {path}")
    yield
    # SHUTDOWN: Clear model registry
    models.clear()
```

- Models are loaded **once** at startup into an in-memory `models` dict
- If a `.pkl` file is missing, the service still starts — that endpoint returns 503
- Models are cleared on graceful shutdown

### Prediction Flow

```python
def _predict(model_name: str, request_data: dict) -> float:
    pipeline = _require_model(model_name)                    # Get loaded pipeline or 503
    feature_cols = get_all_feature_columns(model_name)       # Ordered column list
    df = pd.DataFrame([request_data])[feature_cols]          # Single-row DataFrame
    proba = pipeline.predict_proba(df)[0, 1]                 # Positive-class probability
    return float(proba)
```

Key details:
- `pd.DataFrame([request_data])[feature_cols]` ensures column order matches training
- `predict_proba(df)[0, 1]` returns the probability of the positive class (index 1)
- Latency is measured with `time.perf_counter()` (sub-millisecond precision)

### Starting the Server

```bash
cd ai-service
source ../fastapi-env/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Production:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 11. API Reference

### Base URL

```
http://<host>:8000
```

### Interactive Documentation

- **Swagger UI**: `http://<host>:8000/docs`
- **ReDoc**: `http://<host>:8000/redoc`

---

### GET /health

Health check endpoint. **No authentication required** (for load balancer probes).

**Response:**
```json
{
  "status": "ok",
  "model_version": "0.1.0",
  "models_loaded": {
    "order_risk": true,
    "refund_risk": true,
    "return_risk": true
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always "ok" if service is running |
| `model_version` | string | Current model version from config |
| `models_loaded` | object | Boolean for each model indicating if it's loaded |

---

### POST /predict/order-risk

Predict fraud risk for an order. **Requires Bearer token.**

**Request Headers:**
```
Authorization: Bearer <API_TOKEN>
Content-Type: application/json
```

**Request Body:**
```json
{
  "order_amount": 120.50,
  "currency": "USD",
  "billing_country": "US",
  "shipping_country": "NG",
  "country_mismatch": true,
  "customer_total_orders": 5,
  "customer_refund_rate": 0.45,
  "customer_dispute_rate": 0.02,
  "device_reuse_count": 3,
  "ip_reuse_count": 1,
  "order_velocity_24h": 5
}
```

**Request Field Validation:**

| Field | Type | Constraints |
|-------|------|------------|
| `order_amount` | float | > 0 |
| `currency` | string | length 3 (e.g., "USD") |
| `billing_country` | string | length 2 (e.g., "US") |
| `shipping_country` | string | length 2 |
| `country_mismatch` | boolean | |
| `customer_total_orders` | integer | >= 0 |
| `customer_refund_rate` | float | 0.0 - 1.0 |
| `customer_dispute_rate` | float | 0.0 - 1.0 |
| `device_reuse_count` | integer | >= 0 |
| `ip_reuse_count` | integer | >= 0 |
| `order_velocity_24h` | integer | >= 0 |

**Response (200):**
```json
{
  "risk_score": 0.0079,
  "model_version": "0.1.0"
}
```

---

### POST /predict/refund-risk

Predict refund abuse risk. **Requires Bearer token.**

**Request Body (extends order-risk fields + 4 extra):**
```json
{
  "order_amount": 250.00,
  "currency": "EUR",
  "billing_country": "DE",
  "shipping_country": "DE",
  "country_mismatch": false,
  "customer_total_orders": 12,
  "customer_refund_rate": 0.35,
  "customer_dispute_rate": 0.08,
  "device_reuse_count": 1,
  "ip_reuse_count": 2,
  "order_velocity_24h": 1,
  "refund_reason": "not_as_described",
  "days_since_purchase": 30,
  "delivery_confirmed": false,
  "item_category": "electronics"
}
```

**Extra Fields:**

| Field | Type | Constraints |
|-------|------|------------|
| `refund_reason` | string | Free text (e.g., "not_as_described", "damaged", "wrong_item") |
| `days_since_purchase` | integer | >= 0 |
| `delivery_confirmed` | boolean | |
| `item_category` | string | Free text (e.g., "electronics", "clothing") |

**Response (200):**
```json
{
  "abuse_score": 0.0526,
  "model_version": "0.1.0"
}
```

---

### POST /predict/return-risk

Predict return probability. **Requires Bearer token.**

**Request Body:** Same as `/predict/order-risk` (11 common fields).

**Response (200):**
```json
{
  "return_score": 0.1959,
  "model_version": "0.1.0"
}
```

---

### Error Responses

| Status | Condition | Response Body |
|--------|----------|--------------|
| 401 | Missing or invalid Bearer token | `{"detail": "Invalid token"}` |
| 422 | Invalid request body (Pydantic validation) | `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}` |
| 503 | Model not loaded (`.pkl` missing) | `{"detail": "Model 'order_risk' is not available. Has it been trained?"}` |

**Example 422 Response (validation error):**
```json
{
  "detail": [
    {
      "type": "greater_than",
      "loc": ["body", "order_amount"],
      "msg": "Input should be greater than 0",
      "input": -5.0
    }
  ]
}
```

---

## 12. Authentication & Security

### Token Authentication

The service uses **HTTP Bearer Token** authentication via FastAPI's `HTTPBearer` security scheme.

**How it works:**
1. Laravel sends `Authorization: Bearer <token>` header with every request
2. `verify_token()` dependency compares the token against `settings.API_TOKEN`
3. Mismatch returns HTTP 401 before any model code runs
4. The `/health` endpoint is **unauthenticated** (for load balancer probes)

```python
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != settings.API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials
```

### Security Measures

| Measure | Implementation |
|---------|---------------|
| Token auth | Bearer token verified on all prediction endpoints |
| No PII in logs | Only `order_amount` and scores are logged. No customer IDs, IPs, emails, or device IDs |
| No PII in schemas | Request schemas contain no email, name, or address fields |
| .env gitignored | API token not committed to source control |
| Network restriction | In production, FastAPI should only be accessible from Laravel's IP (via firewall/security group) |
| Input validation | Pydantic enforces type, range, and length constraints at the API boundary |

### Production Security Recommendations

- Generate a strong random token: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- Restrict network access to Laravel's IP only (firewall rules / security groups)
- Use HTTPS in production (TLS termination at load balancer or reverse proxy)
- Hash emails before including in training data
- Rotate API tokens periodically

---

## 13. Laravel Integration Guide

### HTTP Client Setup (Laravel)

```php
// config/services.php
'ai_service' => [
    'base_url' => env('AI_SERVICE_URL', 'http://localhost:8000'),
    'token' => env('AI_SERVICE_TOKEN'),
],
```

```env
# .env
AI_SERVICE_URL=http://ai-service:8000
AI_SERVICE_TOKEN=dev-token-change-in-production
```

### Making Predictions

```php
use Illuminate\Support\Facades\Http;

class FraudDetectionService
{
    private string $baseUrl;
    private string $token;

    public function __construct()
    {
        $this->baseUrl = config('services.ai_service.base_url');
        $this->token = config('services.ai_service.token');
    }

    public function predictOrderRisk(array $orderData): float
    {
        $response = Http::withToken($this->token)
            ->timeout(5)
            ->post("{$this->baseUrl}/predict/order-risk", [
                'order_amount' => $orderData['amount'],
                'currency' => $orderData['currency'],
                'billing_country' => $orderData['billing_country'],
                'shipping_country' => $orderData['shipping_country'],
                'country_mismatch' => $orderData['billing_country'] !== $orderData['shipping_country'],
                'customer_total_orders' => $orderData['customer_total_orders'],
                'customer_refund_rate' => $orderData['customer_refund_rate'],
                'customer_dispute_rate' => $orderData['customer_dispute_rate'],
                'device_reuse_count' => $orderData['device_reuse_count'],
                'ip_reuse_count' => $orderData['ip_reuse_count'],
                'order_velocity_24h' => $orderData['order_velocity_24h'],
            ]);

        if ($response->failed()) {
            throw new \Exception("AI service error: " . $response->body());
        }

        return $response->json('risk_score');
    }

    public function predictRefundRisk(array $refundData): float
    {
        $response = Http::withToken($this->token)
            ->timeout(5)
            ->post("{$this->baseUrl}/predict/refund-risk", $refundData);

        return $response->json('abuse_score');
    }

    public function predictReturnRisk(array $orderData): float
    {
        $response = Http::withToken($this->token)
            ->timeout(5)
            ->post("{$this->baseUrl}/predict/return-risk", $orderData);

        return $response->json('return_score');
    }
}
```

### Decision Engine (Thresholds)

The ML service only predicts. **Laravel applies the business rules:**

```php
class OrderDecisionEngine
{
    public function decide(float $riskScore): string
    {
        if ($riskScore > 0.7) {
            return 'deny';
        }

        if ($riskScore >= 0.4) {
            return 'manual_review';
        }

        return 'approve';
    }
}
```

| Score Range | Decision | Action |
|-------------|----------|--------|
| < 0.4 | Approve | Process order automatically |
| 0.4 - 0.7 | Manual Review | Flag for human review |
| > 0.7 | Deny | Block order automatically |

### Health Check Integration

```php
// Laravel health check for AI service
$response = Http::get("{$baseUrl}/health");
$health = $response->json();

if ($health['status'] !== 'ok' || !$health['models_loaded']['order_risk']) {
    // Alert: AI service or models unavailable
    Log::critical('AI service unhealthy', $health);
}
```

---

## 14. Model Retraining

### Schedule

| Frequency | When | Trigger |
|-----------|------|---------|
| Weekly | Low-risk, stable metrics | Cron job |
| On-demand | After significant data changes or metric degradation | Manual |

### Retraining Steps

```bash
# 1. Export latest data from PostgreSQL to CSV
#    (Use the SQL query from Section 5)

# 2. Place CSVs in ai-service/data/

# 3. Update model version in .env
MODEL_VERSION=0.2.0

# 4. Retrain all models
cd ai-service
source ../fastapi-env/bin/activate
python train_order_model.py --data data/order_fraud_data.csv
python train_refund_model.py --data data/refund_abuse_data.csv
python train_return_model.py --data data/return_data.csv

# 5. Check metrics in models/*_metrics.json
cat models/order_risk_metrics.json

# 6. Restart FastAPI to load new models
# (restart the uvicorn process)
```

### Retraining Safety Checklist

- [ ] Verify new data has sufficient positive examples (fraud rate > 1%)
- [ ] Compare new metrics against previous version (`*_metrics.json`)
- [ ] Ensure AUC has not degraded significantly (> 5% drop = investigate)
- [ ] Back up previous `.pkl` files before overwriting
- [ ] Test predictions with known examples after restarting server
- [ ] Update `MODEL_VERSION` in `.env` to track which version is serving

---

## 15. Logging & Monitoring

### Log Format

```
2026-02-26 05:18:51,147 [INFO] ai-service: Loaded model: order_risk from /path/to/models/order_risk_pipeline.pkl
2026-02-26 05:19:01,234 [INFO] ai-service: Prediction: model=order_risk, score=0.0079, latency=2.3ms, amount=120.5
```

### What Gets Logged

| Event | Level | Fields |
|-------|-------|--------|
| Model loaded at startup | INFO | model name, file path |
| Model not found at startup | WARNING | file path |
| Prediction made | INFO | model name, score, latency (ms), order_amount |
| Models unloaded at shutdown | INFO | — |

### What Is NOT Logged (Security)

- Customer IDs
- Email addresses
- IP addresses
- Device fingerprints
- Billing/shipping addresses
- Full request bodies

### Monitoring Recommendations

**Metric drift detection:**
- Track average prediction scores over time
- Alert if mean score shifts significantly (may indicate data drift)

**Latency monitoring:**
- Parse `latency=X.Xms` from logs
- Alert if inference exceeds 100ms

**Model availability:**
- Poll `/health` endpoint periodically
- Alert if any model shows `false` in `models_loaded`

**Retraining triggers:**
- Track precision/recall on manually labeled samples
- Retrain when metrics drop below thresholds

---

## 16. Deployment

### Local Development

```bash
# 1. Create and activate virtual environment (already done)
cd ai-refund-return-intelligence
source fastapi-env/bin/activate

# 2. Install dependencies
cd ai-service
pip install -r requirements.txt

# 3. Install OpenMP (macOS only, required for XGBoost)
brew install libomp

# 4. Configure
cp .env.example .env
# Edit .env with your API_TOKEN

# 5. Generate synthetic data and train models
python generate_data.py --samples 10000
python train_order_model.py
python train_refund_model.py
python train_return_model.py

# 6. Start server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Production Deployment

```bash
# Multi-worker production server
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment (Recommended for Production)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY ai-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ai-service/ .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```yaml
# docker-compose.yml
version: "3.8"
services:
  ai-service:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./ai-service/models:/app/models
    env_file:
      - ./ai-service/.env
    restart: unless-stopped
```

### Environment Requirements

| Requirement | Development | Production |
|-------------|------------|-----------|
| Python | 3.9+ | 3.9+ |
| RAM | 512MB minimum | 1GB+ recommended |
| Disk | ~100MB (models + dependencies) | Same |
| CPU | Any | Multi-core recommended |
| OpenMP | Required on macOS (`brew install libomp`) | Included in Linux containers |

---

## 17. Troubleshooting

### Common Issues

**XGBoost fails to load on macOS:**
```
XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded.
```
**Fix:** `brew install libomp`

**API returns 503 "Model not available":**
- Models haven't been trained yet
- `.pkl` files are missing from `models/`
- Run training scripts first, then restart the server

**API returns 401 "Invalid token":**
- Check `Authorization: Bearer <token>` header matches `API_TOKEN` in `.env`
- Ensure no extra whitespace in the token

**API returns 422 (validation error):**
- Check field types match the schema (e.g., `order_amount` must be > 0)
- Ensure all required fields are present
- `currency` must be exactly 3 characters, country codes exactly 2

**Port already in use:**
```
ERROR: [Errno 48] error while attempting to bind on address ('0.0.0.0', 8000)
```
**Fix:** Kill the existing process or use a different port: `uvicorn app:app --port 8001`

**Low model metrics (AUC ~0.5):**
- Expected with synthetic data — synthetic labels have weak correlation with features
- Real production data with genuine fraud/abuse patterns will yield much better metrics
- Consider adjusting `scale_pos_weight` in XGB params for severe class imbalance

### Quick Test Commands

```bash
# Health check
curl http://localhost:8000/health

# Order risk prediction
curl -X POST http://localhost:8000/predict/order-risk \
  -H "Authorization: Bearer dev-token-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "order_amount": 120.50,
    "currency": "USD",
    "billing_country": "US",
    "shipping_country": "NG",
    "country_mismatch": true,
    "customer_total_orders": 5,
    "customer_refund_rate": 0.45,
    "customer_dispute_rate": 0.02,
    "device_reuse_count": 3,
    "ip_reuse_count": 1,
    "order_velocity_24h": 5
  }'

# Refund risk prediction
curl -X POST http://localhost:8000/predict/refund-risk \
  -H "Authorization: Bearer dev-token-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "order_amount": 250.00,
    "currency": "EUR",
    "billing_country": "DE",
    "shipping_country": "DE",
    "country_mismatch": false,
    "customer_total_orders": 12,
    "customer_refund_rate": 0.35,
    "customer_dispute_rate": 0.08,
    "device_reuse_count": 1,
    "ip_reuse_count": 2,
    "order_velocity_24h": 1,
    "refund_reason": "not_as_described",
    "days_since_purchase": 30,
    "delivery_confirmed": false,
    "item_category": "electronics"
  }'

# Return risk prediction
curl -X POST http://localhost:8000/predict/return-risk \
  -H "Authorization: Bearer dev-token-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "order_amount": 350.00,
    "currency": "GBP",
    "billing_country": "GB",
    "shipping_country": "US",
    "country_mismatch": true,
    "customer_total_orders": 2,
    "customer_refund_rate": 0.20,
    "customer_dispute_rate": 0.01,
    "device_reuse_count": 5,
    "ip_reuse_count": 3,
    "order_velocity_24h": 4
  }'
```

---

## Future Upgrades

| Enhancement | Description | Priority |
|------------|-------------|----------|
| SHAP Explainability | Add feature importance per prediction for audit trails | High |
| Neural Networks | Deep learning models for complex pattern detection | Medium |
| Cross-merchant Learning | Federated learning across multiple merchants | Medium |
| Graph Fraud Detection | Network analysis of device/IP/email relationships | Low |
| Batch Prediction API | Endpoint for scoring multiple orders at once | Medium |
| A/B Testing Framework | Compare model versions in production | Medium |
| Model Registry | MLflow or similar for versioning and experiment tracking | High |
