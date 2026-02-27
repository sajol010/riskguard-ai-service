import time
import logging
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import settings
from schemas import (
    OrderRiskRequest,
    RefundRiskRequest,
    ReturnRiskRequest,
    RiskScoreResponse,
    AbuseScoreResponse,
    ReturnScoreResponse,
    HealthResponse,
)
from preprocessing import get_all_feature_columns

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai-service")

# Model registry (populated at startup)
models = {}

MODEL_FILES = {
    "order_risk": "order_risk_pipeline.pkl",
    "refund_risk": "refund_risk_pipeline.pkl",
    "return_risk": "return_risk_pipeline.pkl",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name, filename in MODEL_FILES.items():
        path = settings.MODEL_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)
            logger.info(f"Loaded model: {name} from {path}")
        else:
            logger.warning(f"Model file not found: {path}. Endpoint will return 503.")
    yield
    models.clear()
    logger.info("Models unloaded.")


app = FastAPI(
    title="AI Refund & Return Intelligence",
    version=settings.MODEL_VERSION,
    lifespan=lifespan,
)

# Auth
security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != settings.API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials


def _require_model(name: str):
    if name not in models:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{name}' is not available. Has it been trained?",
        )
    return models[name]


def _predict(model_name: str, request_data: dict) -> float:
    pipeline = _require_model(model_name)
    feature_cols = get_all_feature_columns(model_name)
    df = pd.DataFrame([request_data])[feature_cols]

    start = time.perf_counter()
    proba = pipeline.predict_proba(df)[0, 1]
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        f"Prediction: model={model_name}, score={proba:.4f}, "
        f"latency={elapsed_ms:.1f}ms, amount={request_data.get('order_amount')}"
    )
    return float(proba)


# Endpoints

@app.post("/predict/order-risk", response_model=RiskScoreResponse)
def predict_order_risk(
    request: OrderRiskRequest,
    _token: HTTPAuthorizationCredentials = Depends(verify_token),
):
    score = _predict("order_risk", request.model_dump())
    return RiskScoreResponse(risk_score=score, model_version=settings.MODEL_VERSION)


@app.post("/predict/refund-risk", response_model=AbuseScoreResponse)
def predict_refund_risk(
    request: RefundRiskRequest,
    _token: HTTPAuthorizationCredentials = Depends(verify_token),
):
    score = _predict("refund_risk", request.model_dump())
    return AbuseScoreResponse(abuse_score=score, model_version=settings.MODEL_VERSION)


@app.post("/predict/return-risk", response_model=ReturnScoreResponse)
def predict_return_risk(
    request: ReturnRiskRequest,
    _token: HTTPAuthorizationCredentials = Depends(verify_token),
):
    score = _predict("return_risk", request.model_dump())
    return ReturnScoreResponse(return_score=score, model_version=settings.MODEL_VERSION)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        model_version=settings.MODEL_VERSION,
        models_loaded={name: (name in models) for name in MODEL_FILES},
    )
