from pydantic import BaseModel, Field


class CommonFeatures(BaseModel):
    order_amount: float = Field(..., gt=0)
    currency: str = Field(..., min_length=3, max_length=3)
    billing_country: str = Field(..., min_length=2, max_length=2)
    shipping_country: str = Field(..., min_length=2, max_length=2)
    country_mismatch: bool
    customer_total_orders: int = Field(..., ge=0)
    customer_refund_rate: float = Field(..., ge=0.0, le=1.0)
    customer_dispute_rate: float = Field(..., ge=0.0, le=1.0)
    device_reuse_count: int = Field(..., ge=0)
    ip_reuse_count: int = Field(..., ge=0)
    order_velocity_24h: int = Field(..., ge=0)


class OrderRiskRequest(CommonFeatures):
    pass


class RefundRiskRequest(CommonFeatures):
    refund_reason: str
    days_since_purchase: int = Field(..., ge=0)
    delivery_confirmed: bool
    item_category: str


class ReturnRiskRequest(CommonFeatures):
    pass


class RiskScoreResponse(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class AbuseScoreResponse(BaseModel):
    abuse_score: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class ReturnScoreResponse(BaseModel):
    return_score: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    models_loaded: dict
