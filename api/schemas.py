"""
Pydantic Schemas
=================
Request / response models for the Logistics Delay prediction API.
"""

from pydantic import BaseModel, Field


# ── Prediction Request ───────────────────────────────────────────
class PredictionRequest(BaseModel):
    """All 28 features expected by the trained pipeline."""

    # ── Numeric features (19) ────────────────────────────────────
    profit_per_order: float = Field(..., example=50.0, description="Profit per order in USD")
    sales_per_customer: float = Field(..., example=200.0, description="Total sales per customer")
    latitude: float = Field(..., example=28.6, description="Customer latitude")
    longitude: float = Field(..., example=77.2, description="Customer longitude")
    order_item_discount: float = Field(0.0, example=10.0, description="Discount amount on item")
    order_item_discount_rate: float = Field(0.0, example=0.05, description="Discount rate (0-1)")
    order_item_product_price: float = Field(..., example=100.0, description="Product price per item")
    order_item_profit_ratio: float = Field(0.0, example=0.25, description="Profit ratio on item")
    order_item_quantity: int = Field(1, ge=1, example=2, description="Quantity ordered")
    sales: float = Field(..., example=200.0, description="Sale amount")
    product_price: float = Field(..., example=100.0, description="Product base price")
    order_dayofweek: int = Field(..., ge=0, le=6, example=3, description="Day of week (0=Mon, 6=Sun)")
    order_month: int = Field(..., ge=1, le=12, example=6, description="Order month (1-12)")
    order_quarter: int = Field(..., ge=1, le=4, example=2, description="Order quarter (1-4)")
    shipping_dayofweek: int = Field(..., ge=0, le=6, example=5, description="Shipping day of week")
    shipping_month: int = Field(..., ge=1, le=12, example=6, description="Shipping month (1-12)")
    shipping_lead_days: float = Field(..., example=3.5, description="Days between order and shipment")
    discount_ratio: float = Field(0.0, example=0.1, description="discount / product_price")
    profit_margin: float = Field(0.0, example=0.25, description="profit / sales_per_customer")

    # ── Categorical features (9) ─────────────────────────────────
    payment_type: str = Field("DEBIT", example="DEBIT", description="Payment method")
    category_name: str = Field(..., example="Cleats", description="Product category name")
    customer_country: str = Field("EE. UU.", example="EE. UU.", description="Customer country")
    customer_segment: str = Field("Consumer", example="Consumer", description="Customer segment")
    department_name: str = Field("Fan Shop", example="Fan Shop", description="Product department")
    market: str = Field("LATAM", example="LATAM", description="Market region")
    order_region: str = Field(..., example="Central America", description="Order region")
    order_status: str = Field("COMPLETE", example="COMPLETE", description="Order status")
    shipping_mode: str = Field("Standard Class", example="Standard Class", description="Shipping mode")

    model_config = {"json_schema_extra": {
        "examples": [{
            "profit_per_order": 50.0,
            "sales_per_customer": 200.0,
            "latitude": 28.6,
            "longitude": 77.2,
            "order_item_discount": 10.0,
            "order_item_discount_rate": 0.05,
            "order_item_product_price": 100.0,
            "order_item_profit_ratio": 0.25,
            "order_item_quantity": 2,
            "sales": 200.0,
            "product_price": 100.0,
            "order_dayofweek": 3,
            "order_month": 6,
            "order_quarter": 2,
            "shipping_dayofweek": 5,
            "shipping_month": 6,
            "shipping_lead_days": 3.5,
            "discount_ratio": 0.1,
            "profit_margin": 0.25,
            "payment_type": "DEBIT",
            "category_name": "Cleats",
            "customer_country": "EE. UU.",
            "customer_segment": "Consumer",
            "department_name": "Fan Shop",
            "market": "LATAM",
            "order_region": "Central America",
            "order_status": "COMPLETE",
            "shipping_mode": "Standard Class",
        }]
    }}


# ── Prediction Response ──────────────────────────────────────────
class FeatureImpact(BaseModel):
    """Single feature's contribution to the prediction."""
    feature: str
    impact: float = Field(..., description="SHAP value — positive pushes toward predicted class")


class PredictionResponse(BaseModel):
    """Full prediction result with confidence and explainability."""
    label: int = Field(..., description="Predicted class: -1=Early, 0=On-time, 1=Late")
    class_name: str = Field(..., description="Human readable class name")
    confidence: dict[str, float] = Field(
        ..., description="Class probabilities, e.g. {'Early': 0.1, 'On-time': 0.3, 'Late': 0.6}"
    )
    max_confidence: float = Field(..., description="Highest probability among classes")
    top_features: list[FeatureImpact] = Field(
        default_factory=list,
        description="Top contributing features via SHAP (may be empty if SHAP unavailable)",
    )


# ── Health ────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    """API health check response."""
    status: str = "healthy"
    model_name: str = ""
    model_loaded: bool = False
    uptime_seconds: float = 0.0
