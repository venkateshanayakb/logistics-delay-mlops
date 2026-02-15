"""
Test Data Validation (Pydantic Schemas)
========================================
Tests for api/schemas.py — PredictionRequest, PredictionResponse,
HealthResponse field validation and constraints.
"""

import pytest
from pydantic import ValidationError

from api.schemas import (
    FeatureImpact,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)


class TestPredictionRequest:
    """Tests for PredictionRequest schema."""

    def test_valid_input_parses(self, sample_input):
        """A complete, valid input should parse without errors."""
        req = PredictionRequest(**sample_input)
        assert req.profit_per_order == 50.0
        assert req.shipping_mode == "Standard Class"

    def test_missing_required_field_raises(self, sample_input):
        """Missing a required field like 'sales' should raise ValidationError."""
        data = sample_input.copy()
        del data["sales"]
        with pytest.raises(ValidationError):
            PredictionRequest(**data)

    def test_order_dayofweek_min_constraint(self, sample_input):
        """order_dayofweek must be >= 0."""
        data = sample_input.copy()
        data["order_dayofweek"] = -1
        with pytest.raises(ValidationError):
            PredictionRequest(**data)

    def test_order_dayofweek_max_constraint(self, sample_input):
        """order_dayofweek must be <= 6."""
        data = sample_input.copy()
        data["order_dayofweek"] = 7
        with pytest.raises(ValidationError):
            PredictionRequest(**data)

    def test_order_month_constraint(self, sample_input):
        """order_month must be between 1 and 12."""
        data = sample_input.copy()
        data["order_month"] = 13
        with pytest.raises(ValidationError):
            PredictionRequest(**data)

    def test_order_quarter_constraint(self, sample_input):
        """order_quarter must be between 1 and 4."""
        data = sample_input.copy()
        data["order_quarter"] = 0
        with pytest.raises(ValidationError):
            PredictionRequest(**data)

    def test_quantity_must_be_positive(self, sample_input):
        """order_item_quantity must be >= 1."""
        data = sample_input.copy()
        data["order_item_quantity"] = 0
        with pytest.raises(ValidationError):
            PredictionRequest(**data)

    def test_defaults_applied(self):
        """Fields with defaults should work when only required fields provided."""
        # Only required fields (those without defaults)
        req = PredictionRequest(
            profit_per_order=50.0,
            sales_per_customer=200.0,
            latitude=28.6,
            longitude=77.2,
            order_item_product_price=100.0,
            sales=200.0,
            product_price=100.0,
            order_dayofweek=3,
            order_month=6,
            order_quarter=2,
            shipping_dayofweek=5,
            shipping_month=6,
            shipping_lead_days=3.5,
            category_name="Cleats",
            order_region="Central America",
        )
        assert req.order_item_discount == 0.0
        assert req.payment_type == "DEBIT"
        assert req.shipping_mode == "Standard Class"


class TestPredictionResponse:
    """Tests for PredictionResponse schema."""

    def test_valid_response(self):
        """A full response with all fields should parse."""
        resp = PredictionResponse(
            label=1,
            class_name="Late",
            confidence={"Early": 0.1, "On-time": 0.3, "Late": 0.6},
            max_confidence=0.6,
            top_features=[
                {"feature": "shipping_lead_days", "impact": 0.25},
            ],
        )
        assert resp.class_name == "Late"
        assert len(resp.top_features) == 1

    def test_empty_top_features(self):
        """top_features can be an empty list."""
        resp = PredictionResponse(
            label=0,
            class_name="On-time",
            confidence={"Early": 0.1, "On-time": 0.8, "Late": 0.1},
            max_confidence=0.8,
            top_features=[],
        )
        assert resp.top_features == []


class TestFeatureImpact:
    """Tests for FeatureImpact schema."""

    def test_valid_feature_impact(self):
        fi = FeatureImpact(feature="shipping_lead_days", impact=0.35)
        assert fi.feature == "shipping_lead_days"
        assert fi.impact == 0.35


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_defaults(self):
        """HealthResponse should have sensible defaults."""
        h = HealthResponse()
        assert h.status == "healthy"
        assert h.model_loaded is False
        assert h.uptime_seconds == 0.0

    def test_custom_values(self):
        h = HealthResponse(
            status="degraded",
            model_name="test_rf",
            model_loaded=True,
            uptime_seconds=123.45,
        )
        assert h.status == "degraded"
        assert h.model_name == "test_rf"
