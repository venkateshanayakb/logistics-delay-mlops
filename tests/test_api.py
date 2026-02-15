"""
Test FastAPI Endpoints
=======================
Tests for api/main.py — all routes, error handling, and Prometheus metrics.
Uses the test_client fixture from conftest.py.
"""



class TestRootEndpoint:
    """Tests for GET /."""

    def test_root_returns_service_info(self, test_client):
        resp = test_client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "Logistics Delay Prediction API"
        assert "version" in data
        assert "/predict" in data["endpoints"]


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200

    def test_health_model_loaded(self, test_client):
        resp = test_client.get("/health")
        data = resp.json()
        assert data["model_loaded"] is True
        assert data["status"] == "healthy"
        assert data["model_name"] == "test_rf"

    def test_health_has_uptime(self, test_client):
        resp = test_client.get("/health")
        data = resp.json()
        assert data["uptime_seconds"] >= 0


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_valid_input(self, test_client, sample_input):
        resp = test_client.post("/predict", json=sample_input)
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] in [-1, 0, 1]
        assert data["class_name"] in ["Early", "On-time", "Late"]
        assert 0.0 <= data["max_confidence"] <= 1.0
        assert isinstance(data["confidence"], dict)

    def test_predict_missing_required_field(self, test_client, sample_input):
        """Missing required field should return 422."""
        data = sample_input.copy()
        del data["sales"]
        resp = test_client.post("/predict", json=data)
        assert resp.status_code == 422

    def test_predict_invalid_field_value(self, test_client, sample_input):
        """Invalid constraint (e.g. order_month=13) should return 422."""
        data = sample_input.copy()
        data["order_month"] = 13
        resp = test_client.post("/predict", json=data)
        assert resp.status_code == 422

    def test_predict_empty_body(self, test_client):
        """Empty JSON body should return 422."""
        resp = test_client.post("/predict", json={})
        assert resp.status_code == 422

    def test_predict_response_has_top_features(self, test_client, sample_input):
        """Response should include top_features (may be empty list)."""
        resp = test_client.post("/predict", json=sample_input)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["top_features"], list)


class TestMetricsEndpoint:
    """Tests for GET /metrics."""

    def test_metrics_returns_prometheus_format(self, test_client):
        resp = test_client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")
        body = resp.text
        # Should contain at least one of our custom metrics
        assert "prediction_latency_seconds" in body or "predictions_total" in body

    def test_metrics_updated_after_prediction(self, test_client, sample_input):
        """Metrics should reflect predictions after they happen."""
        # Make a prediction first
        test_client.post("/predict", json=sample_input)
        resp = test_client.get("/metrics")
        body = resp.text
        assert "predictions_total" in body
