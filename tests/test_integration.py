"""
Integration Tests
==================
End-to-end flows through the FastAPI app — predict cycle, health state,
sequential consistency, and SHAP feature presence.
"""

import pytest


class TestPredictionCycle:
    """Full predict → verify → check metrics cycle."""

    def test_full_predict_cycle(self, test_client, sample_input):
        """POST /predict should succeed and /metrics should reflect it."""
        # 1. Health check — model should be loaded
        health = test_client.get("/health").json()
        assert health["model_loaded"] is True

        # 2. Make a prediction
        resp = test_client.post("/predict", json=sample_input)
        assert resp.status_code == 200
        data = resp.json()

        # 3. Verify response structure
        assert data["label"] in [-1, 0, 1]
        assert data["class_name"] in ["Early", "On-time", "Late"]
        assert isinstance(data["confidence"], dict)
        assert len(data["confidence"]) == 3  # 3 classes
        assert sum(data["confidence"].values()) == pytest.approx(1.0, abs=0.02)

        # 4. Check that metrics were updated
        metrics = test_client.get("/metrics").text
        assert "predictions_total" in metrics


class TestSequentialConsistency:
    """Multiple predictions should be consistent."""

    def test_same_input_same_output(self, test_client, sample_input):
        """Identical inputs should always produce the same prediction."""
        results = []
        for _ in range(3):
            resp = test_client.post("/predict", json=sample_input)
            assert resp.status_code == 200
            results.append(resp.json())

        # All should match
        for r in results[1:]:
            assert r["label"] == results[0]["label"]
            assert r["class_name"] == results[0]["class_name"]

    def test_different_inputs_accepted(self, test_client, sample_input):
        """Different valid inputs should all succeed (may differ in output)."""
        data1 = sample_input.copy()
        data2 = sample_input.copy()
        data2["shipping_lead_days"] = 15.0
        data2["profit_per_order"] = -20.0

        r1 = test_client.post("/predict", json=data1)
        r2 = test_client.post("/predict", json=data2)

        assert r1.status_code == 200
        assert r2.status_code == 200


class TestSHAPIntegration:
    """Verify SHAP features are present in responses."""

    def test_top_features_in_response(self, test_client, sample_input):
        """top_features should be a list of feature impact dicts."""
        resp = test_client.post("/predict", json=sample_input)
        data = resp.json()
        top_features = data["top_features"]
        assert isinstance(top_features, list)
        # If SHAP is available, features should have correct structure
        for feat in top_features:
            assert "feature" in feat
            assert "impact" in feat
            assert isinstance(feat["feature"], str)
            assert isinstance(feat["impact"], float)
