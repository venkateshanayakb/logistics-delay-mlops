"""
Test Model Service
===================
Tests for api/model_service.py — loading, inference, and SHAP explainability.
"""

import pytest

from api.model_service import ModelService, LABEL_MAP


class TestModelServiceInit:
    """Tests for ModelService initial state."""

    def test_not_loaded_by_default(self):
        """A fresh ModelService should not be loaded."""
        svc = ModelService()
        assert svc.is_loaded is False

    def test_predict_before_load_raises(self, sample_input):
        """Calling predict before load_model should raise RuntimeError."""
        svc = ModelService()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            svc.predict(sample_input)


class TestModelServiceLoad:
    """Tests for ModelService.load_model()."""

    def test_load_model_sets_attributes(self, tmp_model_path):
        """After loading, all model attributes should be populated."""
        svc = ModelService()
        load_time = svc.load_model(tmp_model_path)

        assert svc.is_loaded is True
        assert isinstance(load_time, float)
        assert load_time > 0
        assert svc.model_name == "test_rf"
        assert len(svc.feature_names) > 0
        assert len(svc.numeric_cols) > 0
        assert len(svc.categorical_cols) > 0
        assert svc.label_map == LABEL_MAP

    def test_load_nonexistent_path_raises(self):
        """Loading from a bad path should raise an exception."""
        svc = ModelService()
        with pytest.raises(Exception):
            svc.load_model("/nonexistent/path/model.joblib")


class TestModelServicePredict:
    """Tests for ModelService.predict()."""

    def test_predict_returns_expected_keys(self, tmp_model_path, sample_input):
        """Prediction result should contain all expected keys."""
        svc = ModelService()
        svc.load_model(tmp_model_path)
        result = svc.predict(sample_input)

        assert "label" in result
        assert "class_name" in result
        assert "confidence" in result
        assert "max_confidence" in result
        assert "top_features" in result

    def test_predict_label_is_valid(self, tmp_model_path, sample_input):
        """Predicted label should be one of {-1, 0, 1}."""
        svc = ModelService()
        svc.load_model(tmp_model_path)
        result = svc.predict(sample_input)
        assert result["label"] in [-1, 0, 1]

    def test_predict_class_name_matches_label(self, tmp_model_path, sample_input):
        """class_name should correspond to the label via LABEL_MAP."""
        svc = ModelService()
        svc.load_model(tmp_model_path)
        result = svc.predict(sample_input)
        assert result["class_name"] == LABEL_MAP[result["label"]]

    def test_predict_confidence_sums_to_one(self, tmp_model_path, sample_input):
        """Class probabilities should sum to approximately 1."""
        svc = ModelService()
        svc.load_model(tmp_model_path)
        result = svc.predict(sample_input)
        total = sum(result["confidence"].values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_predict_max_confidence_range(self, tmp_model_path, sample_input):
        """max_confidence should be between 0 and 1."""
        svc = ModelService()
        svc.load_model(tmp_model_path)
        result = svc.predict(sample_input)
        assert 0.0 <= result["max_confidence"] <= 1.0

    def test_predict_top_features_structure(self, tmp_model_path, sample_input):
        """top_features should be a list of dicts with 'feature' and 'impact'."""
        svc = ModelService()
        svc.load_model(tmp_model_path)
        result = svc.predict(sample_input)
        for feat in result["top_features"]:
            assert "feature" in feat
            assert "impact" in feat
            assert isinstance(feat["impact"], float)

    def test_predict_is_deterministic(self, tmp_model_path, sample_input):
        """Same input should produce same output."""
        svc = ModelService()
        svc.load_model(tmp_model_path)
        r1 = svc.predict(sample_input)
        r2 = svc.predict(sample_input)
        assert r1["label"] == r2["label"]
        assert r1["class_name"] == r2["class_name"]
        assert r1["max_confidence"] == r2["max_confidence"]
