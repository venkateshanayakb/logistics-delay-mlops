"""
Model Service
==============
Singleton service for loading the trained pipeline and running inference.
Includes SHAP explainability for individual predictions.
"""

import logging
import os
import time

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LABEL_MAP = {-1: "Early", 0: "On-time", 1: "Late"}
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "best_pipeline.joblib"))


class ModelService:
    """Load once, predict many — singleton model wrapper."""

    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.model_name: str = ""
        self.feature_names: list[str] = []
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.label_map: dict = LABEL_MAP
        self._loaded = False
        self._shap_explainer = None
        self.load_time: float = 0.0

    # ── Load ─────────────────────────────────────────────────────
    def load_model(self, path: str | None = None) -> float:
        """
        Load the pipeline artifact from disk.

        Returns:
            Load time in seconds.
        """
        path = path or MODEL_PATH
        logger.info(f"Loading model from: {path}")
        start = time.time()

        artifact = joblib.load(path)
        self.preprocessor = artifact["preprocessor"]
        self.model = artifact["model"]
        self.model_name = artifact.get("model_name", "unknown")
        self.feature_names = artifact.get("feature_names", [])
        self.numeric_cols = artifact.get("numeric_cols", [])
        self.categorical_cols = artifact.get("categorical_cols", [])
        self.label_map = artifact.get("label_map", LABEL_MAP)
        self._loaded = True

        self.load_time = time.time() - start
        logger.info(
            f"✅ Model loaded: {self.model_name} "
            f"({len(self.feature_names)} features, {self.load_time:.2f}s)"
        )
        return self.load_time

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Predict ──────────────────────────────────────────────────
    def predict(self, input_data: dict) -> dict:
        """
        Run inference on a single input.

        Args:
            input_data: dict of feature_name → value (from PredictionRequest).

        Returns:
            dict with keys: label, class_name, confidence, max_confidence, top_features
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded — call load_model() first")

        # Build DataFrame in the exact feature order the preprocessor expects
        df = pd.DataFrame([input_data])[self.feature_names]

        # Preprocess
        X_processed = self.preprocessor.transform(df)

        # Predict
        label = int(self.model.predict(X_processed)[0])
        class_name = self.label_map.get(label, str(label))

        # Probabilities
        probabilities = {}
        max_conf = 0.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_processed)[0]
            classes = self.model.classes_
            probabilities = {
                self.label_map.get(int(c), str(c)): round(float(p), 4)
                for c, p in zip(classes, proba)
            }
            max_conf = round(float(proba.max()), 4)

        # SHAP explanation (best-effort)
        top_features = self._get_shap_features(X_processed, label)

        return {
            "label": label,
            "class_name": class_name,
            "confidence": probabilities,
            "max_confidence": max_conf,
            "top_features": top_features,
        }

    # ── SHAP ─────────────────────────────────────────────────────
    def _get_shap_explainer(self):
        """Lazy-init SHAP TreeExplainer."""
        if self._shap_explainer is None:
            try:
                import shap
                self._shap_explainer = shap.TreeExplainer(self.model)
                logger.info("SHAP TreeExplainer initialized")
            except Exception as exc:
                logger.warning(f"SHAP unavailable: {exc}")
                self._shap_explainer = False  # sentinel: tried and failed
        return self._shap_explainer if self._shap_explainer is not False else None

    def _get_shap_features(self, X_processed: np.ndarray, predicted_label: int, top_n: int = 5) -> list[dict]:
        """Return top-N feature impacts for the predicted class."""
        explainer = self._get_shap_explainer()
        if explainer is None:
            return []

        try:
            shap_values = explainer.shap_values(X_processed)

            # shap_values shape: (n_classes, n_samples, n_features) or list
            classes = list(self.model.classes_)
            class_idx = classes.index(predicted_label) if predicted_label in classes else 0

            if isinstance(shap_values, list):
                values = shap_values[class_idx][0]
            else:
                values = shap_values[0, :] if shap_values.ndim == 2 else shap_values[class_idx][0]

            # Get feature names after transformation
            try:
                transformed_names = self.preprocessor.get_feature_names_out()
            except Exception:
                transformed_names = [f"feature_{i}" for i in range(len(values))]

            # Sort by absolute impact
            sorted_idx = np.argsort(np.abs(values))[::-1][:top_n]
            return [
                {"feature": str(transformed_names[i]), "impact": round(float(values[i]), 4)}
                for i in sorted_idx
            ]
        except Exception as exc:
            logger.warning(f"SHAP computation failed: {exc}")
            return []


# ── Global singleton ─────────────────────────────────────────────
model_service = ModelService()
