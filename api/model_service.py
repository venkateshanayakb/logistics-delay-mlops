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
        self.label_encoder = None  # LabelEncoder for decoding predictions
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
        self.label_encoder = artifact.get("label_encoder", None)
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

        # Build DataFrame and apply the same feature engineering used in training
        df = pd.DataFrame([input_data])
        df = self._engineer_features(df)
        df = df[self.feature_names]

        # Preprocess
        X_processed = self.preprocessor.transform(df)

        # Predict
        raw_label = int(self.model.predict(X_processed)[0])

        # Decode label if label_encoder exists (for XGBoost compat)
        if self.label_encoder is not None:
            label = int(self.label_encoder.inverse_transform([raw_label])[0])
        else:
            label = raw_label

        class_name = self.label_map.get(label, str(label))

        # Probabilities
        probabilities = {}
        max_conf = 0.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_processed)[0]
            classes = self.model.classes_
            for c, p in zip(classes, proba):
                # Decode class label if encoder exists
                if self.label_encoder is not None:
                    orig_label = int(self.label_encoder.inverse_transform([int(c)])[0])
                else:
                    orig_label = int(c)
                probabilities[self.label_map.get(orig_label, str(orig_label))] = round(float(p), 4)
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

    # ── Feature Engineering (mirrors src/preprocessing.engineer_features) ──
    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same feature engineering that was used during training.
        Only the runtime-derivable features (no raw date parsing needed
        since the API schema already provides date-derived fields).
        """
        df = df.copy()

        # Interaction features
        if "order_item_product_price" in df.columns and "order_item_quantity" in df.columns:
            df["price_x_quantity"] = df["order_item_product_price"] * df["order_item_quantity"]

        if "order_item_discount" in df.columns and "order_item_quantity" in df.columns:
            df["discount_x_quantity"] = df["order_item_discount"] * df["order_item_quantity"]

        if "sales" in df.columns and "product_price" in df.columns:
            df["sales_to_price_ratio"] = df["sales"] / df["product_price"].replace(0, np.nan)

        # Weekend flags
        if "order_dayofweek" in df.columns:
            df["is_weekend_order"] = (df["order_dayofweek"] >= 5).astype(int)

        if "shipping_dayofweek" in df.columns:
            df["is_weekend_ship"] = (df["shipping_dayofweek"] >= 5).astype(int)

        # Shipping speed index
        if "sales" in df.columns and "shipping_lead_days" in df.columns:
            df["shipping_speed_index"] = df["sales"] / df["shipping_lead_days"].replace(0, np.nan)

        # High discount flag
        if "order_item_discount_rate" in df.columns:
            df["high_discount_flag"] = (df["order_item_discount_rate"] > 0.15).astype(int)

        # Log transforms for skewed numerics
        for col in ["sales", "product_price", "sales_per_customer", "profit_per_order"]:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

        return df

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
