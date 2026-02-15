"""
Test Preprocessing Pipeline
============================
Tests for src/preprocessing.py — feature engineering, column selection,
and ColumnTransformer construction.
"""

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.preprocessing import (
    DROP_COLUMNS,
    TARGET,
    build_preprocessor,
    engineer_features,
    get_feature_columns,
)


class TestEngineerFeatures:
    """Tests for engineer_features()."""

    def test_date_features_created(self, sample_dataframe):
        """Date columns should produce dayofweek, month, and quarter features."""
        result = engineer_features(sample_dataframe)
        assert "order_dayofweek" in result.columns
        assert "order_month" in result.columns
        assert "order_quarter" in result.columns
        assert "shipping_dayofweek" in result.columns
        assert "shipping_month" in result.columns

    def test_shipping_lead_days_computed(self, sample_dataframe):
        """shipping_lead_days should be ~3 days (fixtures use +3 day offset)."""
        result = engineer_features(sample_dataframe)
        assert "shipping_lead_days" in result.columns
        assert result["shipping_lead_days"].mean() == pytest.approx(3.0, abs=0.1)

    def test_discount_ratio_computed(self, sample_dataframe):
        """discount_ratio = discount / product_price."""
        result = engineer_features(sample_dataframe)
        assert "discount_ratio" in result.columns
        # Manually verify a row
        idx = 0
        expected = (sample_dataframe.loc[idx, "order_item_discount"]
                    / sample_dataframe.loc[idx, "product_price"])
        assert result.loc[idx, "discount_ratio"] == pytest.approx(expected, rel=1e-5)

    def test_profit_margin_computed(self, sample_dataframe):
        """profit_margin = profit_per_order / sales_per_customer."""
        result = engineer_features(sample_dataframe)
        assert "profit_margin" in result.columns

    def test_no_temp_columns_remain(self, sample_dataframe):
        """Internal __*_parsed columns should be dropped."""
        result = engineer_features(sample_dataframe)
        temp_cols = [c for c in result.columns if c.startswith("__")]
        assert len(temp_cols) == 0

    def test_original_df_not_mutated(self, sample_dataframe):
        """engineer_features should not modify the input DataFrame."""
        original_cols = list(sample_dataframe.columns)
        engineer_features(sample_dataframe)
        assert list(sample_dataframe.columns) == original_cols

    def test_missing_date_columns(self):
        """Should handle DataFrames without date columns gracefully."""
        df = pd.DataFrame({
            "profit_per_order": [10.0, 20.0],
            "sales_per_customer": [100.0, 200.0],
            "product_price": [50.0, 100.0],
            "order_item_discount": [5.0, 10.0],
            "label": [0, 1],
        })
        result = engineer_features(df)
        # No error, and date features should not exist
        assert "order_dayofweek" not in result.columns
        assert "shipping_lead_days" not in result.columns
        # Ratio features should still be created
        assert "discount_ratio" in result.columns


class TestGetFeatureColumns:
    """Tests for get_feature_columns()."""

    def test_returns_numeric_and_categorical(self, sample_dataframe):
        """Should return two lists of column names."""
        df = engineer_features(sample_dataframe)
        numeric_cols, categorical_cols = get_feature_columns(df)
        assert isinstance(numeric_cols, list)
        assert isinstance(categorical_cols, list)
        assert len(numeric_cols) > 0
        assert len(categorical_cols) > 0

    def test_drop_columns_excluded(self, sample_dataframe):
        """IDs and leaky columns should not appear in feature lists."""
        df = engineer_features(sample_dataframe)
        numeric_cols, categorical_cols = get_feature_columns(df)
        all_features = set(numeric_cols + categorical_cols)
        for col in DROP_COLUMNS:
            assert col not in all_features, f"{col} should be dropped"

    def test_target_excluded(self, sample_dataframe):
        """The target column should not appear in features."""
        df = engineer_features(sample_dataframe)
        numeric_cols, categorical_cols = get_feature_columns(df)
        all_features = set(numeric_cols + categorical_cols)
        assert TARGET not in all_features


class TestBuildPreprocessor:
    """Tests for build_preprocessor()."""

    def test_returns_column_transformer(self, sample_dataframe):
        """Should return a ColumnTransformer instance."""
        df = engineer_features(sample_dataframe)
        numeric_cols, categorical_cols = get_feature_columns(df)
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        assert isinstance(preprocessor, ColumnTransformer)

    def test_fit_transform_shape(self, sample_dataframe):
        """fit_transform should produce the expected number of rows."""
        df = engineer_features(sample_dataframe)
        numeric_cols, categorical_cols = get_feature_columns(df)
        feature_names = numeric_cols + categorical_cols
        X = df[feature_names]

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        X_transformed = preprocessor.fit_transform(X)

        assert X_transformed.shape[0] == len(X)
        # Should have at least as many columns as numeric features
        # (OHE expands categoricals)
        assert X_transformed.shape[1] >= len(numeric_cols)
