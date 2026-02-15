"""
Shared Pytest Fixtures
=======================
Reusable fixtures for all test modules.
No real database, model file, or external service required.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier


# ── Feature lists (must match api/schemas.py) ────────────────────
NUMERIC_FEATURES = [
    "profit_per_order", "sales_per_customer", "latitude", "longitude",
    "order_item_discount", "order_item_discount_rate",
    "order_item_product_price", "order_item_profit_ratio",
    "order_item_quantity", "sales", "product_price",
    "order_dayofweek", "order_month", "order_quarter",
    "shipping_dayofweek", "shipping_month",
    "shipping_lead_days", "discount_ratio", "profit_margin",
]

CATEGORICAL_FEATURES = [
    "payment_type", "category_name", "customer_country",
    "customer_segment", "department_name", "market",
    "order_region", "order_status", "shipping_mode",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ── Sample input dict (matches PredictionRequest) ────────────────
@pytest.fixture
def sample_input():
    """A valid dict matching all 28 PredictionRequest fields."""
    return {
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
    }


# ── Raw DataFrame for preprocessing tests ────────────────────────
@pytest.fixture
def sample_dataframe():
    """Synthetic DataFrame with raw columns + label for preprocessing tests."""
    n = 50
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        # Raw date columns (will be engineered)
        "order_date": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
        "shipping_date": (pd.date_range("2024-01-01", periods=n, freq="D")
                          + pd.Timedelta(days=3)).astype(str),
        # Numeric
        "profit_per_order": rng.uniform(10, 100, n),
        "sales_per_customer": rng.uniform(50, 500, n),
        "latitude": rng.uniform(-30, 50, n),
        "longitude": rng.uniform(-120, 120, n),
        "order_item_discount": rng.uniform(0, 20, n),
        "order_item_discount_rate": rng.uniform(0, 0.3, n),
        "order_item_product_price": rng.uniform(20, 300, n),
        "order_item_profit_ratio": rng.uniform(-0.5, 0.5, n),
        "order_item_quantity": rng.randint(1, 10, n),
        "sales": rng.uniform(50, 500, n),
        "product_price": rng.uniform(20, 300, n),
        # Categorical
        "payment_type": rng.choice(["DEBIT", "TRANSFER", "PAYMENT", "CASH"], n),
        "category_name": rng.choice(["Cleats", "Cardio", "Fishing"], n),
        "customer_country": rng.choice(["EE. UU.", "France", "Germany"], n),
        "customer_segment": rng.choice(["Consumer", "Corporate", "Home Office"], n),
        "department_name": rng.choice(["Fan Shop", "Apparel", "Golf"], n),
        "market": rng.choice(["LATAM", "Europe", "Pacific Asia"], n),
        "order_region": rng.choice(["Central America", "Western Europe"], n),
        "order_status": rng.choice(["COMPLETE", "CLOSED", "PENDING"], n),
        "shipping_mode": rng.choice(["Standard Class", "First Class", "Same Day"], n),
        # IDs (should be dropped)
        "customer_id": range(n),
        "order_id": range(n),
        # Target
        "label": rng.choice([-1, 0, 1], n),
    })


# ── Mock model artifact (simulates best_pipeline.joblib) ─────────
@pytest.fixture
def mock_model_artifact(sample_dataframe):
    """
    Build a real (tiny) preprocessor + RandomForest, fit on sample data,
    and return the artifact dict that ModelService.load_model() expects.
    """
    from src.preprocessing import engineer_features, get_feature_columns, build_preprocessor

    df = engineer_features(sample_dataframe)
    numeric_cols, categorical_cols = get_feature_columns(df)
    feature_names = numeric_cols + categorical_cols
    X = df[feature_names]
    y = df["label"]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_transformed = preprocessor.fit_transform(X)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_transformed, y)

    return {
        "preprocessor": preprocessor,
        "model": model,
        "model_name": "test_rf",
        "feature_names": feature_names,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "label_map": {-1: "Early", 0: "On-time", 1: "Late"},
    }


@pytest.fixture
def tmp_model_path(mock_model_artifact, tmp_path):
    """Save mock artifact to a temp .joblib file and yield its path."""
    path = tmp_path / "test_pipeline.joblib"
    joblib.dump(mock_model_artifact, path)
    return str(path)


# ── FastAPI TestClient ───────────────────────────────────────────
@pytest.fixture(scope="session")
def test_client(tmp_model_path_session):
    """
    FastAPI TestClient with the model pre-loaded.
    Uses session scope to avoid Prometheus duplicate metric registration
    that happens when api.main is re-imported.
    """
    from api.model_service import model_service
    model_service.load_model(tmp_model_path_session)

    from api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)

    yield client


@pytest.fixture(scope="session")
def tmp_model_path_session(tmp_path_factory):
    """Session-scoped version of tmp_model_path for use with test_client."""
    from src.preprocessing import engineer_features, get_feature_columns, build_preprocessor

    # Build sample data
    n = 50
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "order_date": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
        "shipping_date": (pd.date_range("2024-01-01", periods=n, freq="D")
                          + pd.Timedelta(days=3)).astype(str),
        "profit_per_order": rng.uniform(10, 100, n),
        "sales_per_customer": rng.uniform(50, 500, n),
        "latitude": rng.uniform(-30, 50, n),
        "longitude": rng.uniform(-120, 120, n),
        "order_item_discount": rng.uniform(0, 20, n),
        "order_item_discount_rate": rng.uniform(0, 0.3, n),
        "order_item_product_price": rng.uniform(20, 300, n),
        "order_item_profit_ratio": rng.uniform(-0.5, 0.5, n),
        "order_item_quantity": rng.randint(1, 10, n),
        "sales": rng.uniform(50, 500, n),
        "product_price": rng.uniform(20, 300, n),
        "payment_type": rng.choice(["DEBIT", "TRANSFER", "PAYMENT", "CASH"], n),
        "category_name": rng.choice(["Cleats", "Cardio", "Fishing"], n),
        "customer_country": rng.choice(["EE. UU.", "France", "Germany"], n),
        "customer_segment": rng.choice(["Consumer", "Corporate", "Home Office"], n),
        "department_name": rng.choice(["Fan Shop", "Apparel", "Golf"], n),
        "market": rng.choice(["LATAM", "Europe", "Pacific Asia"], n),
        "order_region": rng.choice(["Central America", "Western Europe"], n),
        "order_status": rng.choice(["COMPLETE", "CLOSED", "PENDING"], n),
        "shipping_mode": rng.choice(["Standard Class", "First Class", "Same Day"], n),
        "customer_id": range(n),
        "order_id": range(n),
        "label": rng.choice([-1, 0, 1], n),
    })

    df = engineer_features(df)
    numeric_cols, categorical_cols = get_feature_columns(df)
    feature_names = numeric_cols + categorical_cols
    X = df[feature_names]
    y = df["label"]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_transformed = preprocessor.fit_transform(X)

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_transformed, y)

    artifact = {
        "preprocessor": preprocessor,
        "model": model,
        "model_name": "test_rf",
        "feature_names": feature_names,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "label_map": {-1: "Early", 0: "On-time", 1: "Late"},
    }

    path = tmp_path_factory.mktemp("models") / "test_pipeline.joblib"
    joblib.dump(artifact, path)
    return str(path)
