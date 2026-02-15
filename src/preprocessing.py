"""
Preprocessing Pipeline
======================
Feature engineering, selection, and sklearn ColumnTransformer pipeline.

Usage:
    python -m src.preprocessing              # Quick sanity check
    python -m src.preprocessing --summary    # Show feature summary table
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Setup ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Columns to drop ─────────────────────────────────────────────
# IDs, free-text, high-cardinality, and leaky / redundant columns
DROP_COLUMNS = [
    # IDs — no predictive value
    "customer_id",
    "order_id",
    "order_item_id",
    "order_customer_id",
    "order_item_cardprod_id",
    "product_card_id",
    "category_id",
    "department_id",
    "product_category_id",
    # High-cardinality text — would explode OHE
    "customer_city",
    "customer_zipcode",
    "order_city",
    "order_country",
    "order_state",
    "customer_state",
    "product_name",
    # Raw dates — we extract features from these instead
    "order_date",
    "shipping_date",
    # Redundant / near-duplicate of other columns
    "order_profit_per_order",    # very similar to profit_per_order
    "order_item_total_amount",   # derivable from price × quantity
]

# Target column
TARGET = "label"


# ── Feature Engineering ──────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from raw columns before dropping them.

    New features:
        - order_dayofweek, order_month, order_quarter  (from order_date)
        - shipping_dayofweek, shipping_month           (from shipping_date)
        - shipping_lead_days                           (shipping_date − order_date)
        - discount_ratio    = discount / product_price
        - profit_margin     = profit_per_order / sales_per_customer
    """
    df = df.copy()

    # ── Date features ────────────────────────────────────────────
    for date_col, prefix in [("order_date", "order"), ("shipping_date", "shipping")]:
        if date_col in df.columns:
            dt = pd.to_datetime(df[date_col], errors="coerce", utc=True)
            df[f"{prefix}_dayofweek"] = dt.dt.dayofweek          # 0=Mon … 6=Sun
            df[f"{prefix}_month"] = dt.dt.month
            if prefix == "order":
                df["order_quarter"] = dt.dt.quarter

            # Store parsed datetime temporarily for lead-time calc
            df[f"__{date_col}_parsed"] = dt

    # Shipping lead time in days
    if "__shipping_date_parsed" in df.columns and "__order_date_parsed" in df.columns:
        df["shipping_lead_days"] = (
            df["__shipping_date_parsed"] - df["__order_date_parsed"]
        ).dt.total_seconds() / 86400
    # Drop temp columns
    df.drop(columns=[c for c in df.columns if c.startswith("__")], inplace=True)

    # ── Ratio features ───────────────────────────────────────────
    if "order_item_discount" in df.columns and "product_price" in df.columns:
        df["discount_ratio"] = df["order_item_discount"] / df["product_price"].replace(0, np.nan)

    if "profit_per_order" in df.columns and "sales_per_customer" in df.columns:
        df["profit_margin"] = df["profit_per_order"] / df["sales_per_customer"].replace(0, np.nan)

    logger.info("✅ Engineered features: date parts, shipping_lead_days, discount_ratio, profit_margin")
    return df


# ── Feature Selection ────────────────────────────────────────────
def get_feature_columns(df: pd.DataFrame):
    """
    Return (numeric_cols, categorical_cols) after dropping IDs, target, and leaky columns.
    """
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    feature_df = df.drop(columns=cols_to_drop + [TARGET], errors="ignore")

    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info(f"Features — numeric: {len(numeric_cols)}, categorical: {len(categorical_cols)}")
    return numeric_cols, categorical_cols


# ── Preprocessor (ColumnTransformer) ─────────────────────────────
def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer:
        - Numeric:     impute (median) → StandardScaler
        - Categorical: impute (most_frequent) → OneHotEncoder
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    logger.info("✅ Built ColumnTransformer (StandardScaler + OneHotEncoder)")
    return preprocessor


# ── Full Prepare Pipeline ────────────────────────────────────────
def prepare_data(
    test_size: float = 0.2,
    random_state: int = 42,
    from_csv: bool = False,
) -> dict:
    """
    End-to-end: Load → Engineer → Select → Split → Build preprocessor.

    Args:
        test_size: Fraction of data for test split.
        random_state: Random seed for reproducibility.
        from_csv: If True, load from local CSV instead of Postgres.

    Returns:
        dict with keys:
            X_train, X_test, y_train, y_test,
            preprocessor, numeric_cols, categorical_cols, feature_names
    """
    # ── 1. Load data ─────────────────────────────────────────────
    if from_csv:
        csv_path = os.path.join("data", "Logistics Delay.csv")
        logger.info(f"Loading data from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    else:
        from src.data_loader import load_data_from_postgres
        df = load_data_from_postgres()

    logger.info(f"Raw data shape: {df.shape}")

    # ── 2. Feature engineering ───────────────────────────────────
    df = engineer_features(df)

    # ── 3. Separate target ───────────────────────────────────────
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in DataFrame")
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # ── 4. Feature selection ─────────────────────────────────────
    numeric_cols, categorical_cols = get_feature_columns(df)
    feature_names = numeric_cols + categorical_cols
    X = X[feature_names]

    logger.info(f"Selected features shape: {X.shape}")
    logger.info(f"Label distribution:\n{y.value_counts().to_string()}")

    # ── 5. Train / Test split ────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # ── 6. Build preprocessor ────────────────────────────────────
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_names": feature_names,
    }


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing Pipeline — Sanity Check")
    parser.add_argument("--summary", action="store_true", help="Show feature summary table")
    parser.add_argument("--csv", action="store_true", help="Load from CSV instead of Postgres")
    args = parser.parse_args()

    result = prepare_data(from_csv=args.csv)

    print(f"\n{'='*60}")
    print(f"  Preprocessing Complete")
    print(f"{'='*60}")
    print(f"  Train samples : {result['X_train'].shape[0]}")
    print(f"  Test  samples : {result['X_test'].shape[0]}")
    print(f"  Numeric feats : {len(result['numeric_cols'])}")
    print(f"  Categoric feats: {len(result['categorical_cols'])}")
    print(f"  Total features : {len(result['feature_names'])}")

    if args.summary:
        print(f"\n── Numeric Features ──")
        for col in result["numeric_cols"]:
            print(f"  {col}")
        print(f"\n── Categorical Features ──")
        for col in result["categorical_cols"]:
            print(f"  {col}")

    # Quick test: fit the preprocessor
    preprocessor = result["preprocessor"]
    X_train_transformed = preprocessor.fit_transform(result["X_train"])
    print(f"\n  Transformed shape: {X_train_transformed.shape}")
    print(f"  ✅ Preprocessor fit_transform successful!")
