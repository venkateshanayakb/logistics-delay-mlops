"""
Data Loader — Neon Postgres Integration
=========================================
Load CSV data into Neon Postgres and query it back as DataFrames.

Usage:
    python -m src.data_loader          # Upload CSV → Postgres
    python -m src.data_loader --query  # Read back from Postgres
"""

import os
import argparse
import logging

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ── Setup ────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TABLE_NAME = "logistics_delays"
CSV_PATH = os.path.join("data", "Logistics Delay.csv")


# ── Database Engine ──────────────────────────────────────────────
def get_db_engine():
    """Create a SQLAlchemy engine from the DATABASE_URL env variable."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise EnvironmentError(
            "DATABASE_URL not set. Copy .env.example → .env and add your Neon connection string."
        )
    engine = create_engine(db_url, echo=False, pool_pre_ping=True)
    logger.info("Database engine created successfully.")
    return engine


# ── Upload CSV → Postgres ────────────────────────────────────────
def upload_csv_to_postgres(csv_path: str = CSV_PATH, if_exists: str = "replace") -> int:
    """
    Read the CSV file and bulk-insert all rows into Neon Postgres.

    Args:
        csv_path: Path to the Logistics Delay CSV file.
        if_exists: What to do if the table exists ('replace', 'append', 'fail').

    Returns:
        Number of rows inserted.
    """
    logger.info(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"CSV loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Clean column names: lowercase, replace spaces with underscores
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    engine = get_db_engine()

    logger.info(f"Uploading to Postgres table '{TABLE_NAME}' (if_exists='{if_exists}')...")
    df.to_sql(TABLE_NAME, engine, if_exists=if_exists, index=False, method="multi", chunksize=1000)
    logger.info(f"✅ Successfully uploaded {df.shape[0]} rows to '{TABLE_NAME}'")

    return df.shape[0]


# ── Load data from Postgres ──────────────────────────────────────
def load_data_from_postgres(limit: int = None) -> pd.DataFrame:
    """
    Query all rows from the logistics_delays table.

    Args:
        limit: Optional row limit for testing/dev. None = all rows.

    Returns:
        DataFrame with all logistics delay records.
    """
    engine = get_db_engine()
    query = f"SELECT * FROM {TABLE_NAME}"
    if limit:
        query += f" LIMIT {limit}"

    logger.info(f"Querying Postgres: {query}")
    df = pd.read_sql(query, engine)
    logger.info(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns from Postgres")
    return df


# ── Verify connection ────────────────────────────────────────────
def verify_connection() -> bool:
    """Test the database connection. Returns True if successful."""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        logger.info("✅ Database connection verified!")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


# ── Get table info ────────────────────────────────────────────────
def get_table_info() -> dict:
    """Get row count and column info from the Postgres table."""
    engine = get_db_engine()
    with engine.connect() as conn:
        row_count = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
        columns = conn.execute(
            text(
                f"SELECT column_name, data_type FROM information_schema.columns "
                f"WHERE table_name = '{TABLE_NAME}' ORDER BY ordinal_position"
            )
        ).fetchall()
    return {
        "table": TABLE_NAME,
        "row_count": row_count,
        "columns": [{"name": c[0], "type": c[1]} for c in columns],
    }


# ── CLI Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistics Delay Data Loader")
    parser.add_argument("--query", action="store_true", help="Query data from Postgres instead of uploading")
    parser.add_argument("--verify", action="store_true", help="Only verify the database connection")
    parser.add_argument("--info", action="store_true", help="Show table info (row count, columns)")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows when querying")
    args = parser.parse_args()

    if args.verify:
        verify_connection()
    elif args.info:
        info = get_table_info()
        print(f"\nTable: {info['table']}")
        print(f"Rows:  {info['row_count']}")
        print(f"Columns ({len(info['columns'])}):")
        for col in info["columns"]:
            print(f"  - {col['name']} ({col['type']})")
    elif args.query:
        data = load_data_from_postgres(limit=args.limit)
        print(f"\nShape: {data.shape}")
        print(f"\nFirst 5 rows:\n{data.head()}")
        print(f"\nLabel distribution:\n{data['label'].value_counts()}")
    else:
        # Default: upload CSV to Postgres
        rows = upload_csv_to_postgres()
        print(f"\nDone! {rows} rows uploaded to Neon Postgres.")
        # Verify round-trip
        info = get_table_info()
        print(f"Verification: {info['row_count']} rows in table '{info['table']}'")
