"""
db/init_db.py
=============
Initialise the SQLite database by executing schema.sql.
Run once before any ingestion or training step.

Usage:
    python db/init_db.py [--db-path PATH]
"""
import argparse
import logging
import os
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent / "fraud.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create all tables from schema.sql. Safe to re-run (uses IF NOT EXISTS)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

    log.info(f"Connecting to SQLite DB at: {db_path}")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_sql)
        conn.commit()

    # Verify tables were created
    with sqlite3.connect(db_path) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t[0] for t in tables]

    expected = {
        "engineered_features",
        "fraud_labels",
        "model_registry",
        "model_scores",
        "transactions",
    }
    missing = expected - set(table_names)
    if missing:
        raise RuntimeError(f"Schema init failed — missing tables: {missing}")

    log.info(f"Database initialised. Tables: {table_names}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialise the fraud risk SQLite database")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path(os.getenv("DB_PATH", str(DEFAULT_DB_PATH))),
        help="Path to the SQLite .db file",
    )
    args = parser.parse_args()
    init_db(args.db_path)
    print(f"[OK] Database ready at: {args.db_path.resolve()}")
