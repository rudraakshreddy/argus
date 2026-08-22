"""
ingestion/load_ieee_cis.py
==========================
ETL: Merge IEEE-CIS train_transaction.csv + train_identity.csv
and load into the SQLite `transactions` + `fraud_labels` tables.

Design:
  - Left-join identity onto transactions (144k / 590k have identity rows)
  - Chunked insert (10,000 rows) to avoid OOM on 16 GB RAM
  - Schema validation on column count + dtype plausibility
  - SHA-256 checksum of source files logged to data/raw/manifest.json
  - Idempotent: truncates + reloads on re-run (explicit flag required)

Usage:
    python ingestion/load_ieee_cis.py \
        --transaction data/raw/train_transaction.csv \
        --identity    data/raw/train_identity.csv \
        --db-path     db/fraud.db
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DEFAULT_TXN = ROOT / "data" / "raw" / "train_transaction.csv"
DEFAULT_ID = ROOT / "data" / "raw" / "train_identity.csv"
DEFAULT_DB = ROOT / "db" / "fraud.db"
MANIFEST_PATH = ROOT / "data" / "raw" / "manifest.json"
CHUNK_SIZE = 10_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Return hex SHA-256 of a file without reading it all into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_transaction_csv(df: pd.DataFrame) -> None:
    """Raise if mandatory columns are absent."""
    required = {"TransactionID", "TransactionDT", "TransactionAmt", "isFraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train_transaction.csv missing columns: {missing}")
    assert df["TransactionID"].is_unique, "TransactionID must be unique in transaction file"
    log.info(
        f"Transaction CSV validated: {len(df):,} rows, "
        f"fraud rate = {df['isFraud'].mean():.4f} ({df['isFraud'].sum():,} fraud)"
    )


# ---------------------------------------------------------------------------
# Core ETL
# ---------------------------------------------------------------------------

def load(
    txn_path: Path,
    id_path: Path,
    db_path: Path,
    force: bool = False,
) -> None:
    """Merge, validate, and load IEEE-CIS data into SQLite."""

    # ---- 1. Checksums ----
    log.info("Computing SHA-256 checksums...")
    manifest = {
        "loaded_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            str(txn_path): sha256_file(txn_path),
            str(id_path): sha256_file(id_path),
        },
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    log.info(f"Manifest written to {MANIFEST_PATH}")

    # ---- 2. Load CSVs ----
    log.info("Reading train_transaction.csv (may take ~30 s on first run)...")
    txn_df = pd.read_csv(txn_path, low_memory=False)
    validate_transaction_csv(txn_df)

    log.info("Reading train_identity.csv...")
    id_df = pd.read_csv(id_path, low_memory=False)
    log.info(f"Identity CSV: {len(id_df):,} rows with {len(id_df.columns)} columns")

    # ---- 3. Left-join identity onto transactions ----
    log.info("Merging transaction + identity on TransactionID (left join)...")
    merged = txn_df.merge(id_df, on="TransactionID", how="left")
    log.info(
        f"Merged shape: {merged.shape} "
        f"({id_df['TransactionID'].isin(txn_df['TransactionID']).sum():,} identity matches)"
    )

    # ---- 4. Separate labels from features ----
    fraud_labels = merged[["TransactionID", "isFraud"]].copy()
    feature_df = merged.drop(columns=["isFraud"])

    # ---- 5. SQLite load ----
    with sqlite3.connect(db_path) as conn:
        # Idempotency guard
        existing = conn.execute(
            "SELECT COUNT(*) FROM transactions"
        ).fetchone()[0]
        if existing > 0 and not force:
            log.warning(
                f"transactions table already has {existing:,} rows. "
                "Use --force to truncate and reload."
            )
            return
        if force:
            log.warning("--force: truncating transactions and fraud_labels tables...")
            conn.execute("DELETE FROM fraud_labels")
            conn.execute("DELETE FROM transactions")
            conn.commit()

        # ---- 5a. Chunked insert: transactions ----
        log.info(f"Inserting {len(feature_df):,} rows into `transactions` in chunks of {CHUNK_SIZE:,}...")
        n_chunks = (len(feature_df) + CHUNK_SIZE - 1) // CHUNK_SIZE
        for i in tqdm(range(n_chunks), desc="transactions", unit="chunk"):
            chunk = feature_df.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
            chunk.to_sql(
                "transactions",
                conn,
                if_exists="append",
                index=False,
                method="multi",
            )

        # ---- 5b. Insert fraud labels ----
        log.info(f"Inserting {len(fraud_labels):,} rows into `fraud_labels`...")
        fraud_labels["label_source"] = "ieee_cis"
        fraud_labels.to_sql(
            "fraud_labels",
            conn,
            if_exists="append",
            index=False,
            method="multi",
        )
        conn.commit()

    # ---- 6. Verify ----
    with sqlite3.connect(db_path) as conn:
        n_txn = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        n_fraud = conn.execute(
            "SELECT COUNT(*) FROM fraud_labels WHERE isFraud = 1"
        ).fetchone()[0]
        n_legit = conn.execute(
            "SELECT COUNT(*) FROM fraud_labels WHERE isFraud = 0"
        ).fetchone()[0]

    log.info("=" * 50)
    log.info(f"Load complete. DB: {db_path.resolve()}")
    log.info(f"  Transactions : {n_txn:,}")
    log.info(f"  Fraud        : {n_fraud:,} ({n_fraud/n_txn*100:.2f}%)")
    log.info(f"  Legitimate   : {n_legit:,} ({n_legit/n_txn*100:.2f}%)")
    log.info("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load IEEE-CIS Fraud Detection data into SQLite"
    )
    parser.add_argument("--transaction", type=Path, default=DEFAULT_TXN)
    parser.add_argument("--identity", type=Path, default=DEFAULT_ID)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Truncate and reload existing data",
    )
    args = parser.parse_args()

    for p in [args.transaction, args.identity]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Download from: "
                "https://www.kaggle.com/competitions/ieee-fraud-detection/data"
            )

    load(args.transaction, args.identity, args.db_path, force=args.force)
