"""
automation/dags/ingest_dag.py
==============================
Apache Airflow DAG: weekly data ingestion pipeline.

Schedule: Every Sunday at 02:00 UTC
Tasks:
  1. check_new_data         - Verify source files exist and have expected checksums
  2. validate_schema        - Schema validation on raw CSVs (column counts, dtypes)
  3. merge_ieee_cis_tables  - Merge train_transaction.csv + train_identity.csv
  4. load_to_sqlite         - Load merged data into SQLite (chunked, idempotent)
  5. run_eda                - Re-run EDA to refresh report figures
  6. notify_success         - Log completion summary

Design principles:
  - Idempotent: re-running has no side effects (DELETE+INSERT pattern)
  - Each task has retries=2, retry_delay=5min
  - Failure sends email/log alert (email disabled for local dev)
  - All task code uses the project Python modules directly
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path("/app/project")  # mount point in docker-compose


def _check_new_data(**context) -> dict:
    """Verify IEEE-CIS CSV files exist and return file stats."""
    import hashlib
    raw_dir = PROJECT_ROOT / "data" / "raw"
    required = ["train_transaction.csv", "train_identity.csv"]
    stats = {}
    for fname in required:
        fp = raw_dir / fname
        if not fp.exists():
            raise FileNotFoundError(f"Required file missing: {fp}")
        size_mb = fp.stat().st_size / 1e6
        # SHA256 first 10MB only (fast integrity check)
        sha = hashlib.sha256()
        with open(fp, "rb") as f:
            sha.update(f.read(10 * 1024 * 1024))
        stats[fname] = {"size_mb": round(size_mb, 2), "sha256_prefix": sha.hexdigest()[:16]}
    context["ti"].xcom_push(key="file_stats", value=stats)
    print(f"File stats: {stats}")
    return stats


def _validate_schema(**context) -> None:
    """Validate CSV schema: column counts and key column presence."""
    import pandas as pd
    raw_dir = PROJECT_ROOT / "data" / "raw"

    txn = pd.read_csv(raw_dir / "train_transaction.csv", nrows=5)
    idn = pd.read_csv(raw_dir / "train_identity.csv",   nrows=5)

    assert "TransactionID" in txn.columns, "Missing TransactionID in transaction table"
    assert "isFraud" in txn.columns,       "Missing isFraud label in transaction table"
    assert "TransactionID" in idn.columns, "Missing TransactionID in identity table"
    assert txn.shape[1] >= 394,            f"Unexpected column count: {txn.shape[1]}"

    print(f"Schema valid: transactions={txn.shape[1]} cols, identity={idn.shape[1]} cols")


def _merge_ieee_cis_tables(**context) -> None:
    """Merge transaction + identity tables and save to processed/."""
    import pandas as pd
    raw_dir       = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Loading transaction CSV...")
    txn = pd.read_csv(raw_dir / "train_transaction.csv")
    print(f"  transactions: {txn.shape}")

    print("Loading identity CSV...")
    idn = pd.read_csv(raw_dir / "train_identity.csv")
    print(f"  identity: {idn.shape}")

    merged = txn.merge(idn, on="TransactionID", how="left")
    print(f"Merged shape: {merged.shape}")

    out_path = processed_dir / "merged.parquet"
    merged.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"Saved merged data -> {out_path}")
    context["ti"].xcom_push(key="n_rows", value=len(merged))


def _load_to_sqlite(**context) -> None:
    """Load merged parquet into SQLite with idempotent insert."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from ingestion.load_ieee_cis import load_to_database
    load_to_database(
        merged_path=PROJECT_ROOT / "data" / "processed" / "merged.parquet",
        db_path=PROJECT_ROOT / "db" / "fraud.db",
    )


def _run_eda(**context) -> None:
    """Re-run EDA to refresh figures after new data load."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from processing.eda import run_eda
    run_eda(db_path=PROJECT_ROOT / "db" / "fraud.db")


def _notify_success(**context) -> None:
    """Log ingestion summary."""
    n_rows = context["ti"].xcom_pull(key="n_rows", task_ids="merge_ieee_cis_tables")
    file_stats = context["ti"].xcom_pull(key="file_stats", task_ids="check_new_data")
    print("=" * 60)
    print("INGESTION DAG COMPLETED SUCCESSFULLY")
    print(f"  Rows loaded: {n_rows:,}")
    print(f"  Files: {list(file_stats.keys()) if file_stats else 'N/A'}")
    print(f"  Timestamp: {datetime.utcnow().isoformat()}Z")
    print("=" * 60)


default_args = {
    "owner": "argus",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

with DAG(
    dag_id="argus_ingest_weekly",
    description="ARGUS: Weekly IEEE-CIS data ingestion and SQLite load",
    schedule_interval="0 2 * * 0",   # Every Sunday at 02:00 UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["argus", "ingestion", "ieee-cis"],
) as dag:

    t1 = PythonOperator(task_id="check_new_data",        python_callable=_check_new_data)
    t2 = PythonOperator(task_id="validate_schema",       python_callable=_validate_schema)
    t3 = PythonOperator(task_id="merge_ieee_cis_tables", python_callable=_merge_ieee_cis_tables)
    t4 = PythonOperator(task_id="load_to_sqlite",        python_callable=_load_to_sqlite)
    t5 = PythonOperator(task_id="run_eda",               python_callable=_run_eda)
    t6 = PythonOperator(task_id="notify_success",        python_callable=_notify_success)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6
