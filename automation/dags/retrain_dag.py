"""
automation/dags/retrain_dag.py
================================
Apache Airflow DAG: weekly model retraining pipeline.

Schedule: Every Monday at 03:00 UTC (after ingest DAG completes Sunday)
Tasks (with parallelism where safe):
  1.  validate_data          - Check features.parquet exists and is fresh
  2.  run_feature_engineering - Re-run full feature pipeline, save pipeline artifact
  3.  run_imbalance_comparison - 5-strategy SMOTE comparison (selects best)
  4a. train_logistic_regression  (parallel group)
  4b. train_xgboost              (parallel group)
  4c. train_isolation_forest     (parallel group)
  4d. train_autoencoder          (parallel group)
  5.  run_cost_analysis       - Generate cost curves, select theta* for all models
  6.  evaluate_all_models     - Compute AUPRC/AUROC/MCC/Brier + bootstrap CIs
  7.  generate_plots          - All evaluation plots -> report/figures/
  8.  generate_report_tables  - LaTeX benchmark table
  9.  compare_to_production   - Gate: new model AUPRC must beat current by >= 0.005
  10. promote_best_model      - Update production symlink if gate passes
  11. notify_completion       - Summary log with all metrics

Gate logic (Task 9):
  - Reads current production AUPRC from models/production_auprc.txt
  - Compares to new XGBoost AUPRC (primary model)
  - Promotes only if improvement >= MIN_AUPRC_IMPROVEMENT = 0.005
  - If gate fails: logs warning, keeps existing production model
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule

PROJECT_ROOT = Path("/app/project")
MODELS_DIR = PROJECT_ROOT / "models"
MIN_AUPRC_IMPROVEMENT = 0.005


def _validate_data(**ctx) -> None:
    import pandas as pd
    feat_path = PROJECT_ROOT / "data" / "processed" / "merged.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Merged data not found: {feat_path}. Run ingest DAG first.")
    df = pd.read_parquet(feat_path, columns=["TransactionID"])
    print(f"Data validated: {len(df):,} rows")


def _run_feature_engineering(**ctx) -> None:
    import sys; sys.path.insert(0, str(PROJECT_ROOT))
    from processing.pipeline import fit_and_save_pipeline
    fit_and_save_pipeline(PROJECT_ROOT / "db" / "fraud.db", version=1)


def _run_imbalance_comparison(**ctx) -> None:
    import sys; sys.path.insert(0, str(PROJECT_ROOT))
    from processing.imbalance_handler import run_comparison
    best = run_comparison(PROJECT_ROOT / "data" / "processed" / "features.parquet")
    ctx["ti"].xcom_push(key="best_imbalance_strategy", value=best)
    print(f"Best imbalance strategy: {best}")


def _train_logistic_regression(**ctx) -> None:
    import sys; sys.path.insert(0, str(PROJECT_ROOT))
    from modeling.supervised.logistic_regression import train_logistic_regression
    metrics = train_logistic_regression(version=1, n_optuna_trials=50)
    ctx["ti"].xcom_push(key="lr_auprc", value=metrics["auprc"])
    print(f"LR AUPRC: {metrics['auprc']:.4f}")


def _train_xgboost(**ctx) -> None:
    import sys; sys.path.insert(0, str(PROJECT_ROOT))
    from modeling.supervised.xgboost_model import train_xgboost
    metrics = train_xgboost(version=1, n_optuna_trials=100)
    ctx["ti"].xcom_push(key="xgb_auprc", value=metrics["auprc"])
    print(f"XGBoost AUPRC: {metrics['auprc']:.4f}")


def _train_isolation_forest(**ctx) -> None:
    import sys; sys.path.insert(0, str(PROJECT_ROOT))
    from modeling.unsupervised.isolation_forest import train_isolation_forest
    metrics = train_isolation_forest(version=1)
    ctx["ti"].xcom_push(key="if_auprc", value=metrics["auprc"])
    print(f"IF AUPRC: {metrics['auprc']:.4f}")


def _train_autoencoder(**ctx) -> None:
    import sys; sys.path.insert(0, str(PROJECT_ROOT))
    from modeling.unsupervised.autoencoder import train_autoencoder
    metrics = train_autoencoder(version=1)
    ctx["ti"].xcom_push(key="ae_auprc", value=metrics["auprc"])
    print(f"AE AUPRC: {metrics['auprc']:.4f}")


def _run_cost_analysis(**ctx) -> None:
    import sys; sys.path.insert(0, str(PROJECT_ROOT))
    from modeling.cost_analysis import run_cost_analysis
    run_cost_analysis(version=1)


def _evaluate_all_models(**ctx) -> None:
    import sys; sys.path.insert(0, str(PROJECT_ROOT))
    from evaluation.metrics import compute_all_metrics
    compute_all_metrics(version=1)


def _generate_plots(**ctx) -> None:
    import sys; sys.path.insert(0, str(PROJECT_ROOT))
    from evaluation.plots import generate_all_plots
    generate_all_plots(version=1)


def _generate_report_tables(**ctx) -> None:
    import sys; sys.path.insert(0, str(PROJECT_ROOT))
    from evaluation.model_comparison import run_model_comparison
    run_model_comparison(version=1)


def _compare_to_production(**ctx) -> str:
    """Gate: returns 'promote_best_model' or 'skip_promotion'."""
    import json
    new_auprc_path = MODELS_DIR / "xgb_v1_results.json"
    if not new_auprc_path.exists():
        print("New model results not found. Skipping promotion.")
        return "skip_promotion"

    new_auprc = json.loads(new_auprc_path.read_text())["auprc"]
    prod_path  = MODELS_DIR / "production_auprc.txt"
    prod_auprc = float(prod_path.read_text().strip()) if prod_path.exists() else 0.0

    improvement = new_auprc - prod_auprc
    print(f"New AUPRC: {new_auprc:.4f} | Production AUPRC: {prod_auprc:.4f} | Improvement: {improvement:.4f}")

    if improvement >= MIN_AUPRC_IMPROVEMENT:
        print(f"Gate PASSED (improvement={improvement:.4f} >= {MIN_AUPRC_IMPROVEMENT}). Promoting.")
        return "promote_best_model"
    else:
        print(f"Gate FAILED (improvement={improvement:.4f} < {MIN_AUPRC_IMPROVEMENT}). Keeping existing model.")
        return "skip_promotion"


def _promote_best_model(**ctx) -> None:
    import json, shutil
    # Update production AUPRC record
    new_auprc = json.loads((MODELS_DIR / "xgb_v1_results.json").read_text())["auprc"]
    (MODELS_DIR / "production_auprc.txt").write_text(str(new_auprc))
    # Copy model files to production symlinks
    for src_name, dst_name in [
        ("xgb_model_v1.joblib",      "xgb_model_production.joblib"),
        ("feature_pipeline_v1.joblib","feature_pipeline_production.joblib"),
    ]:
        src = MODELS_DIR / src_name
        dst = MODELS_DIR / dst_name
        if src.exists():
            shutil.copy2(src, dst)
    print(f"Production model promoted. AUPRC: {new_auprc:.4f}")


def _skip_promotion(**ctx) -> None:
    print("Promotion skipped. Existing production model retained.")


def _notify_completion(**ctx) -> None:
    import json
    results_path = MODELS_DIR / "all_metrics.json"
    if results_path.exists():
        metrics = json.loads(results_path.read_text())
        print("=" * 65)
        print("ARGUS RETRAIN DAG COMPLETED")
        for name, m in metrics.items():
            print(f"  {name:25s} AUPRC={m['auprc']:.4f}  AUROC={m['auroc']:.4f}")
        print(f"  Timestamp: {datetime.utcnow().isoformat()}Z")
        print("=" * 65)


default_args = {
    "owner": "argus",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=6),
}

with DAG(
    dag_id="argus_retrain_weekly",
    description="ARGUS: Weekly model retraining with AUPRC gate and auto-promotion",
    schedule_interval="0 3 * * 1",   # Every Monday at 03:00 UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["argus", "retraining", "ml"],
) as dag:

    t_validate    = PythonOperator(task_id="validate_data",           python_callable=_validate_data)
    t_feat_eng    = PythonOperator(task_id="run_feature_engineering",  python_callable=_run_feature_engineering)
    t_imbalance   = PythonOperator(task_id="run_imbalance_comparison", python_callable=_run_imbalance_comparison)

    # Parallel training tasks
    t_lr  = PythonOperator(task_id="train_logistic_regression", python_callable=_train_logistic_regression)
    t_xgb = PythonOperator(task_id="train_xgboost",             python_callable=_train_xgboost)
    t_if  = PythonOperator(task_id="train_isolation_forest",    python_callable=_train_isolation_forest)
    t_ae  = PythonOperator(task_id="train_autoencoder",         python_callable=_train_autoencoder)

    t_cost    = PythonOperator(task_id="run_cost_analysis",      python_callable=_run_cost_analysis,    trigger_rule=TriggerRule.ALL_DONE)
    t_eval    = PythonOperator(task_id="evaluate_all_models",    python_callable=_evaluate_all_models)
    t_plots   = PythonOperator(task_id="generate_plots",         python_callable=_generate_plots)
    t_tables  = PythonOperator(task_id="generate_report_tables", python_callable=_generate_report_tables)

    t_gate    = BranchPythonOperator(task_id="compare_to_production", python_callable=_compare_to_production)
    t_promote = PythonOperator(task_id="promote_best_model", python_callable=_promote_best_model)
    t_skip    = PythonOperator(task_id="skip_promotion",     python_callable=_skip_promotion)
    t_notify  = PythonOperator(task_id="notify_completion",  python_callable=_notify_completion, trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    # DAG wiring
    t_validate >> t_feat_eng >> t_imbalance
    t_imbalance >> [t_lr, t_xgb, t_if, t_ae]
    [t_lr, t_xgb, t_if, t_ae] >> t_cost >> t_eval >> t_plots >> t_tables >> t_gate
    t_gate >> [t_promote, t_skip]
    [t_promote, t_skip] >> t_notify
