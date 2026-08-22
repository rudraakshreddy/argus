<div align="center">

# 🛡️ ARGUS
### Adaptive Real-time Grading & Unsupervised Scoring for Transaction Fraud Detection

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Community%20Cloud-FF4B4B.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docs.docker.com/compose/)
[![Airflow](https://img.shields.io/badge/Apache%20Airflow-2.9-017CEE.svg)](https://airflow.apache.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)

**Named after the all-seeing hundred-eyed giant of Greek mythology — ARGUS never sleeps.**

[Live Dashboard](https://share.streamlit.io) · [API Docs](http://localhost:8000/docs) · [Report (PDF)](report/main.pdf)

</div>

---

## Overview

ARGUS is a production-grade, seven-layer fraud detection system built on the
[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
dataset (590,540 transactions, 3.5% fraud prevalence).

It benchmarks **four models** — calibrated Logistic Regression, XGBoost,
Isolation Forest, and a PyTorch Autoencoder — with full scientific rigour:
bootstrap confidence intervals, Wilcoxon significance tests, and decision
thresholds selected by **Expected Cost minimisation** rather than the
arbitrary θ = 0.5 convention.

The full system spans every layer a real data or analytics team operates across:

| Layer | Technology |
|---|---|
| **Ingestion** | IEEE-CIS CSVs → SQLite (chunked, idempotent ETL) |
| **Storage** | SQLite with normalised 4-table schema |
| **Processing** | sklearn Pipeline: log-amount, cyclical time, PCA, SMOTE, leak-free target encoding |
| **Modelling** | XGBoost (Optuna 100-trial) · LR (calibrated) · Isolation Forest · Autoencoder (PyTorch) |
| **Evaluation** | AUPRC primary · Bootstrap 95% CIs · Wilcoxon test · Cost curves |
| **Serving** | FastAPI + Docker (multi-stage, non-root) with SHAP explanations |
| **Automation** | Apache Airflow DAGs for weekly ingestion + AUPRC-gated retraining |

---

## Results (Preview)

> Full benchmark table with bootstrap CIs generated after training run — see `models/benchmark_summary.json`.

| Model | Type | AUPRC | AUROC | MCC | Cost/txn |
|---|---|---|---|---|---|
| XGBoost | Supervised | — [—, —] | — | — | — |
| Logistic Regression | Supervised | — [—, —] | — | — | — |
| Isolation Forest | Unsupervised | — [—, —] | — | — | — |
| Autoencoder | Unsupervised | — [—, —] | — | — | — |

*Run `make run-pipeline` to populate actual values.*

---

## Project Structure

```
argus/
├── ingestion/                  # Data ingestion from IEEE-CIS CSVs → SQLite
│   ├── load_ieee_cis.py        # Merge transaction + identity, chunked insert
│   └── synthetic_generator.py  # Configurable synthetic fraud generator
│
├── db/                         # Database layer
│   ├── schema.sql              # 4-table DDL (transactions, labels, scores, accounts)
│   └── init_db.py              # SQLite initialisation
│
├── processing/                 # Feature engineering pipeline
│   ├── eda.py                  # 6 publication-quality EDA figures
│   ├── feature_engineering.py  # 6 sklearn Transformers (log-amt, cyclical, PCA, ...)
│   ├── imbalance_handler.py    # 5-strategy SMOTE comparison
│   └── pipeline.py             # Full sklearn Pipeline → serialised joblib
│
├── modeling/
│   ├── supervised/
│   │   ├── model_selection.py  # Shared utilities: metrics, CI, threshold, SHAP
│   │   ├── logistic_regression.py  # Calibrated LR + Optuna 50-trial
│   │   └── xgboost_model.py    # XGBoost + Optuna 100-trial + TreeExplainer SHAP
│   ├── unsupervised/
│   │   ├── isolation_forest.py # IF grid search + normalised anomaly scores
│   │   └── autoencoder.py      # PyTorch AE (train on legit only) + early stopping
│   └── cost_analysis.py        # E[Cost](θ) curves + sensitivity analysis
│
├── evaluation/
│   ├── metrics.py              # Full metric suite + Wilcoxon test for all 4 models
│   ├── plots.py                # ROC, PR, confusion matrix, calibration (PDF)
│   └── model_comparison.py     # LaTeX benchmark table + JSON summary
│
├── serving/
│   ├── api/
│   │   ├── schema.py           # Pydantic v2 request/response models
│   │   ├── predictor.py        # Model registry, inference, SHAP, audit log
│   │   ├── middleware.py       # Request ID, access log, Prometheus metrics
│   │   └── main.py             # FastAPI app (5 endpoints)
│   ├── Dockerfile              # Multi-stage, non-root, HEALTHCHECK
│   └── docker-compose.yml      # API + Dashboard + Airflow (3 services)
│
├── dashboard/
│   ├── app.py                  # Streamlit entry point (Community Cloud deployable)
│   ├── pages/
│   │   ├── 1_overview.py       # KPI cards, flagged tx feed, hourly sparkline
│   │   ├── 2_model_performance.py  # ROC/PR, confusion matrix, SHAP, benchmark table
│   │   └── 3_drift_monitor.py  # PSI per feature, distribution overlay, latency trend
│   ├── utils/
│   │   ├── model_loader.py     # Self-contained artifact loader (no API dependency)
│   │   └── data_connector.py   # SQLite + CSV fallback for Cloud
│   └── requirements.txt        # Pinned — for Streamlit Community Cloud
│
├── automation/
│   └── dags/
│       ├── ingest_dag.py       # Weekly ingestion DAG (Sunday 02:00 UTC)
│       └── retrain_dag.py      # Weekly retrain DAG (Monday 03:00 UTC) + AUPRC gate
│
├── tests/
│   ├── test_pipeline.py        # Unit tests: transformers, leakage, serialisation
│   ├── test_metrics.py         # Unit tests: AUPRC, CI, threshold, Wilcoxon
│   └── test_api.py             # Integration tests: all 5 FastAPI endpoints
│
├── report/
│   ├── main.tex                # Elsevier elsarticle master document
│   ├── references.bib          # 18 BibTeX entries (all cited papers)
│   └── chapters/               # 7 LaTeX chapters (intro → conclusion)
│
├── deployment/
│   └── streamlit_cloud_config.toml
├── .streamlit/config.toml      # Dark theme + server config
├── requirements.txt            # Full pinned environment
├── pyproject.toml              # Project metadata
├── Makefile                    # One-command workflows
└── .env.example                # Environment variable template
```

---

## Quickstart

### Prerequisites
- Python 3.11+
- Docker Desktop
- IEEE-CIS dataset CSVs in `data/raw/`
  - `train_transaction.csv`
  - `train_identity.csv`

### 1 — Install dependencies

```bash
git clone https://github.com/rudraakshreddy/argus.git
cd argus
pip install -r requirements.txt
```

### 2 — Place data and initialise database

```bash
# Copy IEEE-CIS CSVs to data/raw/
python db/init_db.py
python ingestion/load_ieee_cis.py
```

### 3 — Run the full ML pipeline

```bash
make run-pipeline
```

This executes in sequence:
1. EDA figures → `report/figures/eda/`
2. Feature pipeline → `models/feature_pipeline_v1.joblib`
3. Imbalance strategy comparison
4. Train all 4 models (XGBoost, LR, IF, Autoencoder)
5. Cost curve analysis → `report/figures/models/`
6. Evaluation metrics + LaTeX benchmark table → `models/all_metrics.json`
7. All evaluation plots → `report/figures/models/`

### 4 — Compile the report

```bash
make report
# Output: report/main.pdf
```

### 5 — Start all services (Docker)

```bash
make up
# API:       http://localhost:8000
# Dashboard: http://localhost:8501
# Airflow:   http://localhost:8080  (admin/admin)
```

### 6 — Run tests

```bash
make test
# Expected: 0 failures
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                     ARGUS System                    │
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │ Ingestion│───▶│  SQLite  │───▶│  Processing  │  │
│  │  (ETL)   │    │  (4 tbl) │    │  Pipeline    │  │
│  └──────────┘    └──────────┘    └──────┬───────┘  │
│                                         │           │
│                            ┌────────────▼────────┐  │
│                            │   Model Training    │  │
│                            │  XGB · LR · IF · AE │  │
│                            └────────────┬────────┘  │
│                                         │           │
│           ┌─────────────────────────────▼─────────┐ │
│           │         FastAPI Scoring API            │ │
│           │  POST /score · POST /score/batch      │ │
│           │  GET /health · /metrics · /model/info │ │
│           └────────────────────────────────────── ┘ │
│                          │                           │
│           ┌──────────────▼──────────────────────┐   │
│           │       Streamlit Dashboard            │   │
│           │  Overview · Performance · Drift      │   │
│           └─────────────────────────────────────┘   │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │            Apache Airflow                   │   │
│  │  argus_ingest_weekly (Sun 02:00 UTC)        │   │
│  │  argus_retrain_weekly (Mon 03:00 UTC)       │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## Scientific Protocol

| Decision | Choice | Rationale |
|---|---|---|
| Primary metric | AUPRC | Invariant to class imbalance; directly measures precision-recall trade-off |
| Threshold selection | Expected Cost minimisation | Economically principled; accounts for asymmetric misclassification costs |
| Hyperparameter search | Optuna TPE (100 trials) | Bayesian efficiency vs. random/grid search |
| Confidence intervals | Bootstrap (n=1,000) | Distribution-free; valid for any metric |
| Model comparison | Wilcoxon signed-rank | Non-parametric; appropriate for paired per-sample comparison |
| Imbalance | SMOTE + class weighting | Compared 5 strategies; selected by 5-fold CV AUPRC |
| Target encoding | Leak-free cross-fitting | Prevents label leakage in high-cardinality categoricals |

**Cost parameters:** $c_{FP} = \$12.00$ (30 min analyst review at $24/hr) · $c_{FN} = \$850.00$ (median IEEE-CIS fraud transaction)

---

## API Reference

**Score a transaction:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"TransactionID": 1234, "TransactionDT": 86400, "TransactionAmt": 299.99, "ProductCD": "W", "card1": 1001}'
```

**Response:**
```json
{
  "TransactionID": 1234,
  "fraud_probability": 0.0823,
  "is_flagged": false,
  "threshold": 0.4200,
  "risk_level": "LOW",
  "model_version": "XGBoost-v1",
  "latency_ms": 12.4,
  "scored_at": "2026-08-23T00:00:00Z",
  "top_contributors": [
    {"feature": "log_amt", "shap_value": 0.0312, "feature_value": 5.703},
    {"feature": "card1_freq", "shap_value": -0.0187, "feature_value": 0.041},
    {"feature": "pca_v_0", "shap_value": 0.0144, "feature_value": -0.823}
  ]
}
```

---

## Makefile Reference

```bash
make run-pipeline    # Full ETL → Feature engineering → Train → Evaluate
make report          # Compile LaTeX report to PDF
make up              # Start API + Dashboard + Airflow (Docker Compose)
make down            # Stop all services
make test            # Run pytest suite
make clean           # Remove generated model artifacts and figures
```

---

## Citation

If you use this work, please cite:

```bibtex
@software{argus2026,
  author  = {Reddy, Rudraaksh},
  title   = {{ARGUS}: Adaptive Real-time Grading and Unsupervised Scoring
             for Transaction Fraud Detection},
  year    = {2026},
  url     = {https://github.com/rudraakshreddy/argus},
  license = {Apache-2.0}
}
```

---

## License

Copyright © 2026 Rudraaksh Reddy. Licensed under the
[Apache License 2.0](LICENSE).
