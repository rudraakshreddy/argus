"""
dashboard/app.py
=================
Main Streamlit entry point — multi-page fraud monitoring dashboard.

Deployable to Streamlit Community Cloud:
  - All model artifacts loaded directly (no API dependency for core pages)
  - SQLite fallback to CSV export on Cloud
  - requirements.txt in dashboard/ for Community Cloud dependency management

Pages:
  1_overview.py       - Live transaction feed, KPI cards, hourly flag rate
  2_model_performance.py - ROC/PR curves, confusion matrix, SHAP, benchmark table
  3_drift_monitor.py  - PSI per feature, distribution shift, drift alerts

Run locally:
  streamlit run dashboard/app.py

Deploy to Community Cloud:
  Repo: github.com/<user>/fraud-risk-engine
  Main file path: dashboard/app.py
"""
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on PYTHONPATH
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# ---- Page configuration ----
st.set_page_config(
    page_title="Fraud Risk Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo/fraud-risk-engine",
        "Report a bug": None,
        "About": "Fraud & Anomaly Risk-Scoring Engine — Sem 8 Major Project",
    },
)

# ---- Sidebar navigation ----
st.sidebar.title("🛡️ Fraud Risk Engine")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Navigation**
    - [📊 Overview](/)
    - [🔬 Model Performance](/model_performance)
    - [📡 Drift Monitor](/drift_monitor)
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Info**")
st.sidebar.info(
    "IEEE-CIS Fraud Detection  \n"
    "XGBoost · LR · Isolation Forest · Autoencoder  \n"
    "Threshold: Cost-optimised (not 0.5)"
)

# ---- Home page (redirect to overview) ----
# In Streamlit multi-page apps, the main app.py is the home page.
# We render a brief landing page here and let pages/ handle the rest.

st.title("🛡️ Fraud & Anomaly Risk-Scoring Engine")
st.markdown(
    r"""
    **Sem 8 Major Project** — Real-time fraud detection using ensemble ML models
    trained on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset.

    ---

    ### System Architecture
    | Layer | Technology |
    |---|---|
    | Data | IEEE-CIS (590K transactions, 3.5% fraud rate) |
    | Storage | SQLite (normalised schema) |
    | Feature Engineering | Sklearn Pipeline (log-amount, cyclical time, V-PCA, SMOTE) |
    | Models | XGBoost · Logistic Regression · Isolation Forest · Autoencoder |
    | Threshold | Cost-optimal θ* (c_FP=\$12, c_FN=\$115) |
    | Serving | FastAPI + Docker |
    | Automation | Apache Airflow (weekly retrain DAG) |

    ---
    """
)

# KPI preview on home page
from dashboard.utils.data_connector import get_kpi_summary
kpis = get_kpi_summary(hours=24)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions Scored (24h)", f"{kpis['total_scored']:,}")
col2.metric("Flagged (24h)", f"{kpis['n_flagged']:,}",
            delta=f"{kpis['flag_rate']*100:.2f}% flag rate")
col3.metric("Median Latency", f"{kpis['median_latency_ms']:.1f} ms")
col4.metric("Est. Cost Saved (24h)", f"${kpis['expected_cost_saved_usd']:,.2f}")

st.markdown("---")
st.markdown("Use the **sidebar** or the links above to navigate to detailed pages.")
st.caption("Built with Streamlit · Deployed on Streamlit Community Cloud")
