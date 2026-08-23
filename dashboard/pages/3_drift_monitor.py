"""
dashboard/pages/3_drift_monitor.py
====================================
Page 3 — Model drift monitoring.

Displays:
  - Population Stability Index (PSI) per feature (bar chart)
    Red alert banner if any feature PSI > 0.2
  - Feature distribution overlay: training vs recent scored transactions
    (KDE plots for top drifting features)
  - Scoring latency trend (p50 / p95 / p99)
  - Model performance over time (rolling AUPRC estimate if labels available)

PSI thresholds:
  PSI < 0.1  — Stable (green)
  PSI 0.1-0.2 — Monitor (orange)
  PSI > 0.2  — Alert, likely model drift (red)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.data_connector import get_psi_features, get_recent_scores
from dashboard.utils.model_loader import load_feature_names, load_y_test_and_probs

st.set_page_config(page_title="Drift Monitor — ARGUS", layout="wide", page_icon="📡")
st.title("📡 Model & Data Drift Monitor")
st.caption(
    "Population Stability Index (PSI) compares the distribution of scored transactions "
    "against the training distribution. PSI > 0.2 indicates significant drift and "
    "should trigger model retraining."
)

# ---- Load ----
@st.cache_data(ttl=120)
def _load_data():
    data_dict = load_y_test_and_probs()
    feat_names = load_feature_names()
    return data_dict, feat_names

data_dict, feat_names = _load_data()

X_train_path = Path(__file__).parent.parent.parent / "models" / "X_train_sample_v1.npy"
X_test_path  = Path(__file__).parent.parent.parent / "models" / "X_test_sample_v1.npy"

X_train = np.load(X_train_path) if X_train_path.exists() else None
X_test  = np.load(X_test_path)  if X_test_path.exists()  else None

# ---- PSI Computation ----
st.subheader("Population Stability Index (PSI) per Feature")

if X_train is not None and X_test is not None and feat_names:
    # Use test set as proxy for "recent" distribution (in production use live scored data)
    psi_df = get_psi_features(X_train, X_test, feat_names, n_buckets=10)

    if not psi_df.empty:
        # Global alert
        max_psi = psi_df["psi"].max()
        n_alert  = (psi_df["psi"] > 0.2).sum()
        n_warn   = ((psi_df["psi"] >= 0.1) & (psi_df["psi"] <= 0.2)).sum()

        if max_psi > 0.2:
            st.error(
                f"🔴 **DRIFT ALERT**: {n_alert} feature(s) have PSI > 0.2. "
                f"Model retraining is recommended."
            )
        elif max_psi > 0.1:
            st.warning(
                f"🟠 **Monitor**: {n_warn} feature(s) have PSI between 0.1 and 0.2. "
                f"Watch for further drift."
            )
        else:
            st.success("✅ All features stable (PSI < 0.1). No drift detected.")

        # PSI Bar Chart (top 30)
        top_psi = psi_df.head(30).copy()
        color_map = {"Stable": "#1a9641", "Minor shift": "#fdae61", "Major shift": "#d7191c"}
        fig_psi = go.Figure()
        for status, color in color_map.items():
            subset = top_psi[top_psi["status"] == status]
            if subset.empty:
                continue
            fig_psi.add_trace(go.Bar(
                x=subset["psi"], y=subset["feature"],
                orientation="h", name=status,
                marker_color=color,
            ))

        fig_psi.add_vline(x=0.1, line_dash="dot", line_color="orange",
                          annotation_text="Monitor (0.1)", annotation_position="top right")
        fig_psi.add_vline(x=0.2, line_dash="dash", line_color="red",
                          annotation_text="Alert (0.2)", annotation_position="top right")
        fig_psi.update_layout(
            title="Top 30 Features by PSI",
            xaxis_title="Population Stability Index (PSI)",
            barmode="stack",
            height=max(400, len(top_psi) * 18),
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_psi, use_container_width=True)

        # PSI Table
        with st.expander("Full PSI table"):
            st.dataframe(
                psi_df.style.applymap(
                    lambda v: "background-color: #d7191c; color: white" if v == "Major shift"
                    else ("background-color: #fdae61" if v == "Minor shift" else ""),
                    subset=["status"],
                ),
                use_container_width=True,
            )
else:
    st.info(
        "Training arrays (X_train_v1.npy) not found. "
        "Run the full pipeline first: make train"
    )

st.markdown("---")

# ---- Feature Distribution Overlay ----
st.subheader("Feature Distribution: Training vs Recent Scoring")

if X_train is not None and X_test is not None and feat_names:
    psi_df_cached = get_psi_features(X_train, X_test, feat_names)
    top_drift_features = psi_df_cached.head(6)["feature"].tolist() if not psi_df_cached.empty else feat_names[:6]

    selected_feat = st.selectbox(
        "Select feature to inspect",
        options=feat_names,
        index=feat_names.index(top_drift_features[0]) if top_drift_features and top_drift_features[0] in feat_names else 0,
    )
    feat_idx = feat_names.index(selected_feat) if selected_feat in feat_names else 0

    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        train_vals = X_train[:, feat_idx]
        fig_tr = px.histogram(
            x=train_vals[np.isfinite(train_vals)],
            nbins=50, title=f"Training distribution — {selected_feat}",
            color_discrete_sequence=["#2c7bb6"],
            labels={"x": selected_feat},
        )
        fig_tr.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
        st.plotly_chart(fig_tr, use_container_width=True)

    with col_dist2:
        test_vals = X_test[:, feat_idx]
        fig_te = px.histogram(
            x=test_vals[np.isfinite(test_vals)],
            nbins=50, title=f"Recent distribution — {selected_feat}",
            color_discrete_sequence=["#d7191c"],
            labels={"x": selected_feat},
        )
        fig_te.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
        st.plotly_chart(fig_te, use_container_width=True)

st.markdown("---")

# ---- Latency Trend ----
st.subheader("API Scoring Latency Trend")
scores_df = get_recent_scores(hours=168, limit=5000)

if not scores_df.empty and "latency_ms" in scores_df.columns and "scored_at" in scores_df.columns:
    scores_df["scored_at"] = pd.to_datetime(scores_df["scored_at"], utc=True)
    scores_df = scores_df.sort_values("scored_at")
    scores_df["hour"] = scores_df["scored_at"].dt.floor("h")

    latency_grouped = (
        scores_df.groupby("hour")["latency_ms"]
        .quantile([0.5, 0.95, 0.99])
        .unstack()
        .reset_index()
    )
    latency_grouped.columns = ["hour", "p50", "p95", "p99"]

    fig_lat = go.Figure()
    for col, color, dash in [("p50", "#1a9641", "solid"), ("p95", "#fdae61", "dash"), ("p99", "#d7191c", "dot")]:
        if col in latency_grouped.columns:
            fig_lat.add_trace(go.Scatter(
                x=latency_grouped["hour"], y=latency_grouped[col],
                mode="lines", name=col.upper(),
                line=dict(color=color, dash=dash, width=1.8),
            ))
    fig_lat.add_hline(y=100, line_dash="dot", line_color="gray",
                      annotation_text="100ms SLO", annotation_position="right")
    fig_lat.update_layout(
        title="API Scoring Latency (p50 / p95 / p99)",
        yaxis_title="Latency (ms)",
        height=300, margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_lat, use_container_width=True)
else:
    st.info("No latency data available yet. Score transactions via the API to populate this chart.")
