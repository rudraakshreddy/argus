"""
dashboard/pages/1_overview.py
==============================
Page 1 — Live transaction overview.

Displays:
  - KPI cards: total scored, flagged, flag rate, cost saved, median latency
  - Sortable table: last 50 flagged transactions with risk level colour coding
  - Hourly fraud flag rate sparkline (last 7 days)
  - Auto-refresh every 30 seconds (configurable)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.data_connector import (
    get_hourly_flag_rates,
    get_kpi_summary,
    get_recent_scores,
)

st.set_page_config(page_title="Overview — Fraud Risk Engine", layout="wide", page_icon="📊")
st.title("📊 Live Transaction Overview")

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto_refresh:
    import time
    st.sidebar.caption(f"Last refresh: {pd.Timestamp.now().strftime('%H:%M:%S')}")

hours_window = st.sidebar.slider("Time window (hours)", 1, 168, 24)

# ---- KPI Cards ----
st.subheader("Key Performance Indicators")
kpis = get_kpi_summary(hours=hours_window)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Transactions Scored", f"{kpis['total_scored']:,}")
c2.metric(
    "Flagged as Fraud",
    f"{kpis['n_flagged']:,}",
    delta=f"{kpis['flag_rate']*100:.2f}%",
    delta_color="inverse",
)
c3.metric("Flag Rate", f"{kpis['flag_rate']*100:.3f}%")
c4.metric("Median API Latency", f"{kpis['median_latency_ms']:.1f} ms",
          delta_color="off")
c5.metric("Est. Cost Saved", f"",
          delta_color="normal")

st.markdown("---")

# ---- Hourly Flag Rate Sparkline ----
st.subheader("Hourly Fraud Flag Rate")
hourly = get_hourly_flag_rates(days=max(1, hours_window // 24))

if not hourly.empty:
    fig_spark = px.area(
        hourly,
        x="hour",
        y="flag_rate",
        labels={"hour": "Time", "flag_rate": "Flag Rate"},
        color_discrete_sequence=["#d7191c"],
    )
    fig_spark.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        xaxis_title=None,
    )
    fig_spark.add_hline(
        y=kpis["flag_rate"],
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Mean {kpis['flag_rate']*100:.3f}%",
        annotation_position="bottom right",
    )
    st.plotly_chart(fig_spark, use_container_width=True)
else:
    st.info("No scoring data available yet. Run the pipeline and score some transactions.")

st.markdown("---")

# ---- Recent Flagged Transactions Table ----
st.subheader("Recent Flagged Transactions")
scores_df = get_recent_scores(hours=hours_window, limit=500)

if scores_df.empty:
    st.info("No scored transactions found in the database.")
else:
    flagged = scores_df[scores_df["is_flagged"] == 1].copy()

    if flagged.empty:
        st.success(f"✅ No fraud flagged in the last {hours_window} hours.")
    else:
        # Risk level colouring
        def risk_badge(prob: float) -> str:
            if prob >= 0.85: return "🔴 CRITICAL"
            if prob >= 0.60: return "🟠 HIGH"
            if prob >= 0.30: return "🟡 MEDIUM"
            return "🟢 LOW"

        flagged["Risk"] = flagged["fraud_prob"].apply(risk_badge)
        flagged["Fraud Prob"] = flagged["fraud_prob"].apply(lambda x: f"{x:.4f}")
        flagged["Latency"] = flagged["latency_ms"].apply(lambda x: f"{x:.1f} ms") if "latency_ms" in flagged.columns else "—"

        display_cols = ["TransactionID", "Risk", "Fraud Prob", "Latency", "scored_at", "model_name"]
        display_cols = [c for c in display_cols if c in flagged.columns]

        st.dataframe(
            flagged[display_cols].head(50).reset_index(drop=True),
            use_container_width=True,
            column_config={
                "Fraud Prob": st.column_config.ProgressColumn(
                    "Fraud Probability",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f",
                ),
            },
        )
        st.caption(f"Showing {min(50, len(flagged))} of {len(flagged)} flagged transactions.")

if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()
