"""
dashboard/pages/2_model_performance.py
=======================================
Page 2 — Model performance metrics and explainability.

Displays:
  - Benchmark comparison table (AUPRC, AUROC, MCC, F1, Brier, Cost)
    with bootstrap confidence intervals and Wilcoxon test result
  - Interactive ROC curves (Plotly)
  - Interactive Precision-Recall curves (Plotly) — PRIMARY metric
  - Confusion matrix at theta* for selected model
  - SHAP beeswarm plot (precomputed XGBoost values)
  - Calibration curves for supervised models
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from dashboard.utils.model_loader import (
    load_all_metrics,
    load_benchmark_summary,
    load_shap_values,
    load_wilcoxon_result,
    load_xgb_threshold,
    load_y_test_and_probs,
)

st.set_page_config(page_title="Model Performance — ARGUS", layout="wide", page_icon="🔬")
st.title("🔬 Model Performance & Explainability")
st.caption(
    "All metrics evaluated on a held-out test set (20% stratified split). "
    "AUPRC is the primary metric for imbalanced fraud detection. "
    "Threshold θ* selected by Expected Cost minimisation — not 0.5."
)

# ---- Load data ----
@st.cache_data(ttl=300)
def _load():
    return load_y_test_and_probs(), load_benchmark_summary(), load_all_metrics(), load_wilcoxon_result()

data_dict, summary, all_metrics, wilcoxon = _load()

MODEL_COLORS = {
    "LogisticRegression": "#1a9641",
    "XGBoost":            "#d7191c",
    "IsolationForest":    "#f4a442",
    "Autoencoder":        "#2c7bb6",
}
MODEL_DISPLAY = {
    "LogisticRegression": "Logistic Regression",
    "XGBoost":            "XGBoost",
    "IsolationForest":    "Isolation Forest",
    "Autoencoder":        "Autoencoder",
}

# ---- Section 1: Benchmark Table ----
st.subheader("📋 Benchmark Comparison Table")

if summary:
    df_summary = pd.DataFrame(summary)
    # Highlight best per column
    numeric_cols = ["AUPRC", "AUROC", "F1", "MCC", "Brier", "Cost/txn"]
    st.dataframe(
        df_summary,
        use_container_width=True,
        column_config={
            "AUPRC":    st.column_config.NumberColumn("AUPRC ↑", format="%.4f"),
            "AUROC":    st.column_config.NumberColumn("AUROC ↑", format="%.4f"),
            "F1":       st.column_config.NumberColumn("F1 ↑",    format="%.4f"),
            "MCC":      st.column_config.NumberColumn("MCC ↑",   format="%.4f"),
            "Brier":    st.column_config.NumberColumn("Brier ↓", format="%.4f"),
            "Cost/txn": st.column_config.NumberColumn("Cost/txn ↓ ($)", format="%.4f"),
            "AUPRC_CI": st.column_config.TextColumn("95% CI (AUPRC)"),
        },
    )
    if wilcoxon:
        p = wilcoxon.get("p_value", 1.0)
        ma = wilcoxon.get("model_A", "")
        mb = wilcoxon.get("model_B", "")
        sig = "✅ Significant" if p < 0.05 else "❌ Not significant"
        st.caption(
            f"**Wilcoxon signed-rank test** ({ma} vs. {mb}): "
            f"p = {p:.4f} — {sig} at α=0.05"
        )
else:
    st.info("Run python evaluation/model_comparison.py to generate benchmark results.")

st.markdown("---")

# ---- Section 2: ROC Curves ----
col_roc, col_pr = st.columns(2)

y_test = data_dict.get("y_test")

with col_roc:
    st.subheader("ROC Curves")
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="gray", width=1))
    for key, display in MODEL_DISPLAY.items():
        if key not in data_dict or y_test is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, data_dict[key])
        roc_auc_val = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{display} (AUC={roc_auc_val:.3f})",
            line=dict(color=MODEL_COLORS.get(key, "#888"), width=2),
        ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate (Recall)",
        legend=dict(x=0.4, y=0.1, font=dict(size=10)),
        height=400, margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with col_pr:
    st.subheader("Precision-Recall Curves (Primary)")
    fig_pr = go.Figure()
    if y_test is not None:
        baseline = float(y_test.mean())
        fig_pr.add_hline(y=baseline, line_dash="dot", line_color="gray",
                         annotation_text=f"No-skill ({baseline:.4f})")
    for key, display in MODEL_DISPLAY.items():
        if key not in data_dict or y_test is None:
            continue
        prec, rec, _ = precision_recall_curve(y_test, data_dict[key])
        ap = auc(rec, prec)
        fig_pr.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines", name=f"{display} (AUPRC={ap:.3f})",
            line=dict(color=MODEL_COLORS.get(key, "#888"), width=2),
        ))
    fig_pr.update_layout(
        xaxis_title="Recall", yaxis_title="Precision",
        legend=dict(x=0.01, y=0.1, font=dict(size=10)),
        height=400, margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_pr, use_container_width=True)

st.markdown("---")

# ---- Section 3: Confusion Matrix ----
st.subheader("Confusion Matrix at θ*")
selected_model = st.selectbox("Select model", list(MODEL_DISPLAY.values()))
model_key = {v: k for k, v in MODEL_DISPLAY.items()}[selected_model]

if model_key in data_dict and y_test is not None:
    # Get threshold for this model
    threshold = load_xgb_threshold() if "XGB" in model_key else 0.5
    if all_metrics:
        for k, v in all_metrics.items():
            if model_key.lower().replace(" ", "") in k.lower().replace(" ", ""):
                threshold = v.get("threshold", threshold)
                break

    y_pred = (data_dict[model_key] >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Legitimate", "Actual: Fraud"],
        columns=["Predicted: Legitimate", "Predicted: Fraud"],
    )
    col_cm, col_cm_stats = st.columns([2, 1])
    with col_cm:
        import plotly.figure_factory as ff
        fig_cm = ff.create_annotated_heatmap(
            z=cm[::-1], x=["Predicted: Legitimate", "Predicted: Fraud"],
            y=["Actual: Fraud", "Actual: Legitimate"],
            colorscale="Blues", showscale=True,
        )
        fig_cm.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_cm, use_container_width=True)
    with col_cm_stats:
        tn, fp, fn, tp = cm.ravel()
        st.metric("True Positives (caught fraud)", tp)
        st.metric("False Positives (false alarms)", fp)
        st.metric("True Negatives (correct clear)", tn)
        st.metric("False Negatives (missed fraud)", fn)
        st.metric("Threshold θ*", f"{threshold:.4f}")

st.markdown("---")

# ---- Section 4: SHAP ----
st.subheader("SHAP Feature Importance (XGBoost)")
shap_path = Path(__file__).parent.parent.parent / "report" / "figures" / "shap" / "xgb_shap_beeswarm_v1.pdf"
if shap_path.exists():
    st.info("📄 SHAP beeswarm plot generated at report/figures/shap/xgb_shap_beeswarm_v1.pdf — view in the report PDF.")
else:
    st.info("📄 Refer to the final LaTeX report for the full SHAP beeswarm plot.")

shap_summary_path = Path(__file__).parent.parent.parent / "models" / "xgb_shap_summary_v1.json"
if shap_summary_path.exists():
    import json
    shap_summary = json.loads(shap_summary_path.read_text())
    shap_df = pd.DataFrame(shap_summary)
    
    fig_shap = go.Figure(go.Bar(
        x=shap_df["Mean |SHAP|"][::-1],
        y=shap_df["Feature"][::-1],
        orientation="h",
        marker_color="#d7191c",
    ))
    fig_shap.update_layout(
        title=f"Top {len(shap_df)} Features by Mean |SHAP| Value",
        xaxis_title="Mean |SHAP value|",
        height=500, margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_shap, use_container_width=True)
else:
    st.warning("SHAP summary not found. Ensure xgb_shap_summary_v1.json is committed.")
