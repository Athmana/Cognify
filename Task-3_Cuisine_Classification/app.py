"""
Cognify – Task 3: Multi-Label Cuisine Classification Web App
Multi-label cuisine classifier and interactive prediction portal.
Run: streamlit run Task-3_Cuisine_Classification/app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
from cuisine_classification import (
    run_pipeline, load_data, FEATURE_COLS, TOP_N_CUISINES, DEFAULT_DATA_PATH
)

# ── Resolve dataset path (works locally and on Streamlit Cloud) ────────────────
def _resolve_data_path(requested: str) -> str | None:
    """Return the first existing path from the candidate list, or None."""
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        requested,
        os.path.join(_script_dir, "..", "data", "Dataset.csv"),
        "/mount/src/cognify/data/Dataset.csv",
        "/mount/src/data/Dataset.csv",
        "/mount/src/cognify/QPredict/data/uploads/Dataset.csv",
        "/mount/src/QPredict/data/uploads/Dataset.csv",
    ]
    for c in candidates:
        resolved = os.path.abspath(c)
        if os.path.exists(resolved):
            return resolved
    return None

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Task 3: Cuisine Classification",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded"
)

ACCENT = "#7C3AED"
ACCENT_SOFT = "#EDE9FE"
GRADIENT = "linear-gradient(135deg, #1E293B 0%, #7C3AED 100%)"

_CSS = """
<style>
    .header-box {
        background: __GRADIENT__;
        padding: 1.5rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
    }
    .header-title { font-size: 2rem; font-weight: 800; margin: 0; }
    .header-sub { font-size: 0.95rem; opacity: 0.92; margin-top: 0.35rem; }
    .header-chip {
        display: inline-block; background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.28);
        padding: 0.2rem 0.7rem; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; margin-top: 0.6rem; margin-right: 0.4rem;
    }
    .kpi-card {
        border-radius: 12px; padding: 0.85rem 1.1rem;
        border: 1px solid #E2E8F0;
        border-left: 5px solid __ACCENT__;
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
        margin-bottom: 0.4rem;
    }
    .kpi-label { font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; font-weight: 700; }
    .kpi-value { font-size: 1.5rem; font-weight: 800; color: #0F172A; margin-top: 0.1rem; }
    .kpi-sub { font-size: 0.76rem; color: #94A3B8; }
    .section-title { font-size: 1.15rem; font-weight: 800; color: #0F172A; margin: 1.1rem 0 0.9rem 0; }
    .side-brand {
        background: __GRADIENT__; color: white; padding: 1rem 1.1rem;
        border-radius: 12px; margin-bottom: 1rem;
    }
    .side-brand-title { font-size: 1.05rem; font-weight: 800; }
    .side-brand-sub { font-size: 0.75rem; opacity: 0.9; margin-top: 0.2rem; }
    .tag-badge {
        display: inline-block; background-color: #F3E8FF;
        color: #6B21A8; padding: 0.5rem 1rem; border-radius: 999px;
        font-weight: 800; font-size: 0.95rem;
        margin: 0.25rem 0.5rem 0.25rem 0;
        border: 1px solid #E9D5FF;
    }
    .top10-grid { margin-top: 1rem; }
</style>
""".replace("__ACCENT__", ACCENT).replace("__GRADIENT__", GRADIENT)
st.markdown(_CSS, unsafe_allow_html=True)

# ── Header Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <div class="header-title">🍜 Task 3: Multi-Label Cuisine Classification</div>
    <div class="header-sub">Multi-Label OneVsRest Classification (Logistic Regression & Random Forest) to predict top cuisine tags.</div>
    <span class="header-chip">⚔️ OneVsRest</span>
    <span class="header-chip">📏 StandardScaler</span>
    <span class="header-chip">🏷️ 10 Cuisine Tags</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Configuration ─────────────────────────────────────────────────────
st.sidebar.markdown("""
<div class="side-brand">
    <div class="side-brand-title">🍜 Cuisine Classifier</div>
    <div class="side-brand-sub">Cognify · Task 3</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("⚙️ Configuration")

with st.sidebar.expander("⚙️ Advanced Data Settings", expanded=False):
    data_path = st.text_input("Dataset Path", value=DEFAULT_DATA_PATH)

resolved_path = _resolve_data_path(data_path)
if resolved_path is None:
    st.error(f"❌ Dataset not found. Searched common locations including `{data_path}`")
    st.stop()

@st.cache_data(show_spinner="Training Multi-Label Classifiers ...")
def get_pipeline_results(path):
    return run_pipeline(path)

results       = get_pipeline_results(resolved_path)
overall_df    = results["overall_df"]
per_label_dfs = results["per_label_dfs"]
trained       = results["trained"]
scaler        = results["scaler"]
encoders      = results["encoders"]
mlb           = results["mlb"]
top_cuisines  = results["top_cuisines"]

raw_df = load_data(resolved_path)

best_model = overall_df["F1 (micro)"].idxmax()
best_f1 = overall_df.loc[best_model, "F1 (micro)"]
best_ham = overall_df["Hamming Loss"].min()

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Summary")
st.sidebar.metric("Target Cuisine Tags", TOP_N_CUISINES)
st.sidebar.metric("Best Model (F1 micro)", f"{best_model} ({best_f1:.2f})")
st.sidebar.metric("Best Hamming Loss", f"{best_ham:.4f}")

# ── KPI Metric Cards ──────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.markdown(
    f'<div class="kpi-card"><div class="kpi-label">Target Cuisine Tags</div>'
    f'<div class="kpi-value">{TOP_N_CUISINES}</div>'
    f'<div class="kpi-sub">multi-label classes</div></div>',
    unsafe_allow_html=True)
k2.markdown(
    f'<div class="kpi-card"><div class="kpi-label">Best Model F1 (micro)</div>'
    f'<div class="kpi-value">{best_f1:.3f}</div>'
    f'<div class="kpi-sub">{best_model}</div></div>',
    unsafe_allow_html=True)
k3.markdown(
    f'<div class="kpi-card"><div class="kpi-label">Best F1 (macro)</div>'
    f'<div class="kpi-value">{overall_df["F1 (macro)"].max():.3f}</div>'
    f'<div class="kpi-sub">{overall_df["F1 (macro)"].idxmax()}</div></div>',
    unsafe_allow_html=True)
k4.markdown(
    f'<div class="kpi-card"><div class="kpi-label">Best Hamming Loss</div>'
    f'<div class="kpi-value">{best_ham:.4f}</div>'
    f'<div class="kpi-sub">lower is better</div></div>',
    unsafe_allow_html=True)

# ── Navigation Tabs ────────────────────────────────────────────────────────────
tab_predict, tab_eval, tab_cuisines = st.tabs([
    "🔮 Predict Cuisines",
    "📊 Model Performance",
    "🍽️ Cuisine Breakdown & Bias"
])

# ── TAB 1: INTERACTIVE PREDICTOR ─────────────────────────────────────────────
with tab_predict:
    st.markdown('<div class="section-title">🔮 Predict Cuisine Tags</div>', unsafe_allow_html=True)
    st.caption(f"Input restaurant details to predict applicable tags among top {TOP_N_CUISINES} cuisines.")

    cities_list = sorted(raw_df["City"].dropna().unique().tolist())
    country_codes = sorted(raw_df["Country Code"].unique().tolist())
    currencies = sorted(raw_df["Currency"].dropna().unique().tolist())

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("##### 📍 Location & Pricing")
        inp_city = st.selectbox("City", cities_list, index=0)
        inp_country = st.selectbox("Country Code", country_codes, index=0)
        inp_cost = st.number_input("Average Cost for Two", min_value=0, max_value=500000, value=600, step=50)
        inp_price_range = st.select_slider("Price Range (1 - 4)", options=[1, 2, 3, 4], value=2)
        inp_currency = st.selectbox("Currency", currencies, index=0)

    with c_right:
        st.markdown("##### ⭐ Rating & Services")
        inp_rating = st.slider("Aggregate Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
        inp_votes = st.number_input("Votes Count", min_value=0, max_value=50000, value=250, step=10)

        st.markdown("<br>", unsafe_allow_html=True)
        cb1, cb2, cb3 = st.columns(3)
        with cb1:
            inp_booking = st.radio("Table Booking?", ["No", "Yes"])
        with cb2:
            inp_delivery = st.radio("Online Delivery?", ["No", "Yes"])
        with cb3:
            inp_delivering_now = st.radio("Delivering Now?", ["No", "Yes"])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡ Predict Cuisine Tags", type="primary", use_container_width=True):
        def encode_val(col_name, val):
            le = encoders.get(col_name)
            if le and val in le.classes_:
                return le.transform([val])[0]
            return 0

        row_dict = {
            "Country Code": inp_country,
            "City": encode_val("City", inp_city),
            "Average Cost for two": inp_cost,
            "Currency": encode_val("Currency", inp_currency),
            "Has Table booking": 1 if inp_booking == "Yes" else 0,
            "Has Online delivery": 1 if inp_delivery == "Yes" else 0,
            "Is delivering now": 1 if inp_delivering_now == "Yes" else 0,
            "Price range": inp_price_range,
            "Aggregate rating": inp_rating,
            "Votes": inp_votes,
        }

        input_unscaled = pd.DataFrame([row_dict])[FEATURE_COLS]
        input_scaled = pd.DataFrame(scaler.transform(input_unscaled), columns=FEATURE_COLS)

        st.markdown('<div class="section-title">🏷️ Model Predictions</div>', unsafe_allow_html=True)
        model_colors = {
            "Logistic Regression": ("#3B82F6", "#DBEAFE", "#1D4ED8"),
            "Random Forest": ("#7C3AED", "#EDE9FE", "#6B21A8"),
        }
        res_cols = st.columns(len(trained))
        for r_col, (m_name, model_obj) in zip(res_cols, trained.items()):
            binary_pred = model_obj.predict(input_scaled)
            pred_labels = mlb.inverse_transform(binary_pred)[0]
            base_color, bg_color, fg_color = model_colors.get(m_name, (ACCENT, ACCENT_SOFT, "#4C1D95"))
            with r_col:
                st.markdown(
                    f'<div style="border:1px solid #E2E8F0;border-top:4px solid {base_color};'
                    f'border-radius:12px;padding:1rem 1.2rem;background:#F8FAFC;'
                    f'box-shadow:0 2px 8px rgba(15,23,42,0.06);">'
                    f'<div style="font-size:0.85rem;font-weight:800;color:#64748B;'
                    f'text-transform:uppercase;letter-spacing:0.04em;margin-bottom:0.6rem;">{m_name}</div>',
                    unsafe_allow_html=True
                )
                if pred_labels:
                    for tag in pred_labels:
                        st.markdown(f'<span class="tag-badge">🍜 {tag}</span>', unsafe_allow_html=True)
                else:
                    st.info("No primary cuisine tags triggered.")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">⚖️ Tag Coverage Comparison</div>', unsafe_allow_html=True)
        cov_rows = []
        for m_name, model_obj in trained.items():
            binary_pred = model_obj.predict(input_scaled)
            pred_labels = mlb.inverse_transform(binary_pred)[0]
            cov_rows.append({"Model": m_name, "Predicted Tags": len(pred_labels), "Cuisines": ", ".join(pred_labels) if pred_labels else "—"})
        cov_df = pd.DataFrame(cov_rows)
        st.dataframe(cov_df, use_container_width=True)

# ── TAB 2: OVERALL PERFORMANCE ────────────────────────────────────────────────
with tab_eval:
    st.markdown('<div class="section-title">📊 Classifier Benchmark Metrics</div>', unsafe_allow_html=True)

    st.dataframe(
        overall_df.style
            .highlight_max(subset=["F1 (micro)", "F1 (macro)", "Precision (micro)", "Recall (micro)", "Subset Accuracy"], color="#DCFCE7")
            .highlight_min(subset=["Hamming Loss"], color="#DCFCE7"),
        use_container_width=True
    )

    metric_cols = ["Precision (micro)", "Recall (micro)", "F1 (micro)", "F1 (macro)"]
    fig_ov = px.bar(
        overall_df[metric_cols].reset_index().melt(id_vars="Model"),
        x="variable", y="value", color="Model", barmode="group",
        title="Multi-Label Classification Metric Comparison",
        labels={"variable": "Metric", "value": "Score"},
        color_discrete_sequence=["#3B82F6", "#7C3AED"]
    )
    fig_ov.update_layout(height=350, template="plotly_white")
    st.plotly_chart(fig_ov, use_container_width=True)

# ── TAB 3: PER-CUISINE BREAKDOWN & BIAS ───────────────────────────────────────
with tab_cuisines:
    st.markdown('<div class="section-title">🍽️ Per-Cuisine Performance & Class Bias</div>', unsafe_allow_html=True)
    sel_model = st.selectbox("Select Model", list(per_label_dfs.keys()), index=1)
    per_df = per_label_dfs[sel_model].reset_index()

    l_col, r_col = st.columns(2)
    with l_col:
        fig_f1 = px.bar(
            per_df.sort_values("F1", ascending=True),
            x="F1", y="Cuisine", orientation="h",
            color="F1", color_continuous_scale=["#EDE9FE", "#6D28D9"],
            title=f"{sel_model} — F1 Score per Cuisine Tag"
        )
        fig_f1.update_layout(height=400, coloraxis_showscale=False, template="plotly_white")
        st.plotly_chart(fig_f1, use_container_width=True)

    with r_col:
        fig_support = px.bar(
            per_df.sort_values("Support", ascending=True),
            x="Support", y="Cuisine", orientation="h",
            color="Support", color_continuous_scale=["#FDE68A", "#D97706"],
            title=f"{sel_model} — Sample Support (class imbalance)"
        )
        fig_support.update_layout(height=400, coloraxis_showscale=False, template="plotly_white")
        st.plotly_chart(fig_support, use_container_width=True)

    st.markdown("#### ⚠️ Class Imbalance Analysis")
    rf_per = per_label_dfs["Random Forest"].reset_index()
    low_f1 = rf_per[rf_per["F1"] < 0.4].sort_values("F1")
    high_f1 = rf_per[rf_per["F1"] >= 0.4].sort_values("F1", ascending=False)

    l_col, h_col = st.columns(2)
    with l_col:
        st.markdown("**Lower Support Cuisines (F1 < 0.40)**")
        for _, r in low_f1.iterrows():
            st.warning(f"**{r['Cuisine']}** — F1: {r['F1']:.3f} (Support: {r['Support']})")

    with h_col:
        st.markdown("**Dominant Cuisines (F1 ≥ 0.40)**")
        for _, r in high_f1.iterrows():
            st.success(f"**{r['Cuisine']}** — F1: {r['F1']:.3f} (Support: {r['Support']})")

st.divider()
st.caption("Cognify · Task 3 – Cuisine Classification Project")