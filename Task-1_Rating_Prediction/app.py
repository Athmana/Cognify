"""
Cognify – Task 1: Restaurant Rating Prediction Web App
Streamlit web app for predicting restaurant ratings.
Run: streamlit run Task-1_Rating_Prediction/app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Append directory to sys.path for local module import
sys.path.insert(0, os.path.dirname(__file__))
from rating_prediction import (
    run_pipeline, load_data, FEATURE_COLS, TARGET_COL, DEFAULT_DATA_PATH
)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Task 1: Rating Prediction",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

ACCENT = "#3B82F6"
ACCENT_SOFT = "#DBEAFE"
GRADIENT = "linear-gradient(135deg, #0F172A 0%, #3B82F6 100%)"

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
    .pred-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E2E8F0; border-top: 4px solid __ACCENT__;
        border-radius: 12px; padding: 1rem 1.2rem; text-align: center;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
    }
    .pred-model { font-size: 0.85rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.04em; }
    .pred-value { font-size: 2.2rem; font-weight: 800; color: #0F172A; margin: 0.2rem 0; }
    .pred-note { font-size: 0.8rem; color: #94A3B8; }
    a[href*="localhost"] { text-decoration: none; }
</style>
""".replace("__ACCENT__", ACCENT).replace("__GRADIENT__", GRADIENT)
st.markdown(_CSS, unsafe_allow_html=True)

# ── Header Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <div class="header-title">⭐ Task 1: Restaurant Rating Prediction</div>
    <div class="header-sub">Predict aggregate restaurant ratings using trained Machine Learning regression models (Linear Regression, Decision Tree, Random Forest).</div>
    <span class="header-chip">🧠 Linear Regression</span>
    <span class="header-chip">🌲 Decision Tree</span>
    <span class="header-chip">🌳 Random Forest</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Setup ─────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div class="side-brand">
    <div class="side-brand-title">⭐ Rating Predictor</div>
    <div class="side-brand-sub">Cognify · Task 1</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("⚙️ Configuration")

with st.sidebar.expander("⚙️ Advanced Data Settings", expanded=False):
    data_path = st.text_input("Dataset Path", value=DEFAULT_DATA_PATH)

if not os.path.exists(data_path):
    st.error(f"❌ Dataset not found at: `{data_path}`. Please verify the CSV path in sidebar settings.")
    st.stop()

# ── Model Pipeline Execution (Cached) ─────────────────────────────────────────
@st.cache_data(show_spinner="Training ML Regression Models ...")
def get_pipeline_results(path):
    return run_pipeline(path)

results = get_pipeline_results(data_path)
metrics_df    = results["metrics_df"]
importance_df = results["importance_df"]
trained       = results["trained"]
encoders      = results["encoders"]
X_test        = results["X_test"]
y_test        = results["y_test"]

raw_df = load_data(data_path)
rated_df = raw_df[raw_df[TARGET_COL] > 0]

best_model = metrics_df["R²"].idxmax()
best_r2 = metrics_df.loc[best_model, "R²"]

# ── Sidebar Summary ───────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Summary")
st.sidebar.metric("Total Records", f"{len(raw_df):,}")
st.sidebar.metric("Rated Restaurants", f"{len(rated_df):,}")
st.sidebar.metric("Avg Rating", f"{rated_df[TARGET_COL].mean():.2f} ⭐")
st.sidebar.metric("Best Model (R²)", f"{best_model} ({best_r2:.2f})")

# ── KPI Metric Cards ──────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.markdown(
    f'<div class="kpi-card"><div class="kpi-label">Rated Restaurants</div>'
    f'<div class="kpi-value">{len(rated_df):,}</div>'
    f'<div class="kpi-sub">of {len(raw_df):,} total records</div></div>',
    unsafe_allow_html=True)
k2.markdown(
    f'<div class="kpi-card"><div class="kpi-label">ML Models Trained</div>'
    f'<div class="kpi-value">{len(trained)}</div>'
    f'<div class="kpi-sub">regression families</div></div>',
    unsafe_allow_html=True)
k3.markdown(
    f'<div class="kpi-card"><div class="kpi-label">Best Model R²</div>'
    f'<div class="kpi-value">{best_r2:.3f}</div>'
    f'<div class="kpi-sub">{best_model}</div></div>',
    unsafe_allow_html=True)
k4.markdown(
    f'<div class="kpi-card"><div class="kpi-label">Feature Dimensions</div>'
    f'<div class="kpi-value">{len(FEATURE_COLS)}</div>'
    f'<div class="kpi-sub">raw + encoded inputs</div></div>',
    unsafe_allow_html=True)

# ── Navigation Tabs ────────────────────────────────────────────────────────────
tab_predict, tab_eval, tab_importance, tab_overview = st.tabs([
    "🔮 Predict Rating",
    "📊 Model Performance",
    "🔍 Feature Importance",
    "📂 Dataset Explorer"
])

# ── TAB 1: INTERACTIVE PREDICTOR ─────────────────────────────────────────────
with tab_predict:
    st.markdown('<div class="section-title">🔮 Enter Restaurant Attributes</div>', unsafe_allow_html=True)
    st.caption("Fill in restaurant details to calculate instantaneous rating predictions.")

    cities_list = sorted(raw_df["City"].dropna().unique().tolist())
    cuisines_list = sorted({c.strip() for row in raw_df["Cuisines"].dropna() for c in row.split(",")})
    country_codes = sorted(raw_df["Country Code"].unique().tolist())
    currencies = sorted(raw_df["Currency"].dropna().unique().tolist())

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("##### 📍 Location & Cuisine Info")
        input_city = st.selectbox("City", cities_list, index=0)
        input_cuisine = st.selectbox("Primary Cuisine", cuisines_list, index=0)
        input_country = st.selectbox("Country Code", country_codes, index=0)
        input_currency = st.selectbox("Currency", currencies, index=0)

    with col_right:
        st.markdown("##### 💰 Pricing, Service & Engagement")
        input_cost = st.number_input("Average Cost for Two", min_value=0, max_value=500000, value=500, step=50)
        input_price_range = st.select_slider("Price Range Tier (1 - 4)", options=[1, 2, 3, 4], value=2)
        input_votes = st.number_input("Customer Votes Count", min_value=0, max_value=50000, value=150, step=10)

        c_b1, c_b2, c_b3 = st.columns(3)
        with c_b1:
            input_booking = st.radio("Table Booking?", ["No", "Yes"])
        with c_b2:
            input_delivery = st.radio("Online Delivery?", ["No", "Yes"])
        with c_b3:
            input_delivering_now = st.radio("Delivering Now?", ["No", "Yes"])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡ Generate Rating Prediction", type="primary", use_container_width=True):
        def encode_val(col_name, val):
            le = encoders.get(col_name)
            if le and val in le.classes_:
                return le.transform([val])[0]
            return 0

        input_data = pd.DataFrame([{
            "Country Code": input_country,
            "City": encode_val("City", input_city),
            "Cuisines": encode_val("Cuisines", input_cuisine),
            "Average Cost for two": input_cost,
            "Currency": encode_val("Currency", input_currency),
            "Has Table booking": 1 if input_booking == "Yes" else 0,
            "Has Online delivery": 1 if input_delivery == "Yes" else 0,
            "Is delivering now": 1 if input_delivering_now == "Yes" else 0,
            "Price range": input_price_range,
            "Votes": input_votes,
        }])[FEATURE_COLS]

        st.markdown('<div class="section-title">🏆 Predicted Ratings</div>', unsafe_allow_html=True)
        res_cols = st.columns(len(trained))
        model_notes = {
            "Linear Regression": "Baseline · interpretable",
            "Decision Tree": "Split-based · robust",
            "Random Forest": "Ensemble · most accurate",
        }
        for r_col, (m_name, model_obj) in zip(res_cols, trained.items()):
            pred_val = float(model_obj.predict(input_data)[0])
            pred_val = max(1.0, min(5.0, pred_val))
            with r_col:
                st.markdown(
                    f'<div class="pred-card">'
                    f'<div class="pred-model">{m_name}</div>'
                    f'<div class="pred-value">{pred_val:.2f} <span style="font-size:1.2rem;">⭐</span></div>'
                    f'<div class="pred-note">{model_notes.get(m_name, "")}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown('<div class="section-title">🎯 Confidence View</div>', unsafe_allow_html=True)
        rf_pred = float(trained["Random Forest"].predict(input_data)[0])
        rf_pred = max(1.0, min(5.0, rf_pred))
        g1, g2 = st.columns([2, 3])
        with g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rf_pred,
                number={"suffix": " / 5", "font": {"size": 38}},
                title={"text": f"Random Forest Prediction", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 5], "tickwidth": 1, "tickcolor": "#94A3B8"},
                    "bar": {"color": ACCENT, "thickness": 0.28},
                    "bgcolor": "#F1F5F9",
                    "borderwidth": 2,
                    "bordercolor": "#E2E8F0",
                    "steps": [
                        {"range": [0, 2.5], "color": "#FEF3C7"},
                        {"range": [2.5, 4.0], "color": "#FDE68A"},
                        {"range": [4.0, 5.0], "color": "#BBF7D0"},
                    ],
                },
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=30, r=30, t=50, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with g2:
            st.caption("All model predictions at a glance")
            comp_data = []
            for m_name, model_obj in trained.items():
                pred = max(1.0, min(5.0, float(model_obj.predict(input_data)[0])))
                comp_data.append({"Model": m_name, "Predicted Rating": pred})
            comp_df = pd.DataFrame(comp_data)
            fig_comp = px.bar(
                comp_df, x="Model", y="Predicted Rating",
                color="Predicted Rating", color_continuous_scale=["#FCA5A5", "#F59E0B", "#10B981"],
                text_auto=".2f",
                title="Comparison Across Models",
            )
            fig_comp.update_layout(height=300, yaxis=dict(range=[1, 5]), coloraxis_showscale=False, template="plotly_white")
            st.plotly_chart(fig_comp, use_container_width=True)

# ── TAB 2: MODEL PERFORMANCE ──────────────────────────────────────────────────
with tab_eval:
    st.markdown('<div class="section-title">📊 Model Evaluation Metrics</div>', unsafe_allow_html=True)

    st.dataframe(
        metrics_df.style
            .highlight_max(subset=["R²"], color="#DCFCE7")
            .highlight_min(subset=["RMSE", "MAE", "MSE"], color="#DCFCE7")
            .format({c: "{:.4f}" for c in ["MSE", "RMSE", "MAE"]})
            .format({"R²": "{:.4f}"}),
        use_container_width=True
    )

    metric_long = metrics_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig_metrics = px.bar(
        metric_long,
        x="Metric", y="Score", color="Model", barmode="group",
        title="Metric Scores Across Regression Models",
        color_discrete_sequence=["#3B82F6", "#8B5CF6", "#10B981"],
    )
    fig_metrics.update_layout(height=350, template="plotly_white")
    st.plotly_chart(fig_metrics, use_container_width=True)

    st.markdown('<div class="section-title">🎯 Actual vs. Predicted Ratings Analysis</div>', unsafe_allow_html=True)
    selected_model_name = st.selectbox("Select Model to Inspect", list(trained.keys()), index=2)
    selected_model = trained[selected_model_name]
    y_pred = selected_model.predict(X_test)

    scatter_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    # Sample to keep the scatter readable
    if len(scatter_df) > 4000:
        scatter_df = scatter_df.sample(4000, random_state=42)
    fig_scatter = px.scatter(
        scatter_df, x="Actual", y="Predicted", opacity=0.45,
        color_discrete_sequence=["#2563EB"],
        title=f"{selected_model_name} — Actual vs. Predicted Ratings",
    )
    lo, hi = scatter_df["Actual"].min(), scatter_df["Actual"].max()
    fig_scatter.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="#EF4444", dash="dash", width=2))
    fig_scatter.update_layout(height=380, template="plotly_white")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── TAB 3: FEATURE IMPORTANCE ─────────────────────────────────────────────────
with tab_importance:
    st.markdown('<div class="section-title">🔍 Influential Feature Analysis</div>', unsafe_allow_html=True)
    tree_models = [m for m in trained.keys() if m != "Linear Regression"]
    imp_model = st.selectbox("Select Model", tree_models, index=1)

    imp_sub = importance_df[importance_df["Model"] == imp_model].sort_values("Importance", ascending=True)
    fig_imp = px.bar(
        imp_sub, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale=["#DBEAFE", "#1D4ED8"],
        title=f"Feature Importances ({imp_model})",
    )
    fig_imp.update_layout(height=max(380, 40 * len(imp_sub)), coloraxis_showscale=False, template="plotly_white")
    st.plotly_chart(fig_imp, use_container_width=True)

# ── TAB 4: DATASET EXPLORER ───────────────────────────────────────────────────
with tab_overview:
    st.markdown('<div class="section-title">📂 Dataset Sample Preview</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(15), use_container_width=True)

    d_left, d_right = st.columns(2)
    with d_left:
        fig_dist = px.histogram(
            raw_df[raw_df[TARGET_COL] > 0],
            x=TARGET_COL, nbins=25, color_discrete_sequence=["#3B82F6"],
            title="Aggregate Rating Target Distribution"
        )
        fig_dist.update_layout(height=340, template="plotly_white")
        st.plotly_chart(fig_dist, use_container_width=True)

    with d_right:
        cost_df = raw_df[raw_df["Average Cost for two"] > 0]
        fig_cost = px.box(
            cost_df.head(8000), x="Price range", y="Average Cost for two",
            color_discrete_sequence=["#8B5CF6"],
            title="Cost Distribution by Price Range"
        )
        fig_cost.update_layout(height=340, template="plotly_white")
        st.plotly_chart(fig_cost, use_container_width=True)

st.divider()
st.caption("Cognify · Task 1 – Restaurant Rating Prediction Web App")