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

# ── Custom Styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #1E293B 0%, #3B82F6 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .header-sub {
        font-size: 0.95rem;
        opacity: 0.9;
        margin-top: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <div class="header-title">⭐ Task 1: Restaurant Rating Prediction</div>
    <div class="header-sub">Predict aggregate restaurant ratings using trained Machine Learning models (Linear Regression, Decision Tree, Random Forest).</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Setup ─────────────────────────────────────────────────────────────
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

# Quick Sidebar Summary Metrics
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Summary")
st.sidebar.metric("Total Records", f"{len(raw_df):,}")
st.sidebar.metric("Best Model (R²)", "Random Forest (0.61)")

# ── Navigation Tabs ────────────────────────────────────────────────────────────
tab_predict, tab_eval, tab_importance, tab_overview = st.tabs([
    "🔮 Predict Rating",
    "📊 Model Performance",
    "🔍 Feature Importance",
    "📂 Dataset Explorer"
])

# ── TAB 1: INTERACTIVE PREDICTOR ─────────────────────────────────────────────
with tab_predict:
    st.subheader("🔮 Enter Restaurant Attributes")
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

        st.markdown("### 🏆 Predicted Ratings")
        res_cols = st.columns(len(trained))
        for r_col, (m_name, model_obj) in zip(res_cols, trained.items()):
            pred_val = model_obj.predict(input_data)[0]
            pred_val = max(1.0, min(5.0, pred_val))
            with r_col:
                st.metric(m_name, f"{pred_val:.2f} ⭐")

# ── TAB 2: MODEL PERFORMANCE ──────────────────────────────────────────────────
with tab_eval:
    st.subheader("📊 Model Evaluation Metrics")
    
    st.dataframe(
        metrics_df.style
            .highlight_max(subset=["R²"], color="#DCFCE7")
            .highlight_min(subset=["RMSE", "MAE", "MSE"], color="#DCFCE7"),
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

    st.subheader("🎯 Actual vs. Predicted Ratings Analysis")
    selected_model_name = st.selectbox("Select Model to Inspect", list(trained.keys()), index=2)
    selected_model = trained[selected_model_name]
    y_pred = selected_model.predict(X_test)
    
    scatter_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    fig_scatter = px.scatter(
        scatter_df, x="Actual", y="Predicted", opacity=0.4,
        color_discrete_sequence=["#2563EB"],
        title=f"{selected_model_name} — Actual vs. Predicted Ratings",
    )
    lo, hi = scatter_df["Actual"].min(), scatter_df["Actual"].max()
    fig_scatter.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="#EF4444", dash="dash", width=2))
    fig_scatter.update_layout(height=380, template="plotly_white")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── TAB 3: FEATURE IMPORTANCE ─────────────────────────────────────────────────
with tab_importance:
    st.subheader("🔍 Influential Feature Analysis")
    tree_models = [m for m in trained.keys() if m != "Linear Regression"]
    imp_model = st.selectbox("Select Model", tree_models, index=1)
    
    imp_sub = importance_df[importance_df["Model"] == imp_model].sort_values("Importance", ascending=True)
    fig_imp = px.bar(
        imp_sub, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale=["#DBEAFE", "#1D4ED8"],
        title=f"Feature Importances ({imp_model})",
    )
    fig_imp.update_layout(height=380, coloraxis_showscale=False, template="plotly_white")
    st.plotly_chart(fig_imp, use_container_width=True)

# ── TAB 4: DATASET EXPLORER ───────────────────────────────────────────────────
with tab_overview:
    st.subheader("📂 Dataset Sample Preview")
    st.dataframe(raw_df.head(15), use_container_width=True)

    fig_dist = px.histogram(
        raw_df[raw_df[TARGET_COL] > 0],
        x=TARGET_COL, nbins=25, color_discrete_sequence=["#3B82F6"],
        title="Aggregate Rating Target Distribution"
    )
    fig_dist.update_layout(height=320, template="plotly_white")
    st.plotly_chart(fig_dist, use_container_width=True)

st.divider()
st.caption("Cognify · Task 1 – Restaurant Rating Prediction Web App")
