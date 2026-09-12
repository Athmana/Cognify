"""
Cognify – Task 2: Restaurant Recommendation Web Application
Content-based recommendation system powered by TF-IDF & Cosine Similarity.
Run: streamlit run Task-2_Restaurant_Recommendation/app.py
"""

import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.insert(0, os.path.dirname(__file__))
from restaurant_recommendation import build_recommender, recommend, DEFAULT_DATA_PATH

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Task 2: Restaurant Recommendation",
    page_icon="🍴",
    layout="wide",
    initial_sidebar_state="expanded"
)

ACCENT = "#059669"
ACCENT_SOFT = "#D1FAE5"
GRADIENT = "linear-gradient(135deg, #0F172A 0%, #059669 100%)"

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
    .rec-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.07);
    }
    .rec-rank {
        display: inline-flex; align-items: center; justify-content: center;
        background: __ACCENT__; color: white; font-weight: 800;
        width: 30px; height: 30px; border-radius: 9px; font-size: 0.95rem;
        margin-right: 0.6rem; vertical-align: middle;
    }
    .rec-name { font-size: 1.05rem; font-weight: 800; color: #0F172A; vertical-align: middle; }
    .rec-loc { font-size: 0.85rem; color: #64748B; margin-top: 0.2rem; }
    .badge-pill {
        display: inline-block; padding: 0.22rem 0.7rem;
        border-radius: 999px; font-size: 0.78rem; font-weight: 700;
        margin-right: 0.45rem; margin-top: 0.45rem;
    }
    .badge-green { background-color: #DCFCE7; color: #15803D; }
    .badge-gray { background-color: #F1F5F9; color: #475569; }
    .badge-blue { background-color: #DBEAFE; color: #1D4ED8; }
    .badge-amber { background-color: #FEF3C7; color: #92400E; }
    .sim-hint { font-size: 0.78rem; color: #475569; font-weight: 700; margin-top: 0.55rem; }
</style>
""".replace("__ACCENT__", ACCENT).replace("__GRADIENT__", GRADIENT)
st.markdown(_CSS, unsafe_allow_html=True)

# ── Header Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <div class="header-title">🍴 Task 2: Restaurant Recommendation System</div>
    <div class="header-sub">Content-based recommendation engine powered by TF-IDF tokenization and Cosine Similarity scoring.</div>
    <span class="header-chip">📝 TF-IDF Vectorization</span>
    <span class="header-chip">📐 Cosine Similarity</span>
    <span class="header-chip">⚙️ Hybrid Filtering</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Configuration ─────────────────────────────────────────────────────
st.sidebar.markdown("""
<div class="side-brand">
    <div class="side-brand-title">🍴 Recommender</div>
    <div class="side-brand-sub">Cognify · Task 2</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("⚙️ Search Controls")

with st.sidebar.expander("⚙️ Advanced Data Settings", expanded=False):
    data_path = st.text_input("Dataset Path", value=DEFAULT_DATA_PATH)

if not os.path.exists(data_path):
    st.error(f"❌ Dataset file not found at: `{data_path}`")
    st.stop()

@st.cache_resource(show_spinner="Indexing restaurant profiles ...")
def get_recommender(path):
    return build_recommender(path)

df_clean, vectorizer, matrix = get_recommender(data_path)

all_cities = sorted(df_clean["City"].dropna().unique().tolist())
all_cuisines = sorted({c.strip() for row in df_clean["Cuisines"].dropna() for c in row.split(",")})

st.sidebar.subheader("🎯 User Preferences")

cuisine_dropdown = st.sidebar.selectbox("Preferred Cuisine", options=[""] + all_cuisines, index=0)
cuisine_text = st.sidebar.text_input("...or Type Cuisine Keyword", value="", placeholder="e.g. Italian, Japanese, Cafe")
cuisine = cuisine_text.strip() if cuisine_text.strip() else cuisine_dropdown

price_range = st.sidebar.select_slider(
    "Budget Tier",
    options=[1, 2, 3, 4],
    value=2,
    format_func=lambda x: {1: "1 - Budget", 2: "2 - Moderate", 3: "3 - Upscale", 4: "4 - Fine Dining"}[x]
)

city_filter = st.sidebar.selectbox("Filter City", options=["All Cities"] + all_cities, index=0)
city = "" if city_filter == "All Cities" else city_filter

c_col1, c_col2 = st.sidebar.columns(2)
with c_col1:
    delivery_opt = st.radio("Online Delivery", options=["Any", "Yes", "No"])
with c_col2:
    booking_opt = st.radio("Table Booking", options=["Any", "Yes", "No"])

online_delivery = None if delivery_opt == "Any" else (delivery_opt == "Yes")
table_booking = None if booking_opt == "Any" else (booking_opt == "Yes")

min_rating = st.sidebar.slider("Minimum Rating", min_value=0.0, max_value=5.0, value=3.5, step=0.1)
top_n = st.sidebar.slider("Recommendations Count", min_value=5, max_value=25, value=10)

search_clicked = st.sidebar.button("🔍 Search Recommendations", type="primary", use_container_width=True)

# ── KPI Metric Cards ──────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.markdown(
    f'<div class="kpi-card"><div class="kpi-label">Indexed Restaurants</div>'
    f'<div class="kpi-value">{len(df_clean):,}</div>'
    f'<div class="kpi-sub">rated & processed</div></div>',
    unsafe_allow_html=True)
k2.markdown(
    f'<div class="kpi-card"><div class="kpi-label">Available Cities</div>'
    f'<div class="kpi-value">{df_clean["City"].nunique():,}</div>'
    f'<div class="kpi-sub">unique locations</div></div>',
    unsafe_allow_html=True)
k3.markdown(
    f'<div class="kpi-card"><div class="kpi-label">Cuisine Categories</div>'
    f'<div class="kpi-value">{len(all_cuisines):,}</div>'
    f'<div class="kpi-sub">flavours to explore</div></div>',
    unsafe_allow_html=True)
k4.markdown(
    f'<div class="kpi-card"><div class="kpi-label">Avg Dataset Rating</div>'
    f'<div class="kpi-value">{df_clean["Aggregate rating"].mean():.2f} ⭐</div>'
    f'<div class="kpi-sub">across all listings</div></div>',
    unsafe_allow_html=True)

# ── Main Display Area ────────────────────────────────────────────────────────
if not search_clicked:
    st.info("👈 Select your culinary preferences in the sidebar and click **Search Recommendations**.")

    st.markdown('<div class="section-title">📊 Top Cuisines Overview</div>', unsafe_allow_html=True)
    cuisine_counts = (
        pd.Series([c.strip() for row in df_clean["Cuisines"].dropna() for c in row.split(",")])
        .value_counts().head(15).reset_index()
    )
    cuisine_counts.columns = ["Cuisine", "Count"]
    fig = px.bar(
        cuisine_counts, x="Count", y="Cuisine", orientation="h",
        color="Count", color_continuous_scale=["#D1FAE5", "#059669"],
        title="Top 15 Cuisines in Dataset"
    )
    fig.update_layout(height=420, coloraxis_showscale=False, yaxis=dict(autorange="reversed"), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        city_counts = df_clean["City"].value_counts().head(10).reset_index()
        city_counts.columns = ["City", "Count"]
        fig_city = px.bar(
            city_counts, x="Count", y="City", orientation="h",
            color="Count", color_continuous_scale=["#FDE68A", "#059669"],
            title="Top 10 Cities by Restaurants"
        )
        fig_city.update_layout(height=340, coloraxis_showscale=False, yaxis=dict(autorange="reversed"), template="plotly_white")
        st.plotly_chart(fig_city, use_container_width=True)

    with c_right:
        rating_hist = px.histogram(
            df_clean, x="Aggregate rating", nbins=25, color_discrete_sequence=["#059669"],
            title="Rating Distribution of Indexed Restaurants"
        )
        rating_hist.update_layout(height=340, template="plotly_white")
        st.plotly_chart(rating_hist, use_container_width=True)

else:
    if not cuisine:
        st.warning("⚠️ Please select or type a cuisine preference in the sidebar.")
        st.stop()

    recs = recommend(
        df_clean, vectorizer, matrix,
        cuisine=cuisine, price_range=price_range, city=city,
        online_delivery=online_delivery, table_booking=table_booking,
        min_rating=min_rating, top_n=top_n
    )

    if recs.empty:
        st.warning(f"No restaurants matched your exact criteria for **'{cuisine}'**. Try relaxing the minimum rating or city filters.")
        st.stop()

    st.markdown(f'<div class="section-title">🏆 Top Recommendations for “{cuisine}”</div>', unsafe_allow_html=True)
    st.caption(f"Budget Tier: {price_range} | Min Rating: {min_rating} ⭐ | Location: {city if city else 'All Cities'}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Matches Found", len(recs))
    m2.metric("Avg Rating", f"{recs['Aggregate rating'].mean():.2f} ⭐")
    m3.metric("Top Match Rating", f"{recs['Aggregate rating'].max():.2f} ⭐")
    m4.metric("Avg Similarity Score", f"{recs['Similarity'].mean():.3f}")

    tab_cards, tab_charts = st.tabs(["🍴 Recommended Restaurants List", "📊 Similarity & Data Metrics"])

    price_labels = {1: "Budget", 2: "Moderate", 3: "Upscale", 4: "Fine Dining"}

    with tab_cards:
        for i, row in recs.iterrows():
            price_str = price_labels.get(row["Price range"], f"Tier {row['Price range']}")
            cuisine_badges = "".join(
                f'<span class="badge-pill badge-blue">🍽️ {c.strip()}</span>'
                for c in str(row["Cuisines"]).split(",")[:4]
            )
            del_badge = (
                '<span class="badge-pill badge-green">🛵 Online Delivery</span>'
                if row["Has Online delivery"] else
                '<span class="badge-pill badge-gray">🚫 No Delivery</span>'
            )
            book_badge = (
                '<span class="badge-pill badge-green">📅 Table Booking</span>'
                if row["Has Table booking"] else
                '<span class="badge-pill badge-gray">🚫 No Booking</span>'
            )
            sim_pct = float(row["Similarity"]) * 100
            st.markdown(
                f'<div class="rec-card">'
                f'<span class="rec-rank">{i + 1}</span>'
                f'<span class="rec-name">{row["Restaurant Name"]}</span>'
                f'<div class="rec-loc">📍 {row["City"]} &nbsp;·&nbsp; 💰 <b>{price_str}</b> &nbsp;·&nbsp; 🍜 Similarity <b>{row["Similarity"]:.3f}</b></div>'
                f'<div>{cuisine_badges}{del_badge}{book_badge}</div>'
                f'<div class="sim-hint">Match confidence</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.progress(min(int(sim_pct), 100), text=f"Rating {row['Aggregate rating']} ⭐ · {row['Votes']:,} votes")
            st.divider()

    with tab_charts:
        l_col, r_col = st.columns(2)
        with l_col:
            fig_r = px.bar(
                recs, x="Restaurant Name", y="Aggregate rating",
                color="Similarity", color_continuous_scale=["#A7F3D0", "#047857"],
                title="Restaurant Rating & Match Similarity"
            )
            fig_r.update_layout(height=380, xaxis_tickangle=-30, template="plotly_white")
            st.plotly_chart(fig_r, use_container_width=True)

        with r_col:
            sim_only = recs.sort_values("Similarity", ascending=True)
            fig_sim = px.bar(
                sim_only, x="Restaurant Name", y="Similarity", orientation="v",
                color="Similarity", color_continuous_scale=["#D1FAE5", "#047857"],
                title="Cosine Similarity Ranking"
            )
            fig_sim.update_layout(height=380, xaxis_tickangle=-30, coloraxis_showscale=False, template="plotly_white")
            st.plotly_chart(fig_sim, use_container_width=True)

        st.markdown("##### Detailed Recommendations Data Table")
        st.dataframe(recs, use_container_width=True)

st.divider()
st.caption("Cognify · Task 2 – Restaurant Recommendation System Project")