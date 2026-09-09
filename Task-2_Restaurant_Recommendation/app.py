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

# ── Custom Styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #0F172A 0%, #059669 100%);
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
    .rec-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-green { background-color: #DCFCE7; color: #15803D; }
    .badge-gray { background-color: #F1F5F9; color: #475569; }
    .badge-blue { background-color: #DBEAFE; color: #1D4ED8; }
</style>
""", unsafe_allow_html=True)

# ── Header Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <div class="header-title">🍴 Task 2: Restaurant Recommendation System</div>
    <div class="header-sub">Content-based recommendation engine powered by TF-IDF tokenization and Cosine Similarity scoring.</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Configuration ─────────────────────────────────────────────────────
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

# ── Main Display Area ────────────────────────────────────────────────────────
if not search_clicked:
    st.info("👈 Select your culinary preferences in the sidebar and click **Search Recommendations**.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Indexed Restaurants", f"{len(df_clean):,}")
    col2.metric("Available Cities", df_clean["City"].nunique())
    col3.metric("Cuisine Categories", len(all_cuisines))
    col4.metric("Avg Dataset Rating", f"{df_clean['Aggregate rating'].mean():.2f} ⭐")

    st.markdown("### 📊 Top Cuisines Overview")
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
    fig.update_layout(height=400, coloraxis_showscale=False, yaxis=dict(autorange="reversed"), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

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

    st.subheader(f"🏆 Top Recommendations for '{cuisine}'")
    st.caption(f"Budget Tier: {price_range} | Min Rating: {min_rating} ⭐ | Location: {city if city else 'All Cities'}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Matches Found", len(recs))
    m2.metric("Avg Rating", f"{recs['Aggregate rating'].mean():.2f} ⭐")
    m3.metric("Top Match Rating", f"{recs['Aggregate rating'].max():.2f} ⭐")
    m4.metric("Avg Similarity Score", f"{recs['Similarity'].mean():.3f}")

    tab_cards, tab_charts = st.tabs(["🍴 Recommended Restaurants List", "📊 Similarity & Data Metrics"])

    with tab_cards:
        for i, row in recs.iterrows():
            with st.container():
                c_main, c_metrics = st.columns([3, 1])
                with c_main:
                    st.markdown(f"#### {i+1}. {row['Restaurant Name']}")
                    st.write(f"📍 **Location**: {row['City']}  |  🍽️ **Cuisines**: {row['Cuisines']}")
                    
                    price_str = {1: "Budget", 2: "Moderate", 3: "Upscale", 4: "Fine Dining"}.get(row["Price range"], f"Tier {row['Price range']}")
                    del_badge = "✅ Delivery" if row["Has Online delivery"] else "❌ No Delivery"
                    book_badge = "✅ Table Booking" if row["Has Table booking"] else "❌ No Booking"
                    st.markdown(
                        f"`Price: {price_str}` &nbsp;&nbsp;|&nbsp;&nbsp; `{del_badge}` &nbsp;&nbsp;|&nbsp;&nbsp; `{book_badge}`"
                    )

                with c_metrics:
                    st.metric("Rating", f"{row['Aggregate rating']} ⭐")
                    st.caption(f"Similarity: **{row['Similarity']:.3f}**")
            st.divider()

    with tab_charts:
        fig_r = px.bar(
            recs, x="Restaurant Name", y="Aggregate rating",
            color="Similarity", color_continuous_scale=["#A7F3D0", "#047857"],
            title="Restaurant Rating & Match Similarity"
        )
        fig_r.update_layout(height=380, xaxis_tickangle=-30, template="plotly_white")
        st.plotly_chart(fig_r, use_container_width=True)

        st.markdown("##### Detailed Recommendations Data Table")
        st.dataframe(recs, use_container_width=True)

st.divider()
st.caption("Cognify · Task 2 – Restaurant Recommendation System Project")
