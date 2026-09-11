"""
Cognify - Task 2: Restaurant Recommendation System
===================================================
Content-Based Recommendation Engine using TF-IDF Vectorization and Cosine Similarity.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────

# Shared dataset: <repo_root>/data/Dataset.csv
DEFAULT_DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "Dataset.csv")
)


def load_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the Zomato restaurant dataset with multiple fallback paths."""
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        path,
        # Repo-root relative (works locally and on Streamlit Cloud)
        os.path.join(_script_dir, "..", "data", "Dataset.csv"),
        # Streamlit Cloud mount paths
        "/mount/src/cognify/data/Dataset.csv",
        "/mount/src/data/Dataset.csv",
        # Legacy QPredict paths (kept for backwards compatibility)
        "/mount/src/cognify/QPredict/data/uploads/Dataset.csv",
        "/mount/src/QPredict/data/uploads/Dataset.csv",
    ]
    for candidate in candidates:
        resolved = os.path.abspath(candidate)
        if os.path.exists(resolved):
            return pd.read_csv(resolved, encoding='utf-8')
    raise FileNotFoundError(
        f"Dataset CSV not found. Tried paths:\n" +
        "\n".join(f"  • {os.path.abspath(c)}" for c in candidates)
    )


# ──────────────────────────────────────────────
# 2. PREPROCESSING & FEATURE PROFILES
# ──────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset and build text profiles combining cuisines, price tier, city, and delivery features.
    """
    df = df.copy()

    # Filter to rated restaurants
    df = df[df["Aggregate rating"] > 0].reset_index(drop=True)
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")

    # Map binary Yes/No to booleans
    for col in ["Has Table booking", "Has Online delivery", "Is delivering now"]:
        df[col] = df[col].astype(str).str.strip().str.lower().map({"yes": True, "no": False}).fillna(False)

    # Build descriptive text tokens
    price_tag = df["Price range"].map({1: "cheap", 2: "moderate", 3: "expensive", 4: "luxury"}).fillna("moderate")
    delivery_tag = df["Has Online delivery"].map({True: "delivery", False: ""})
    booking_tag = df["Has Table booking"].map({True: "booking", False: ""})
    city_clean = df["City"].astype(str).str.lower().str.replace(" ", "_")
    cuisine_clean = df["Cuisines"].astype(str).str.lower().str.replace(",", " ").str.replace("  ", " ")

    # Combine text profile (price tag doubled for higher TF-IDF weight)
    df["profile"] = (
        cuisine_clean + " " +
        price_tag + " " + price_tag + " " +
        city_clean + " " +
        delivery_tag + " " +
        booking_tag
    ).str.strip()

    return df


# ──────────────────────────────────────────────
# 3. TF-IDF VECTORIZATION
# ──────────────────────────────────────────────

def build_tfidf_matrix(df: pd.DataFrame):
    """Fit TF-IDF Vectorizer on restaurant profile texts."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(df["profile"])
    return vectorizer, matrix


# ──────────────────────────────────────────────
# 4. QUERY BUILDER
# ──────────────────────────────────────────────

def build_query(
    cuisine: str,
    price_range: int,
    city: str = "",
    online_delivery: bool | None = None,
    table_booking: bool | None = None,
) -> str:
    """Transform user preferences into TF-IDF search query token string."""
    price_tag = {1: "cheap", 2: "moderate", 3: "expensive", 4: "luxury"}.get(price_range, "moderate")
    parts = [cuisine.lower().strip(), price_tag, price_tag]
    
    if city:
        parts.append(city.lower().strip().replace(" ", "_"))
    if online_delivery is True:
        parts.append("delivery")
    if table_booking is True:
        parts.append("booking")
        
    return " ".join(parts)


# ──────────────────────────────────────────────
# 5. RECOMMENDATION ENGINE
# ──────────────────────────────────────────────

def recommend(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    cuisine: str,
    price_range: int,
    city: str = "",
    online_delivery: bool | None = None,
    table_booking: bool | None = None,
    min_rating: float = 3.5,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Recommend top N restaurants matching specified user preferences.
    """
    working = df.copy()

    # Hard filtering criteria
    working = working[working["Aggregate rating"] >= min_rating]
    if city:
        working = working[working["City"].str.lower() == city.lower()]
    if online_delivery is not None:
        working = working[working["Has Online delivery"] == online_delivery]
    if table_booking is not None:
        working = working[working["Has Table booking"] == table_booking]

    if working.empty:
        return pd.DataFrame(columns=[
            "Restaurant Name", "City", "Cuisines", "Price range",
            "Aggregate rating", "Votes", "Has Online delivery",
            "Has Table booking", "Similarity"
        ])

    # Transform user query into vector space
    query = build_query(cuisine, price_range, city, online_delivery, table_booking)
    query_vec = vectorizer.transform([query])

    # Compute cosine similarity
    subset_idx = working.index.tolist()
    subset_matrix = tfidf_matrix[subset_idx]
    sims = cosine_similarity(query_vec, subset_matrix).flatten()

    working = working.copy()
    working["Similarity"] = np.round(sims, 4)

    # Sort results by Similarity, then Rating, then Votes
    result = (
        working.sort_values(["Similarity", "Aggregate rating", "Votes"], ascending=[False, False, False])
        .head(top_n)
        .reset_index(drop=True)
    )

    display_cols = [
        "Restaurant Name", "City", "Cuisines", "Price range",
        "Aggregate rating", "Votes", "Has Online delivery",
        "Has Table booking", "Rating text", "Similarity"
    ]
    return result[display_cols]


# ──────────────────────────────────────────────
# 6. PIPELINE RUNNER
# ──────────────────────────────────────────────

def build_recommender(data_path: str = DEFAULT_DATA_PATH):
    """Load data, preprocess, and construct TF-IDF index."""
    df_raw = load_data(data_path)
    df_clean = preprocess(df_raw)
    vectorizer, matrix = build_tfidf_matrix(df_clean)
    return df_clean, vectorizer, matrix


if __name__ == "__main__":
    print("Building Task 2 Restaurant Recommendation Index ...")
    df_clean, vectorizer, matrix = build_recommender()
    print(f"Index created for {len(df_clean):,} rated restaurants.")

    print("\n--- Test Recommendation ---")
    recs = recommend(df_clean, vectorizer, matrix, cuisine="Italian", price_range=2, min_rating=4.0, top_n=5)
    print(recs[["Restaurant Name", "City", "Cuisines", "Aggregate rating", "Similarity"]].to_string(index=False))
    print("\nDone.")
