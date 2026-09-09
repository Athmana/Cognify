import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
import sys

sys.path.insert(0, os.path.dirname(__file__))
from restaurant_recommendation import build_recommender, recommend, DEFAULT_DATA_PATH


def test_task2_pipeline():
    print("Running Task 2 test suite ...")
    assert os.path.exists(DEFAULT_DATA_PATH), f"Dataset file missing: {DEFAULT_DATA_PATH}"

    df_clean, vectorizer, matrix = build_recommender(DEFAULT_DATA_PATH)
    assert len(df_clean) > 0, "Cleaned dataset should not be empty"
    assert matrix.shape[0] == len(df_clean), "TF-IDF matrix row count mismatch"

    # Test recommendation query
    recs = recommend(
        df_clean, vectorizer, matrix,
        cuisine="Italian", price_range=2, min_rating=3.5, top_n=5
    )

    print("\nRecommendations Output:")
    print(recs[["Restaurant Name", "City", "Cuisines", "Aggregate rating", "Similarity"]])

    assert not recs.empty, "Recommendations returned empty result for valid query"
    assert "Similarity" in recs.columns, "Missing Similarity column in output dataframe"
    assert recs["Similarity"].iloc[0] > 0, "Top similarity score should be positive"

    print("\n[SUCCESS] Task 2 Pipeline Integration Test Passed!")


if __name__ == "__main__":
    test_task2_pipeline()
