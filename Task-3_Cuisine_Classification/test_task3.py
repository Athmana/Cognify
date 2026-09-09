import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
import sys

sys.path.insert(0, os.path.dirname(__file__))
from cuisine_classification import run_pipeline, DEFAULT_DATA_PATH


def test_task3_pipeline():
    print("Running Task 3 test suite ...")
    assert os.path.exists(DEFAULT_DATA_PATH), f"Dataset file missing: {DEFAULT_DATA_PATH}"

    results = run_pipeline(DEFAULT_DATA_PATH)

    assert "overall_df" in results, "Missing overall_df in pipeline output"
    assert "per_label_dfs" in results, "Missing per_label_dfs in pipeline output"
    assert "trained" in results, "Missing trained models dictionary"

    overall = results["overall_df"]
    print("\nOverall Performance Summary:")
    print(overall)

    assert "Random Forest" in overall.index
    assert "Logistic Regression" in overall.index
    assert overall.loc["Random Forest", "F1 (micro)"] > 0.25, "Random Forest micro F1 unexpectedly low"

    print("\n[SUCCESS] Task 3 Pipeline Integration Test Passed!")


if __name__ == "__main__":
    test_task3_pipeline()
