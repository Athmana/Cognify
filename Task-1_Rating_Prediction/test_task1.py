import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from rating_prediction import run_pipeline, DEFAULT_DATA_PATH


def test_task1_pipeline():
    print("Running Task 1 test suite ...")
    assert os.path.exists(DEFAULT_DATA_PATH), f"Dataset path missing: {DEFAULT_DATA_PATH}"
    
    results = run_pipeline(DEFAULT_DATA_PATH)
    
    # Verify results dict contents
    assert "metrics_df" in results, "Missing metrics_df in pipeline output"
    assert "importance_df" in results, "Missing importance_df in pipeline output"
    assert "trained" in results, "Missing trained models in pipeline output"
    
    metrics = results["metrics_df"]
    print("Metrics Table:")
    print(metrics)
    
    # Assert models exist and R2 is reasonably above baseline threshold
    assert "Random Forest" in metrics.index
    assert "Linear Regression" in metrics.index
    assert "Decision Tree" in metrics.index
    assert metrics.loc["Random Forest", "R²"] > 0.5, "Random Forest R² score unexpectedly low"
    
    print("\n[SUCCESS] Task 1 Pipeline Integration Test Passed!")


if __name__ == "__main__":
    test_task1_pipeline()
