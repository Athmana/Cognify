"""
Cognify - Task 1: Restaurant Rating Prediction
===============================================
Predicts the Aggregate Rating of a restaurant using Linear Regression, Decision Tree, and Random Forest.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_MAIN_FREE"] = "1"

from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────

DEFAULT_DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "QPredict", "data", "uploads", "Dataset.csv")
)


def load_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the Zomato restaurant dataset CSV."""
    if not os.path.exists(path):
        # Fallback to direct absolute path check
        alt_path = r"d:\projects\QPredict\data\uploads\Dataset.csv"
        if os.path.exists(alt_path):
            path = alt_path
        else:
            raise FileNotFoundError(f"Dataset CSV not found at '{path}'. Please verify the dataset location.")
    df = pd.read_csv(path)
    return df


# ──────────────────────────────────────────────
# 2. PREPROCESSING
# ──────────────────────────────────────────────

FEATURE_COLS = [
    "Country Code",
    "City",
    "Cuisines",
    "Average Cost for two",
    "Currency",
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now",
    "Price range",
    "Votes",
]
TARGET_COL = "Aggregate rating"


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Clean, encode, and prepare feature matrix X and target y.
    Returns (X, y, encoders) where encoders store fitted LabelEncoder objects.
    """
    df = df.copy()

    # Exclude unrated restaurants (Aggregate rating == 0.0)
    df = df[df[TARGET_COL] > 0].reset_index(drop=True)

    # Missing value handling
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")

    # Binary flag encoding
    for col in ["Has Table booking", "Has Online delivery", "Is delivering now"]:
        df[col] = (df[col].astype(str).str.strip().str.lower() == "yes").astype(int)

    # Label encoding for high-cardinality categorical columns
    encoders = {}
    for col in ["City", "Cuisines", "Currency"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y, encoders


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Split dataset into 80% train and 20% test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ──────────────────────────────────────────────
# 3. MODEL TRAINING
# ──────────────────────────────────────────────

def build_models() -> dict:
    """Return dictionary of ML regression models."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
    }


def train_all(models: dict, X_train, y_train) -> dict:
    """Fit all regression models on training data."""
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


# ──────────────────────────────────────────────
# 4. EVALUATION
# ──────────────────────────────────────────────

def evaluate(model, X_test, y_test) -> dict:
    """Compute MSE, RMSE, MAE, and R² for a fitted model."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "MSE":  round(float(mse), 4),
        "RMSE": round(float(np.sqrt(mse)), 4),
        "MAE":  round(float(mean_absolute_error(y_test, y_pred)), 4),
        "R²":   round(float(r2_score(y_test, y_pred)), 4),
    }


def evaluate_all(trained: dict, X_test, y_test) -> pd.DataFrame:
    """Return DataFrame summarizing performance across models."""
    rows = []
    for name, model in trained.items():
        metrics = evaluate(model, X_test, y_test)
        metrics["Model"] = name
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("Model")[["MSE", "RMSE", "MAE", "R²"]]


# ──────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ──────────────────────────────────────────────

def feature_importance(trained: dict, feature_names: list) -> pd.DataFrame:
    """Extract feature importances for tree models and normalized coefficients for linear regression."""
    rows = []
    for name, model in trained.items():
        if hasattr(model, "feature_importances_"):
            scores = model.feature_importances_
        elif hasattr(model, "coef_"):
            scores = np.abs(model.coef_)
            total = scores.sum()
            scores = scores / total if total > 0 else scores
        else:
            continue
        for feat, score in zip(feature_names, scores):
            rows.append({"Model": name, "Feature": feat, "Importance": round(float(score), 6)})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 6. PIPELINE RUNNER
# ──────────────────────────────────────────────

def run_pipeline(data_path: str = DEFAULT_DATA_PATH) -> dict:
    """Execute complete Task 1 machine learning pipeline."""
    df = load_data(data_path)
    X, y, encoders = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = build_models()
    trained = train_all(models, X_train, y_train)

    metrics_df = evaluate_all(trained, X_test, y_test)
    importance_df = feature_importance(trained, FEATURE_COLS)

    return {
        "metrics_df":    metrics_df,
        "importance_df": importance_df,
        "trained":       trained,
        "encoders":      encoders,
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "feature_names": FEATURE_COLS,
    }


if __name__ == "__main__":
    print("Executing Task 1: Restaurant Rating Prediction pipeline ...")
    results = run_pipeline()
    print("\n-- Model Performance --")
    print(results["metrics_df"].to_string())
    print("\n-- Feature Importances (Random Forest) --")
    rf_imp = results["importance_df"].query("Model == 'Random Forest'").sort_values("Importance", ascending=False)
    print(rf_imp.to_string(index=False))
    print("\nDone.")
