"""
Cognify - Task 3: Cuisine Classification
=========================================
Multi-Label Cuisine Classification using OneVsRest with Logistic Regression and Random Forest.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, hamming_loss
)

# ──────────────────────────────────────────────
# 1. CONFIG & DATA LOADING
# ──────────────────────────────────────────────

# Shared dataset: <repo_root>/data/Dataset.csv
DEFAULT_DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "Dataset.csv")
)

TOP_N_CUISINES = 10

FEATURE_COLS = [
    "Country Code",
    "City",
    "Average Cost for two",
    "Currency",
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now",
    "Price range",
    "Aggregate rating",
    "Votes",
]


def load_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load Zomato dataset CSV with multiple fallback paths."""
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
# 2. PREPROCESSING & MULTI-LABEL BINARIZATION
# ──────────────────────────────────────────────

def get_top_cuisines(df: pd.DataFrame, top_n: int = TOP_N_CUISINES) -> list[str]:
    """Extract top N most frequent individual cuisines."""
    counts = Counter(
        c.strip()
        for row in df["Cuisines"].dropna()
        for c in row.split(",")
    )
    return [c for c, _ in counts.most_common(top_n)]


def preprocess(df: pd.DataFrame, top_cuisines: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, MultiLabelBinarizer, StandardScaler, dict]:
    """
    Clean features, apply StandardScaler, and binarise multi-label target.
    Returns (X_scaled_df, Y_df, mlb, scaler, encoders).
    """
    df = df.copy()
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")

    # Binary flags
    for col in ["Has Table booking", "Has Online delivery", "Is delivering now"]:
        df[col] = (df[col].astype(str).str.strip().str.lower() == "yes").astype(int)

    # Label encode string columns
    encoders = {}
    for col in ["City", "Currency"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Filter target labels to top_cuisines
    def filter_labels(raw: str) -> list[str]:
        labels = [c.strip() for c in str(raw).split(",")]
        return [l for l in labels if l in top_cuisines]

    df["label_list"] = df["Cuisines"].apply(filter_labels)
    df = df[df["label_list"].map(len) > 0].reset_index(drop=True)

    # MultiLabelBinarizer transform
    mlb = MultiLabelBinarizer(classes=top_cuisines)
    Y_arr = mlb.fit_transform(df["label_list"])
    Y = pd.DataFrame(Y_arr, columns=mlb.classes_)

    X_raw = df[FEATURE_COLS].copy()

    # Scale feature matrix to prevent LogisticRegression convergence warnings
    scaler = StandardScaler()
    X_scaled_arr = scaler.fit_transform(X_raw)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=FEATURE_COLS)

    return X_scaled, Y, mlb, scaler, encoders


# ──────────────────────────────────────────────
# 3. MODELS & TRAINING
# ──────────────────────────────────────────────

def build_models() -> dict:
    """Return OneVsRest multi-label classifiers."""
    return {
        "Logistic Regression": OneVsRestClassifier(
            LogisticRegression(max_iter=2000, random_state=42, C=1.0, solver="lbfgs")
        ),
        "Random Forest": OneVsRestClassifier(
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        ),
    }


def train_all(models: dict, X_train, Y_train) -> dict:
    """Fit all classifiers."""
    trained = {}
    for name, model in models.items():
        model.fit(X_train, Y_train)
        trained[name] = model
    return trained


# ──────────────────────────────────────────────
# 4. EVALUATION
# ──────────────────────────────────────────────

def evaluate_overall(trained: dict, X_test, Y_test) -> pd.DataFrame:
    """Compute overall multi-label evaluation metrics."""
    rows = []
    for name, model in trained.items():
        Y_pred = model.predict(X_test)
        rows.append({
            "Model":             name,
            "Subset Accuracy":   round(float(accuracy_score(Y_test, Y_pred)), 4),
            "Hamming Loss":      round(float(hamming_loss(Y_test, Y_pred)), 4),
            "Precision (micro)": round(float(precision_score(Y_test, Y_pred, average="micro", zero_division=0)), 4),
            "Recall (micro)":    round(float(recall_score(Y_test, Y_pred, average="micro", zero_division=0)), 4),
            "F1 (micro)":        round(float(f1_score(Y_test, Y_pred, average="micro", zero_division=0)), 4),
            "F1 (macro)":        round(float(f1_score(Y_test, Y_pred, average="macro", zero_division=0)), 4),
        })
    return pd.DataFrame(rows).set_index("Model")


def evaluate_per_label(trained: dict, X_test, Y_test, label_names: list[str]) -> dict[str, pd.DataFrame]:
    """Compute per-cuisine precision, recall, F1-score, and support."""
    result = {}
    for name, model in trained.items():
        Y_pred = model.predict(X_test)
        report = classification_report(
            Y_test, Y_pred,
            target_names=label_names,
            output_dict=True,
            zero_division=0
        )
        rows = []
        for cuisine in label_names:
            r = report.get(cuisine, {})
            rows.append({
                "Cuisine":   cuisine,
                "Precision": round(float(r.get("precision", 0)), 4),
                "Recall":    round(float(r.get("recall", 0)), 4),
                "F1":        round(float(r.get("f1-score", 0)), 4),
                "Support":   int(r.get("support", 0)),
            })
        result[name] = pd.DataFrame(rows).set_index("Cuisine")
    return result


# ──────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ──────────────────────────────────────────────

def feature_importance_rf(model: OneVsRestClassifier, feature_names: list[str]) -> pd.DataFrame:
    """Average feature importances across all binary Random Forest estimators."""
    importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    total = importances.sum()
    importances = importances / total if total > 0 else importances
    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": np.round(importances, 6)
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# 6. PIPELINE RUNNER
# ──────────────────────────────────────────────

def run_pipeline(data_path: str = DEFAULT_DATA_PATH) -> dict:
    """Execute complete Task 3 pipeline."""
    df_raw = load_data(data_path)
    top_cuisines = get_top_cuisines(df_raw, TOP_N_CUISINES)
    X, Y, mlb, scaler, encoders = preprocess(df_raw, top_cuisines)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    models = build_models()
    trained = train_all(models, X_train, Y_train)

    overall_df = evaluate_overall(trained, X_test, Y_test)
    per_label_dfs = evaluate_per_label(trained, X_test, Y_test, top_cuisines)
    rf_importance = feature_importance_rf(trained["Random Forest"], FEATURE_COLS)

    return {
        "overall_df":    overall_df,
        "per_label_dfs": per_label_dfs,
        "rf_importance": rf_importance,
        "trained":       trained,
        "X_train":       X_train,
        "X_test":        X_test,
        "Y_train":       Y_train,
        "Y_test":        Y_test,
        "top_cuisines":  top_cuisines,
        "feature_names": FEATURE_COLS,
        "mlb":           mlb,
        "scaler":        scaler,
        "encoders":      encoders,
    }


if __name__ == "__main__":
    print("Running Task 3 Cuisine Classification Pipeline ...")
    results = run_pipeline()
    print("\n-- Overall Multi-Label Performance --")
    print(results["overall_df"].to_string())
    print("\n-- Per-Cuisine Metrics (Random Forest) --")
    print(results["per_label_dfs"]["Random Forest"].to_string())
    print("\nDone.")
