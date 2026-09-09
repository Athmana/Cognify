# Task 3: Multi-Label Cuisine Classification

Predicts the top cuisine categories served by restaurants using **OneVsRest Multi-Label Machine Learning Classifiers**.

---

## 📌 Project Overview
- **Objective**: Predict which cuisine tags (from the top 10 most frequent cuisines: *North Indian, Chinese, Fast Food, Mughlai, Italian, Bakery, Continental, Cafe, Desserts, South Indian*) apply to a restaurant.
- **Models Evaluated**:
  - OneVsRest Logistic Regression (with `StandardScaler` feature scaling and L-BFGS solver)
  - OneVsRest Random Forest Classifier (`n_estimators=100`)
- **Key Enhancements**:
  - Multi-label binarization with `MultiLabelBinarizer`.
  - Feature scaling with `StandardScaler` to resolve solver convergence limits.
  - Per-cuisine precision, recall, F1-score evaluation and class-imbalance bias analysis.

---

## 🛠️ Project Structure
```
Task-3_Cuisine_Classification/
├── cuisine_classification.py   # Multi-label ML pipeline (preprocessing, scaling, OneVsRest training)
├── app.py                      # Interactive Streamlit dashboard & live cuisine classifier tool
├── test_task3.py               # Integration test suite for multi-label pipeline
├── requirements.txt            # Dependency specifications
└── README.md                   # Project documentation
```

---

## 🚀 Execution Instructions

### 1. Run Machine Learning Pipeline (CLI)
```bash
python Task-3_Cuisine_Classification/cuisine_classification.py
```

### 2. Run Interactive Web Portal (Streamlit)
```bash
streamlit run Task-3_Cuisine_Classification/app.py
```

### 3. Run Automated Tests
```bash
python Task-3_Cuisine_Classification/test_task3.py
```

---

## 📊 Benchmark Model Performance

| Model | Subset Accuracy | Hamming Loss | Precision (micro) | Recall (micro) | F1 (micro) | F1 (macro) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.0832 | **0.1692** | **0.5419** | 0.1972 | 0.2892 | 0.1013 |
| **Random Forest** | **0.1327** | 0.1770 | 0.4894 | **0.3238** | **0.3897** | **0.2504** |

*Random Forest delivers stronger recall and F1 macro performance across imbalanced cuisine classes.*
