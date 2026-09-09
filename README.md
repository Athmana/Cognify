# Cognify Machine Learning Projects

This repository contains **3 distinct, self-contained Machine Learning projects** developed for restaurant dataset analysis:

```
cognify/
├── Task-1_Rating_Prediction/         # Project 1: Restaurant Rating Prediction (Regression)
├── Task-2_Restaurant_Recommendation/  # Project 2: Restaurant Recommendation System (TF-IDF Content Filtering)
└── Task-3_Cuisine_Classification/     # Project 3: Multi-Label Cuisine Classification (OneVsRest Classification)
```

---

## 📁 Projects Breakdown

### 1. 🍽️ [Task 1: Restaurant Rating Prediction](file:///d:/projects/cognify/Task-1_Rating_Prediction/README.md)
Predicts the **Aggregate Rating** of a restaurant using Linear Regression, Decision Trees, and Random Forests. Includes an interactive Streamlit web dashboard and single-restaurant rating predictor.

- **Directory**: `Task-1_Rating_Prediction/`
- **CLI Execution**: `python Task-1_Rating_Prediction/rating_prediction.py`
- **Web UI Execution**: `streamlit run Task-1_Rating_Prediction/app.py`
- **Tests Execution**: `python Task-1_Rating_Prediction/test_task1.py`

---

### 2. 🍴 [Task 2: Restaurant Recommendation System](file:///d:/projects/cognify/Task-2_Restaurant_Recommendation/README.md)
Recommends top matching restaurants based on user culinary preferences, budget tier, city, delivery, and table booking options using **TF-IDF vectorization** and **Cosine Similarity**.

- **Directory**: `Task-2_Restaurant_Recommendation/`
- **CLI Execution**: `python Task-2_Restaurant_Recommendation/restaurant_recommendation.py`
- **Web UI Execution**: `streamlit run Task-2_Restaurant_Recommendation/app.py`
- **Tests Execution**: `python Task-2_Restaurant_Recommendation/test_task2.py`

---

### 3. 🍜 [Task 3: Cuisine Classification](file:///d:/projects/cognify/Task-3_Cuisine_Classification/README.md)
Classifies restaurants into top cuisine categories using **OneVsRest Multi-Label Classifiers** (Logistic Regression & Random Forest) with `StandardScaler` feature normalization.

- **Directory**: `Task-3_Cuisine_Classification/`
- **CLI Execution**: `python Task-3_Cuisine_Classification/cuisine_classification.py`
- **Web UI Execution**: `streamlit run Task-3_Cuisine_Classification/app.py`
- **Tests Execution**: `python Task-3_Cuisine_Classification/test_task3.py`

---

## 💻 Installation & Requirements

Ensure Python 3.10+ is installed along with the core dependencies:
```bash
pip install -r Task-1_Rating_Prediction/requirements.txt
```
*(Dependencies across all 3 projects: `pandas`, `numpy`, `scikit-learn`, `streamlit`, `plotly`)*
