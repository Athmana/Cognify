# Task 1: Restaurant Rating Prediction

Predicts the **Aggregate Rating** of restaurants using Machine Learning regression algorithms based on location, cost, delivery capabilities, price tier, and customer vote metrics.

---

## 📌 Project Overview
- **Objective**: Build and evaluate machine learning regression models to accurately predict restaurant ratings.
- **Models Evaluated**:
  - Linear Regression
  - Decision Tree Regressor (`max_depth=10`)
  - Random Forest Regressor (`n_estimators=100`)
- **Key Features**: Votes, Cuisines, Average Cost for Two, Country Code, City, Price Range, Online Delivery, Table Booking.

---

## 🛠️ Project Structure
```
Task-1_Rating_Prediction/
├── rating_prediction.py   # Machine learning pipeline (preprocessing, training, evaluation)
├── app.py                 # Interactive Streamlit dashboard & live rating predictor tool
├── test_task1.py          # Integration & pipeline verification tests
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## 🚀 Execution Instructions

### 1. Run Machine Learning Pipeline (CLI)
```bash
python Task-1_Rating_Prediction/rating_prediction.py
```

### 2. Run Interactive Web Portal (Streamlit)
```bash
streamlit run Task-1_Rating_Prediction/app.py
```

### 3. Run Automated Tests
```bash
python Task-1_Rating_Prediction/test_task1.py
```

---

## 📊 Benchmark Model Performance

| Model | MSE | RMSE | MAE | R² Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.1873 | 0.4328 | 0.3425 | 0.3943 |
| **Decision Tree** | 0.1441 | 0.3797 | 0.2826 | 0.5340 |
| **Random Forest** | **0.1203** | **0.3469** | **0.2584** | **0.6110** |

*Random Forest achieves the highest accuracy with R² = 0.6110.*
