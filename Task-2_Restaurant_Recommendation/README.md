# Task 2: Restaurant Recommendation System

Content-Based Restaurant Recommendation Engine built using **TF-IDF Text Vectorization** and **Cosine Similarity** scoring.

---

## 📌 Project Overview
- **Objective**: Recommend top-matching restaurants based on user-defined criteria (cuisine preference, budget price level, city, online delivery, table booking, minimum aggregate rating).
- **Technique**:
  - Combined feature text profile creation (weighted by feature importance).
  - TF-IDF Vectorization with unigram and bigram tokenization (`ngram_range=(1, 2)`).
  - High-performance Cosine Similarity calculation for rank ordering.

---

## 🛠️ Project Structure
```
Task-2_Restaurant_Recommendation/
├── restaurant_recommendation.py   # Recommender engine & TF-IDF indexing pipeline
├── app.py                         # Streamlit interactive search portal & recommendation UI
├── test_task2.py                  # Integration test suite for recommendation query engine
├── requirements.txt               # Dependency specifications
└── README.md                      # Project documentation
```

---

## 🚀 Execution Instructions

### 1. Run Recommender Pipeline (CLI Test Cases)
```bash
python Task-2_Restaurant_Recommendation/restaurant_recommendation.py
```

### 2. Run Interactive Web Portal (Streamlit)
```bash
streamlit run Task-2_Restaurant_Recommendation/app.py
```

### 3. Run Automated Tests
```bash
python Task-2_Restaurant_Recommendation/test_task2.py
```
