# 📰 News Classification Project

A machine learning pipeline for multi-class news classification using TF-IDF and advanced linear models.

---

## 🚀 Features

- Data preprocessing pipeline
- Advanced TF-IDF feature engineering
- Hyperparameter tuning (GridSearchCV)
- Model evaluation with classification report
- Model & vectorizer artifact saving

---

## 🧠 Model Used

- SGDClassifier / LinearSVC
- TF-IDF with unigrams, bigrams, trigrams
- 20-class text classification

---

## 📊 Performance

- Accuracy: ~0.85 (depends on tuning)
- Macro F1 Score: ~0.85

---

## 🛠️ How to Run

```bash
pip install -r requirements.txt
python -m src.main
