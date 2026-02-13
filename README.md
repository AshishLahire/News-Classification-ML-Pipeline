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
##Folder Structure

    📁 bbc-news-classification/
    │
    ├── 📁 src/
    │   ├── 📄 data_preprocessing.py      # Text cleaning, stopword removal, stemming
    │   ├── 📄 feature_engineering.py     # TF-IDF vectorization, train-test split
    │   ├── 📄 train.py                  # Logistic Regression model training
    │   ├── 📄 evaluate.py               # Model evaluation, metrics generation
    │   ├── 📄 main.py                  # Pipeline orchestrator
    │   └── 📄 download_20news.py       # Dataset download utility (optional)
    │
    ├── 📁 data/
    │   ├── 📄 bbc_news.csv            # Original dataset (downloaded)
    │   ├── 📄 processed_data.csv      # Cleaned and preprocessed data
    │   ├── 📄 X_train.csv            # Training texts (for reference)
    │   └── 📄 X_test.csv             # Test texts (for reference)
    │
    ├── 📁 models/
    │   ├── 📄 logistic_regression_model.joblib  # Trained classifier
    │   └── 📄 tfidf_vectorizer.joblib           # Fitted TF-IDF vectorizer
    │
    ├── 📁 results/
    │   ├── 📄 metrics.txt            # Accuracy, confusion matrix, classification report
    │   └── 📄 confusion_matrix.png   # Optional: visualization
    │
    ├── 📄 requirements.txt         # Project dependencies
    ├── 📄 README.md              # Project documentation
    ├── 📄 .gitignore            # Git ignore rules


## 🛠️ How to Run

```bash
pip install -r requirements.txt
python -m src.main
