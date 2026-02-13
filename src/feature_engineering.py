from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import os
import numpy as np

class FeatureEngineer:
    def __init__(self, max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            sublinear_tf=True
        )
        self.X_train_texts = None
        self.X_test_texts = None
        self.y_train = None
        self.y_test = None
        self.X_train_tfidf = None
        self.X_test_tfidf = None
        
    def create_features(self, df, text_column='cleaned_text', label_column='category'):
        """Convert text to TF-IDF features and split data"""
        print("Creating TF-IDF features...")
        
        # Extract features and labels
        X = df[text_column].fillna('').values
        y = df[label_column].values
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("\nClass distribution:")
        for cls, count in zip(unique, counts):
            print(f"  {cls}: {count} samples")
        
        # Filter out classes with less than 2 samples
        valid_classes = unique[counts >= 2]
        if len(valid_classes) < len(unique):
            print(f"\nRemoving classes with less than 2 samples: {set(unique) - set(valid_classes)}")
            mask = np.isin(y, valid_classes)
            X = X[mask]
            y = y[mask]
            print(f"Remaining samples: {len(X)}")
        
        # Check if we have enough samples
        if len(X) < 10:
            raise ValueError(f"Not enough samples after filtering: {len(X)}")
        
        # Check if stratification is possible
        unique, counts = np.unique(y, return_counts=True)
        min_samples = min(counts)
        
        if min_samples < 2:
            print("Warning: Some classes still have less than 2 samples. Using non-stratified split.")
            stratify_param = None
        else:
            stratify_param = y
        
        # Split data - use stratification only if possible
        try:
            self.X_train_texts, self.X_test_texts, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_param
            )
        except ValueError as e:
            print(f"Stratified split failed: {e}")
            print("Falling back to regular split...")
            self.X_train_texts, self.X_test_texts, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )
        
        print(f"\nTraining set size: {len(self.X_train_texts)}")
        print(f"Test set size: {len(self.X_test_texts)}")
        
        # Check final class distribution
        print("\nTraining set class distribution:")
        train_unique, train_counts = np.unique(self.y_train, return_counts=True)
        for cls, count in zip(train_unique, train_counts):
            print(f"  {cls}: {count} samples")
        
        print("\nTest set class distribution:")
        test_unique, test_counts = np.unique(self.y_test, return_counts=True)
        for cls, count in zip(test_unique, test_counts):
            print(f"  {cls}: {count} samples")
        
        # Fit and transform training data
        print("\nFitting TF-IDF vectorizer...")
        self.X_train_tfidf = self.vectorizer.fit_transform(self.X_train_texts)
        self.X_test_tfidf = self.vectorizer.transform(self.X_test_texts)
        
        print(f"TF-IDF features created. Shape: {self.X_train_tfidf.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        return self.X_train_tfidf, self.X_test_tfidf, self.y_train, self.y_test
    
    def get_raw_texts(self):
        """Return raw text splits"""
        return self.X_train_texts, self.X_test_texts, self.y_train, self.y_test
    
    def save_vectorizer(self, path='models/tfidf_vectorizer.joblib'):
        """Save the fitted vectorizer"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.vectorizer, path)
        print(f"Vectorizer saved to {path}")
        
    def save_split_data(self, path='data/'):
        """Save the split data for reference"""
        os.makedirs(path, exist_ok=True)
        if self.X_train_texts is not None:
            pd.DataFrame({'text': self.X_train_texts, 'label': self.y_train}).to_csv(
                f'{path}X_train.csv', index=False
            )
        if self.X_test_texts is not None:
            pd.DataFrame({'text': self.X_test_texts, 'label': self.y_test}).to_csv(
                f'{path}X_test.csv', index=False
            )
        print("Split data saved")

def main():
    """Test the feature engineer"""
    # Load preprocessed data
    if os.path.exists('data/processed_data.csv'):
        df = pd.read_csv('data/processed_data.csv')
        
        # Create features
        engineer = FeatureEngineer()
        X_train, X_test, y_train, y_test = engineer.create_features(df)
        engineer.save_vectorizer()
        engineer.save_split_data()
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
    else:
        print("Processed data not found. Please run data_preprocessing.py first.")

if __name__ == "__main__":
    main()