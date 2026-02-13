from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineer
from data_preprocessing import DataPreprocessor

class ModelTrainer:
    def __init__(self):
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs',
            C=1.0,
            class_weight='balanced',
            n_jobs=-1
        )
        self.X_train_tfidf = None
        self.X_test_tfidf = None
        self.y_train = None
        self.y_test = None
        
    def train(self, X_train_tfidf, y_train):
        """Train the logistic regression model"""
        print("Training Logistic Regression model...")
        self.X_train_tfidf = X_train_tfidf
        self.y_train = y_train
        self.model.fit(X_train_tfidf, y_train)
        print("Model training completed")
        return self.model
    
    def evaluate_training(self, X_test_tfidf, y_test):
        """Evaluate model performance"""
        self.X_test_tfidf = X_test_tfidf
        self.y_test = y_test
        
        y_train_pred = self.model.predict(self.X_train_tfidf)
        y_test_pred = self.model.predict(self.X_test_tfidf)
        
        train_acc = accuracy_score(self.y_train, y_train_pred)
        test_acc = accuracy_score(self.y_test, y_test_pred)
        
        print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Calculate per-class accuracy
        print("\nPer-class Test Accuracy:")
        labels = np.unique(self.y_test)
        for label in labels:
            mask = self.y_test == label
            if np.sum(mask) > 0:
                class_acc = accuracy_score(self.y_test[mask], y_test_pred[mask])
                print(f"  {label}: {class_acc:.4f} ({class_acc*100:.2f}%)")
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'classification_report': classification_report(self.y_test, y_test_pred),
            'confusion_matrix': confusion_matrix(self.y_test, y_test_pred)
        }
    
    def save_model(self, path='models/logistic_regression_model.joblib'):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
        
    def load_model(self, path='models/logistic_regression_model.joblib'):
        """Load a trained model"""
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
        return self.model

def main():
    """Train the model"""
    print("="*60)
    print("BBC NEWS CLASSIFICATION - MODEL TRAINING")
    print("="*60)
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    df = preprocessor.preprocess_dataframe(df)
    
    # Create features
    engineer = FeatureEngineer()
    X_train_tfidf, X_test_tfidf, y_train, y_test = engineer.create_features(df)
    engineer.save_vectorizer()
    
    # Train model
    trainer = ModelTrainer()
    model = trainer.train(X_train_tfidf, y_train)
    evaluation = trainer.evaluate_training(X_test_tfidf, y_test)
    trainer.save_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Test Accuracy: {evaluation['test_accuracy']:.4f} ({evaluation['test_accuracy']*100:.2f}%)")
    print("\nClassification Report:")
    print(evaluation['classification_report'])

if __name__ == "__main__":
    main()