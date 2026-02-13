from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import numpy as np
import os
import pandas as pd
from feature_engineering import FeatureEngineer
from data_preprocessing import DataPreprocessor

class ModelEvaluator:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        
    def load_artifacts(self, 
                      model_path='models/logistic_regression_model.joblib',
                      vectorizer_path='models/tfidf_vectorizer.joblib'):
        """Load trained model and vectorizer"""
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            print("Model and vectorizer loaded successfully")
            return True
        else:
            print("Model or vectorizer not found. Please run train.py first.")
            return False
        
    def evaluate(self, X_test_tfidf, y_test):
        """Evaluate model and return metrics"""
        # Make predictions
        y_pred = self.model.predict(X_test_tfidf)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        # Get unique labels
        labels = np.unique(y_test)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'labels': labels,
            'y_pred': y_pred,
            'y_true': y_test
        }
    
    def save_metrics(self, metrics, path='results/metrics.txt'):
        """Save evaluation metrics to file"""
        os.makedirs('results', exist_ok=True)
        
        with open(path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("BBC NEWS CLASSIFICATION - MODEL EVALUATION METRICS\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Test Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Accuracy Percentage: {metrics['accuracy']*100:.2f}%\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write("-"*40 + "\n")
            f.write("Labels: " + ", ".join(metrics['labels']) + "\n")
            f.write("-"*40 + "\n")
            
            # Write confusion matrix in a readable format
            cm = metrics['confusion_matrix']
            f.write("Predicted ->\n")
            f.write("Actual\n")
            for i, label in enumerate(metrics['labels']):
                f.write(f"{label[:12]:<12} ")
                for j in range(len(metrics['labels'])):
                    f.write(f"{cm[i, j]:<6}")
                f.write("\n")
            
            f.write("\n" + "-"*40 + "\n\n")
            f.write("Classification Report:\n")
            f.write("-"*40 + "\n")
            f.write(metrics['classification_report'])
            
        print(f"Evaluation metrics saved to {path}")
        
    def print_metrics(self, metrics):
        """Print metrics to console"""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print("\nConfusion Matrix:")
        print("-"*50)
        
        # Print confusion matrix with labels
        cm = metrics['confusion_matrix']
        labels = metrics['labels']
        
        print(" " * 12, end="")
        for label in labels:
            print(f"{label[:8]:>8}", end="")
        print("\n" + "-"*60)
        
        for i, label in enumerate(labels):
            print(f"{label[:12]:<12}", end="")
            for j in range(len(labels)):
                print(f"{cm[i, j]:>8}", end="")
            print()
        
        print("\nClassification Report:")
        print("-"*50)
        print(metrics['classification_report'])

def main():
    """Run evaluation"""
    print("="*60)
    print("BBC NEWS CLASSIFICATION - MODEL EVALUATION")
    print("="*60)
    
    # Load model and vectorizer
    evaluator = ModelEvaluator()
    if not evaluator.load_artifacts():
        print("Please train the model first by running: python src/train.py")
        return
    
    # Load preprocessed data
    if not os.path.exists('data/processed_data.csv'):
        print("Processed data not found. Preprocessing data...")
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data()
        df = preprocessor.preprocess_dataframe(df)
        preprocessor.save_processed_data(df)
    else:
        df = pd.read_csv('data/processed_data.csv')
    
    # Create features to get the test split
    engineer = FeatureEngineer()
    # We only need the test data, but we need to recreate the same split
    X = df['cleaned_text'].fillna('').values
    y = df['category'].values
    
    from sklearn.model_selection import train_test_split
    _, X_test_texts, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Transform test data using loaded vectorizer
    print("Transforming test data...")
    X_test_tfidf = evaluator.vectorizer.transform(X_test_texts)
    print(f"Test data shape: {X_test_tfidf.shape}")
    
    # Evaluate
    metrics = evaluator.evaluate(X_test_tfidf, y_test)
    
    # Save and print results
    evaluator.save_metrics(metrics)
    evaluator.print_metrics(metrics)

if __name__ == "__main__":
    main()