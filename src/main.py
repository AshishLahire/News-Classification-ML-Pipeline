#!/usr/bin/env python3
"""
Main entry point for BBC News Classification Pipeline
"""

import sys
import os
import time
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from train import ModelTrainer
from evaluate import ModelEvaluator

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def check_dependencies():
    """Check if all required packages are installed"""
    try:
        import pandas
        import numpy
        import sklearn
        import nltk
        import joblib
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install required packages: pip install -r requirements.txt")
        return False

def cleanup_old_files():
    """Clean up old model and vectorizer files"""
    files_to_remove = [
        'models/logistic_regression_model.joblib',
        'models/tfidf_vectorizer.joblib',
        'data/processed_data.csv',
        'data/X_train.csv',
        'data/X_test.csv',
        'results/metrics.txt'
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✓ Removed old file: {file_path}")

def verify_dataset():
    """Verify that the dataset exists and has proper format"""
    if not os.path.exists('data/bbc_news.csv'):
        print("✗ Dataset not found at data/bbc_news.csv")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/sahilkirpekar/bbcnews-dataset")
        print("and place it in the data/ folder as 'bbc_news.csv'")
        return False
    return True

def main():
    """Execute complete ML pipeline"""
    start_time = time.time()
    
    print_header("BBC NEWS CLASSIFICATION PIPELINE")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Verify dataset
    if not verify_dataset():
        sys.exit(1)
    
    # Clean up old files
    print_header("CLEANING OLD FILES")
    cleanup_old_files()
    
    # Step 1: Data Preprocessing
    print_header("STEP 1: DATA PREPROCESSING")
    try:
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data()
        df = preprocessor.preprocess_dataframe(df)
        
        # Check if we have enough data after preprocessing
        if len(df) < 10:
            raise ValueError(f"Not enough data after preprocessing: {len(df)} samples")
            
        preprocessor.save_processed_data(df)
        print("✓ Data preprocessing completed")
    except Exception as e:
        print(f"✗ Data preprocessing failed: {e}")
        raise
    
    # Step 2: Feature Engineering
    print_header("STEP 2: FEATURE ENGINEERING")
    try:
        engineer = FeatureEngineer()
        X_train_tfidf, X_test_tfidf, y_train, y_test = engineer.create_features(df)
        engineer.save_vectorizer()
        engineer.save_split_data()
        print(f"✓ Feature engineering completed")
        print(f"  - Training features shape: {X_train_tfidf.shape}")
        print(f"  - Test features shape: {X_test_tfidf.shape}")
        print(f"  - Vocabulary size: {len(engineer.vectorizer.vocabulary_)}")
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        raise
    
    # Step 3: Model Training
    print_header("STEP 3: MODEL TRAINING")
    try:
        trainer = ModelTrainer()
        model = trainer.train(X_train_tfidf, y_train)
        trainer.save_model()
        training_metrics = trainer.evaluate_training(X_test_tfidf, y_test)
        print(f"✓ Model training completed")
        print(f"  - Training accuracy: {training_metrics['train_accuracy']:.4f} ({training_metrics['train_accuracy']*100:.2f}%)")
        print(f"  - Test accuracy: {training_metrics['test_accuracy']:.4f} ({training_metrics['test_accuracy']*100:.2f}%)")
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        raise
    
    # Step 4: Model Evaluation
    print_header("STEP 4: MODEL EVALUATION")
    try:
        evaluator = ModelEvaluator()
        evaluator.load_artifacts()
        
        # Use the same test data from feature engineering
        metrics = evaluator.evaluate(X_test_tfidf, y_test)
        evaluator.save_metrics(metrics)
        evaluator.print_metrics(metrics)
        print(f"✓ Model evaluation completed")
    except Exception as e:
        print(f"✗ Model evaluation failed: {e}")
        raise
    
    # Final Summary
    print_header("PIPELINE EXECUTION SUMMARY")
    execution_time = time.time() - start_time
    print(f"✅ Pipeline executed successfully!")
    print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
    print(f"📊 Final Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"📈 Training Accuracy: {training_metrics['train_accuracy']:.4f} ({training_metrics['train_accuracy']*100:.2f}%)")
    print(f"💾 Model saved: models/logistic_regression_model.joblib")
    print(f"📝 Vectorizer saved: models/tfidf_vectorizer.joblib")
    print(f"📊 Metrics saved: results/metrics.txt")
    print(f"📁 Processed data: data/processed_data.csv")
    print("\n🎉 News classification pipeline completed successfully!")
    
    return metrics['accuracy']

if __name__ == "__main__":
    try:
        accuracy = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)