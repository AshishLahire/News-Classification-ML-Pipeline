import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class DataPreprocessor:
    def __init__(self, data_path='data/bbc_news.csv'):
        self.data_path = data_path
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
    def load_data(self):
        """Load the dataset from CSV file"""
        try:
            df = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Display first few rows
            print("\nFirst few rows:")
            print(df.head())
            
            return df
        except FileNotFoundError:
            print(f"Error: Dataset not found at {self.data_path}")
            print("Please ensure the dataset is downloaded and placed in the data folder.")
            raise
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        words = text.split()
        
        # Remove stopwords and apply stemming
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def preprocess_dataframe(self, df):
        """Apply preprocessing to entire dataframe"""
        print("\nStarting text preprocessing...")
        
        # Check column names and map them correctly
        # Based on the error, the dataset has 'description' and 'labels' columns
        text_column = 'description' if 'description' in df.columns else 'Text' if 'Text' in df.columns else df.columns[0]
        label_column = 'labels' if 'labels' in df.columns else 'Category' if 'Category' in df.columns else df.columns[1]
        
        print(f"Using text column: '{text_column}'")
        print(f"Using label column: '{label_column}'")
        
        # Handle missing values
        initial_shape = df.shape
        df = df.dropna(subset=[text_column, label_column])
        print(f"Dropped {initial_shape[0] - df.shape[0]} rows with missing values")
        
        # Clean text
        print("Cleaning text data...")
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        print(f"Removed {df[df['cleaned_text'].str.len() == 0].shape[0]} rows with empty text after cleaning")
        
        # Standardize column names
        df['category'] = df[label_column]
        
        # Clean category names (remove any leading/trailing spaces, standardize)
        df['category'] = df['category'].str.strip().str.title()
        
        # Display category distribution
        print("\nCategory distribution before filtering:")
        print(df['category'].value_counts())
        
        # Filter categories with less than 2 samples
        category_counts = df['category'].value_counts()
        valid_categories = category_counts[category_counts >= 2].index
        df = df[df['category'].isin(valid_categories)]
        
        print("\nCategory distribution after filtering (min 2 samples):")
        print(df['category'].value_counts())
        
        print(f"\nPreprocessing complete. Final shape: {df.shape}")
        return df
    
    def save_processed_data(self, df, output_path='data/processed_data.csv'):
        """Save preprocessed data"""
        os.makedirs('data', exist_ok=True)
        # Save only necessary columns
        df[['cleaned_text', 'category']].to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
        return df

def main():
    """Test the preprocessor"""
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    df = preprocessor.preprocess_dataframe(df)
    preprocessor.save_processed_data(df)

if __name__ == "__main__":
    main()