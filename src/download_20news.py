import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import os

def download_20newsgroups():
    """Download and save 20 Newsgroups dataset in the required format"""
    print("Downloading 20 Newsgroups dataset...")
    
    # Fetch training data
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    
    # Create dataframe
    df = pd.DataFrame({
        'description': newsgroups.data,
        'labels': [newsgroups.target_names[i] for i in newsgroups.target]
    })
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/bbc_news.csv', index=False)
    print(f"Dataset saved to data/bbc_news.csv")
    print(f"Shape: {df.shape}")
    print(f"Categories: {df['labels'].nunique()}")
    print(df['labels'].value_counts())
    
    return df

if __name__ == "__main__":
    download_20newsgroups()
    