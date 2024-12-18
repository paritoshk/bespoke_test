import pandas as pd
import os
import random
from pathlib import Path
from tqdm import tqdm

def prepare_data():
    """Use the downloaded parquet file to create training data"""
    # Use the specific parquet file we found
    parquet_file = "/Users/paritoshkulkarni/.cache/huggingface/hub/datasets--stanford-oval--ccnews/snapshots/d733e654c9a506df519e1a166a86c118c7657ce4/2024_0000.parquet"
    
    print(f"Reading parquet file: {os.path.basename(parquet_file)}")
    df = pd.read_parquet(parquet_file)
    
    # Setup directories
    base_dir = Path('data')
    train_dir = base_dir / 'train'
    positive_dir = train_dir / 'positive'
    negative_dir = train_dir / 'negative'

    # Create directories
    for dir_path in [positive_dir, negative_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Get text from the parquet file
    print("Extracting texts from parquet file...")
    texts = df['plain_text'].dropna().tolist()
    print(f"Found {len(texts)} total documents")
    
    # Filter valid documents
    print("Filtering valid documents...")
    valid_docs = []
    for text in tqdm(texts):
        text = str(text).strip()
        if len(text.split()) > 10:  # Basic validation
            valid_docs.append(text)
            if len(valid_docs) >= 40000:  # Stop once we have enough
                break
    
    print(f"Valid documents after filtering: {len(valid_docs)}")
    
    if len(valid_docs) < 40000:
        print(f"Warning: Only found {len(valid_docs)} valid documents")
        
    # Split into positive and negative
    random.shuffle(valid_docs)
    split_point = len(valid_docs) // 2
    positive_docs = valid_docs[:split_point]
    negative_docs = valid_docs[split_point:2*split_point]

    print("\nWriting positive documents...")
    for i, doc in enumerate(tqdm(positive_docs)):
        with open(positive_dir / f"doc_{i}.txt", 'w', encoding='utf-8') as f:
            f.write(doc)

    print("\nWriting negative documents...")
    for i, doc in enumerate(tqdm(negative_docs)):
        with open(negative_dir / f"doc_{i}.txt", 'w', encoding='utf-8') as f:
            f.write(doc)

    print(f"\nCreated {len(positive_docs)} positive and {len(negative_docs)} negative documents")
    print(f"Positive documents in: {positive_dir}")
    print(f"Negative documents in: {negative_dir}")

if __name__ == "__main__":
    prepare_data()