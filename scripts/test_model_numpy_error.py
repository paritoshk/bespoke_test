import requests
import random
import os
from tqdm import tqdm
import time

def test_model():
    # 1. Train the model
    print("Training model...")
    train_response = requests.post("http://localhost:8000/train")
    if train_response.status_code != 200:
        print("Training failed:", train_response.json())
        return
    
    model_id = train_response.json()['model_id']
    print(f"Model trained successfully. ID: {model_id}")
    
    # 2. Test scoring with various documents
    print("\nTesting scoring...")
    
    # Get some test documents
    test_docs = []
    
    # Get some positive examples
    positive_dir = "data/train/positive"
    for filename in random.sample(os.listdir(positive_dir), 5):
        with open(os.path.join(positive_dir, filename), 'r') as f:
            test_docs.append(("positive", f.read()))
            
    # Get some negative examples
    negative_dir = "data/train/negative"
    for filename in random.sample(os.listdir(negative_dir), 5):
        with open(os.path.join(negative_dir, filename), 'r') as f:
            test_docs.append(("negative", f.read()))
            
    # Score documents
    for label, doc in test_docs:
        response = requests.post("http://localhost:8000/score",
            json={
                "model_id": model_id,
                "documents": [doc]
            })
        
        if response.status_code == 200:
            score = response.json()['scores'][0]
            print(f"\nDocument type: {label}")
            print(f"Score: {score:.4f}")
            print(f"Text preview: {doc[:100]}...")
        else:
            print("Scoring failed:", response.json())

if __name__ == "__main__":
    test_model()