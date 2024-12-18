import requests
import random

# Get a document from our prepared data to test
with open('data/train/positive/doc_0.txt', 'r') as f:
    test_doc = f.read()

# Use the UUID you got from the train endpoint
model_id = "c42970a2-6307-4f33-9439-51fc69909406"

# Test scoring
response = requests.post("http://localhost:8000/score", 
    json={
        "model_id": model_id,
        "documents": [test_doc]
    })

print(response.json())