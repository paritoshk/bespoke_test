import pytest
from fastapi.testclient import TestClient
import json

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_train_endpoint(client):
    """Test training with local data"""
    response = client.post("/train")
    assert response.status_code == 200
    data = response.json()
    assert "model_id" in data
    assert isinstance(data["model_id"], str)

def test_train_endpoint_with_file(client, tmp_path):
    """Test training with uploaded file"""
    # Create test JSONL file
    test_file = tmp_path / "test.jsonl"
    test_documents = [{"text": f"Test document {i}"} for i in range(20001)]
    
    with open(test_file, 'w') as f:
        for doc in test_documents:
            f.write(json.dumps(doc) + '\n')
    
    with open(test_file, 'rb') as f:
        response = client.post(
            "/train",
            files={"file": ("test.jsonl", f, "application/json")}
        )
    
    assert response.status_code == 200
    assert "model_id" in response.json()

def test_score_endpoint_with_invalid_model(client):
    """Test scoring with invalid model ID"""
    response = client.post(
        "/score",
        json={
            "model_id": "invalid-id",
            "documents": ["test document"]
        }
    )
    assert response.status_code == 404

def test_score_endpoint_with_valid_model(client):
    """Test scoring with valid model"""
    # First train model
    train_response = client.post("/train")
    assert train_response.status_code == 200
    model_id = train_response.json()["model_id"]
    
    # Then score
    response = client.post(
        "/score",
        json={
            "model_id": model_id,
            "documents": ["test document"]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "scores" in data
    assert len(data["scores"]) == 1
    assert isinstance(data["scores"][0], float)