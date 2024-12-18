import pytest
from fastapi.testclient import TestClient

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_train_endpoint(client):
    response = client.post("/train")
    assert response.status_code == 200
    data = response.json()
    assert "model_id" in data
    assert "metrics" in data
    assert isinstance(data["metrics"], dict)
    assert "num_test_samples" in data["metrics"]
    assert "predictions" in data["metrics"]

def test_score_endpoint_with_invalid_model(client):
    request_data = {
        "model_id": "non-existent-model",
        "documents": ["test document"]
    }
    response = client.post("/score", json=request_data)
    assert response.status_code == 404

def test_score_endpoint_with_valid_model(client):
    # First train a model
    train_response = client.post("/train")
    assert train_response.status_code == 200
    model_id = train_response.json()["model_id"]
    
    # Then test scoring
    request_data = {
        "model_id": model_id,
        "documents": ["test document to score"]
    }
    response = client.post("/score", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "scores" in data
    assert isinstance(data["scores"], list)
    assert all(isinstance(score, float) for score in data["scores"])
    assert all(0 <= score <= 1 for score in data["scores"])