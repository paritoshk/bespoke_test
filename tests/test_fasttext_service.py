import pytest
import os
from app.services.fasttext_service import FastTextService

@pytest.fixture
async def trained_model_id(service):
    """Create a trained model for testing"""
    model_id = await service.train_model()
    return model_id

@pytest.fixture
def service():
    """Create FastTextService instance"""
    service = FastTextService()
    # Create trained_models directory if it doesn't exist
    os.makedirs(service.models_dir, exist_ok=True)
    return service

@pytest.mark.asyncio
async def test_score_documents(service, trained_model_id, test_documents):
    """Test scoring with a trained model"""
    scores = await service.score_documents(trained_model_id, test_documents)
    assert len(scores) == len(test_documents)
    assert all(isinstance(score, float) for score in scores)
    assert all(0 <= score <= 1 for score in scores)

@pytest.mark.asyncio
async def test_model_persistence(service, trained_model_id):
    """Test model loading from disk"""
    # Clear any cached models
    service.models = {}
    
    # Try to load and use model
    test_docs = ["test document"]
    scores = await service.score_documents(trained_model_id, test_docs)
    assert len(scores) == 1
    assert isinstance(scores[0], float)
    assert 0 <= scores[0] <= 1

@pytest.mark.asyncio
async def test_invalid_model_id(service):
    """Test handling of invalid model IDs"""
    with pytest.raises(ValueError) as exc_info:
        await service.score_documents("invalid-model-id", ["test document"])
    assert "not found" in str(exc_info.value)