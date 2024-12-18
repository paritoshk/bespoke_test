import pytest
import os
from app.services.fasttext_service import FastTextService
import random

def get_random_documents(folder_path: str, num_docs: int = 3) -> list:
    """Get random documents from a data folder"""
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    selected_files = random.sample(all_files, min(num_docs, len(all_files)))
    documents = []
    for file_name in selected_files:
        with open(os.path.join(folder_path, file_name), 'r') as f:
            documents.append(f.read().strip())
    return documents

@pytest.fixture
def service():
    """Create FastTextService instance"""
    return FastTextService()

@pytest.fixture
def test_documents():
    """Get test documents from both positive and negative folders"""
    pos_docs = get_random_documents('data/train/positive', 2)
    neg_docs = get_random_documents('data/train/negative', 2)
    return pos_docs + neg_docs

@pytest.mark.asyncio
async def test_score_documents(service, test_documents):
    """Test scoring with an existing model"""
    # Get an existing model ID from trained_models directory
    model_files = [f for f in os.listdir('trained_models') if f.endswith('.bin')]
    assert len(model_files) > 0, "No trained models found in trained_models directory"
    
    model_id = model_files[0].replace('.bin', '')
    print(f"Testing with model: {model_id}")
    
    # Score the documents
    scores = await service.score_documents(model_id, test_documents)
    
    print("\nScoring Results:")
    for doc, score in zip(test_documents, scores):
        print(f"Score: {score:.4f} | Document preview: {doc[:100]}...")
    
    assert len(scores) == len(test_documents)
    assert all(isinstance(score, float) for score in scores)
    assert all(0 <= score <= 1 for score in scores)

@pytest.mark.asyncio
async def test_model_persistence(service):
    """Test model loading from disk"""
    # Get an existing model ID
    model_files = [f for f in os.listdir('trained_models') if f.endswith('.bin')]
    assert len(model_files) > 0, "No trained models found in trained_models directory"
    model_id = model_files[0].replace('.bin', '')
    
    # Clear any cached models
    service.models = {}
    
    # Get test documents
    test_docs = get_random_documents('data/train/positive', 1)
    scores = await service.score_documents(model_id, test_docs)
    
    assert len(scores) == 1
    assert isinstance(scores[0], float)
    assert 0 <= scores[0] <= 1

@pytest.mark.asyncio
async def test_invalid_model_id(service, test_documents):
    """Test handling of invalid model IDs"""
    with pytest.raises(ValueError) as exc_info:
        await service.score_documents("invalid-model-id", test_documents)
    assert "not found" in str(exc_info.value)