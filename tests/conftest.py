import pytest
from fastapi.testclient import TestClient
from app.main import app
import os
import tempfile
import gzip

@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)

@pytest.fixture
def test_documents():
    """Get sample test documents from data directory"""
    # Get one positive and one negative document
    documents = []
    try:
        with open('data/train/positive/doc_0.txt', 'r') as f:
            documents.append(f.read().strip())
        with open('data/train/negative/doc_0.txt', 'r') as f:
            documents.append(f.read().strip())
    except Exception as e:
        documents = ["Test positive document", "Test negative document"]
    return documents

@pytest.fixture
def test_gz_file():
    """Create a temporary gz file for testing"""
    # Create temp file
    with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as temp_file:
        with gzip.open(temp_file.name, 'wt') as f:
            for i in range(50000):  # Enough documents for testing
                f.write(f"Test document {i}\n")
        yield temp_file.name
    # Cleanup
    os.unlink(temp_file.name)