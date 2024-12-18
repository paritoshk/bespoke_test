import pytest
from fastapi.testclient import TestClient
from app.main import app
import gzip
import tempfile
import os
import pytest_asyncio

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client

@pytest.fixture
def test_gz_file(tmp_path):
    """Create a temporary gz file with test data"""
    file_path = tmp_path / "test_robotstxt.paths.gz"
    test_data = [f"test document {i}\n" for i in range(50000)]  # Create enough test documents
    
    with gzip.open(file_path, 'wt') as f:
        f.writelines(test_data)
    
    return str(file_path)