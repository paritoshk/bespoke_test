import pytest
from app.models import DataProcessor

def test_prepare_training_data():
    processor = DataProcessor()
    positive = ["pos1", "pos2"]
    negative = ["neg1", "neg2"]
    
    training_data = processor.prepare_training_data(positive, negative)
    assert len(training_data) == 4
    assert any("__label__positive" in x for x in training_data)
    assert any("__label__negative" in x for x in training_data)

def test_load_evaluation_data():
    processor = DataProcessor()
    test_data = [" doc1 ", "doc2 ", " doc3 "]
    processed = processor.load_evaluation_data(test_data)
    
    assert len(processed) == len(test_data)
    assert all(doc == doc.strip() for doc in processed)

def test_load_and_split_data(test_gz_file):
    processor = DataProcessor()
    train_data, test_data = processor.load_and_split_data(test_gz_file)
    
    assert len(train_data) >= 20000
    assert len(test_data) >= 20000
    assert all(isinstance(doc, str) for doc in train_data)
    assert all(isinstance(doc, str) for doc in test_data)