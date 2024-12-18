from pydantic import BaseModel
from typing import List, Tuple, Dict
import gzip
import random
import os

# Pydantic models for API requests/responses
class ScoreRequest(BaseModel):
    model_id: str
    documents: List[str]

class ScoreResponse(BaseModel):
    scores: List[float]

class TrainResponse(BaseModel):
    model_id: str
    metrics: dict

# Data handling class
class DataProcessor:
    @staticmethod
    def load_and_split_data(file_path: str, n_samples: int = 40000) -> Tuple[List[str], List[str]]:
        """
        Load data from gz file and split into training and testing sets.
        
        Args:
            file_path: Path to the robotstxt.paths.gz file
            n_samples: Total number of samples to use (default 40k for 20k each in train/test)
        
        Returns:
            Tuple of (training_data, testing_data)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read all lines from gz file
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if len(lines) < n_samples:
            raise ValueError(f"Not enough samples in file. Found {len(lines)}, need {n_samples}")
        
        # Randomly sample and split data
        selected_data = random.sample(lines, n_samples)
        split_point = len(selected_data) // 2  # Split 50/50
        
        return selected_data[:split_point], selected_data[split_point:]

    @staticmethod
    def prepare_training_data(positive_examples: List[str], negative_examples: List[str]) -> List[str]:
        """
        Prepare data in FastText format with labels.
        
        Args:
            positive_examples: List of positive examples
            negative_examples: List of negative examples
        
        Returns:
            List of labeled examples in FastText format
        """
        training_data = []
        
        # Add positive examples with label
        for example in positive_examples:
            training_data.append(f"__label__positive {example}")
        
        # Add negative examples with label
        for example in negative_examples:
            training_data.append(f"__label__negative {example}")
        
        # Shuffle the combined data
        random.shuffle(training_data)
        
        return training_data

    @staticmethod
    def load_evaluation_data(test_examples: List[str]) -> List[str]:
        """
        Prepare test data for evaluation.
        
        Args:
            test_examples: List of test examples
        
        Returns:
            List of processed test examples
        """
        return [example.strip() for example in test_examples]