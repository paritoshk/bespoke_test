import gzip
import asyncio
import os
import random
from typing import List, Tuple



async def load_common_crawl_samples(n_samples: int) -> List[str]:
    """
    Load random samples from robotstxt.paths.gz file.
    
    Args:
        n_samples: Number of samples to load
    Returns:
        List of text samples to use as negative examples
    """
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'robotstxt.paths.gz')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"robotstxt.paths.gz not found at {file_path}")
    
    all_samples = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        all_samples = f.readlines()
    
    # Clean the samples (remove empty lines and whitespace)
    all_samples = [line.strip() for line in all_samples if line.strip()]
    
    if len(all_samples) < n_samples:
        raise ValueError(f"Not enough samples in robotstxt.paths.gz. Need {n_samples}, but only found {len(all_samples)}")
    
    # Randomly select n_samples
    selected_samples = random.sample(all_samples, n_samples)
    return selected_samples


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

def load_evaluation_data(test_examples: List[str]) -> List[str]:
    """
    Prepare test data for evaluation.
    
    Args:
        test_examples: List of test examples
    
    Returns:
        List of processed test examples
    """
    return [example.strip() for example in test_examples]