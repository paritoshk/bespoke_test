# FastText Classification Service

A high-performance, scalable FastText classification service built with FastAPI and Python. This service provides endpoints for training quality classifiers and scoring documents at scale, particularly useful for data pipeline quality filtering.

## Features

- **Efficient Training Pipeline**: Train FastText classifiers using positive examples and automatically sampled negative examples from Common Crawl
- **High-Performance Scoring**: Score large batches of documents efficiently using trained models
- **REST API Interface**: Clean API interface with FastAPI, including automatic OpenAPI documentation
- **Scalable Architecture**: Designed for handling large-scale document processing
- **Model Persistence**: Trained models are persisted and can be reused across sessions

## Technical Implementation

### Architecture

The service is structured into three main components:

1. **API Layer** (`app/main.py`):
   - FastAPI application handling HTTP requests
   - Input validation using Pydantic models
   - Error handling and response formatting

2. **Service Layer** (`app/services/fasttext_service.py`):
   - FastText model training and management
   - Document scoring logic
   - Model persistence handling

3. **Data Layer** (`app/utils/data_loader.py`):
   - Common Crawl data sampling
   - Training data preparation

### API Endpoints

#### POST /train
- Accepts positive training documents (minimum 20k examples)
- Automatically samples negative examples from Common Crawl
- Returns a UUID for the trained model

```python
Response:
{
    "model_id": "uuid-string"
}
```

#### POST /score
- Accepts a batch of documents and a model ID
- Returns classification scores for each document

```python
Request:
{
    "model_id": "uuid-string",
    "documents": ["doc1", "doc2", ...]
}

Response:
{
    "scores": [0.92, 0.45, ...]
}
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fasttext-service.git
cd fasttext-service
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
uvicorn app.main:app --reload
```

2. Access the API documentation:
- Navigate to `http://localhost:8000/docs` for the Swagger UI
- Navigate to `http://localhost:8000/redoc` for the ReDoc documentation

### Example Usage

Training a model:
```python
import requests

with open('positive_examples.txt', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/train',
        files={'documents': f}
    )
model_id = response.json()['model_id']
```

Scoring documents:
```python
response = requests.post(
    'http://localhost:8000/score',
    json={
        'model_id': model_id,
        'documents': ['document to classify', 'another document']
    }
)
scores = response.json()['scores']
```

## Technical Details

### FastText Configuration

The FastText model is configured for optimal performance in document classification:
- Word n-grams (n=2) for capturing short phrases
- Learning rate of 0.5 for stable convergence
- 25 training epochs for model robustness
- Minimum word count of 1 to handle rare terms

### Scaling Considerations

The service is designed with scalability in mind:
- Asynchronous request handling
- Efficient model loading/unloading
- Batch processing capabilities
- Persistent model storage

### Error Handling

Comprehensive error handling is implemented for:
- Invalid input validation
- Model not found scenarios
- Training data requirements
- Server-side processing errors

## Testing

Run the test suite:
```bash
pytest tests/
```

## Analyzing Results

To analyze the results of the training, run the following command:
```bash
python scripts/analyze_results.py
```

## Bugs

### Technical Challenges Addressed:

NumPy Compatibility Issue: Fixed incompatibility with newer NumPy versions by patching FastText's predict method
Data Processing: Built robust data loading from positive/negative examples
Model Persistence: Implemented proper model saving and loading
Async Support: Built async-compatible API endpoints


### Architecture Decisions:

Service-based design separating concerns
Comprehensive logging and monitoring
Proper error handling and validation
Clean separation of training and inference


### Implementation Details:

Used FastText for efficient text classification
Built binary classifier (positive/negative)
Implemented model versioning with UUIDs
Added performance monitoring and visualization



### To explain the approach:

Data Pipeline:

Organized training data into positive/negative examples
Built efficient data loading mechanism
Implemented proper text cleaning and normalization


### Model Training:

Used FastText for efficient text classification
Implemented proper hyperparameter configuration
Added comprehensive logging and monitoring
Built model persistence with versioning


### Inference:

Efficient document scoring
Proper error handling
Async support for scalability
Clean API interface


### Quality Assurance:

Comprehensive test suite
Performance monitoring
Error handling
Data validation

## Future Improvements
    
- Add model versioning
- Implement distributed training
- Add model performance metrics
- Add data preprocessing pipeline
- Implement model caching strategy
- Add support for custom negative examples

## License

MIT License - See LICENSE file for details


