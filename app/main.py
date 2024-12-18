from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
from .services.fasttext_service import FastTextService

app = FastAPI(title="FastText Classification Service")
fasttext_service = FastTextService()

class ScoreRequest(BaseModel):
    model_id: str
    documents: List[str]

class ScoreResponse(BaseModel):
    scores: List[float]

class TrainResponse(BaseModel):
    model_id: str

@app.post("/train", response_model=TrainResponse)
async def train_model():
    """
    Train a FastText classifier using documents from robotstxt.paths.gz.
    Uses 20K documents as positive examples and 20K as negative examples.
    Returns a UUID for the trained model.
    """
    try:
        model_id = await fasttext_service.train_model()
        return TrainResponse(model_id=model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score", response_model=ScoreResponse)
async def score_documents(request: ScoreRequest):
    """
    Score documents using a trained FastText classifier.
    Returns probability scores for documents belonging to the positive class.
    """
    try:
        scores = await fasttext_service.score_documents(
            request.model_id, 
            request.documents
        )
        return ScoreResponse(scores=scores)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))