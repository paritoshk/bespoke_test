from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import List
from pydantic import BaseModel
import json
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
async def train_model(file: UploadFile = None):
    """Train a FastText classifier using either uploaded positive documents or local data."""
    try:
        if file is None:
            print("DEBUG: No file provided, using local data")
            model_id = await fasttext_service.train_model(positive_documents=None)
            return TrainResponse(model_id=model_id)

        print(f"DEBUG: Received file: {file.filename}")
        
        # Read and parse uploaded file
        content = await file.read()
        lines = content.decode('utf-8').splitlines()
        print(f"DEBUG: Number of lines read: {len(lines)}")
        
        positive_documents = []
        for i, line in enumerate(lines):
            try:
                doc = json.loads(line)
                print(f"DEBUG: Processing line {i}, type: {type(doc)}")
                if isinstance(doc, str):
                    positive_documents.append(doc)
                elif isinstance(doc, dict) and 'text' in doc:
                    positive_documents.append(doc['text'])
                else:
                    print(f"DEBUG: Skipping line {i}: Invalid format - {doc}")
            except json.JSONDecodeError as e:
                print(f"DEBUG: JSON decode error at line {i}: {str(e)}")
                continue
                
        print(f"DEBUG: Parsed {len(positive_documents)} valid documents")
        
        if not positive_documents:
            print("DEBUG: No valid documents found in file")
            raise HTTPException(
                status_code=400,
                detail="No valid documents found in uploaded file"
            )

        model_id = await fasttext_service.train_model(positive_documents=positive_documents)
        return TrainResponse(model_id=model_id)
        
    except Exception as e:
        print(f"DEBUG: Error during training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/score", response_model=ScoreResponse)
async def score_documents(request: ScoreRequest):
    """Score documents using a trained FastText classifier."""
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