from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI(title="Code Embedding Service")

# Load model globally for reuse
model = SentenceTransformer('all-MiniLM-L6-v2')


class EmbeddingRequest(BaseModel):
    text: str


class BatchEmbeddingRequest(BaseModel):
    texts: List[str]


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimension: int


class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int


@app.post("/embed", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    try:
        # Generate embedding
        embedding = model.encode(request.text)
        # Convert to Python list for JSON serialization
        embedding_list = embedding.tolist()

        return EmbeddingResponse(
            embedding=embedding_list,
            dimension=len(embedding_list)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed-batch", response_model=BatchEmbeddingResponse)
async def create_batch_embedding(request: BatchEmbeddingRequest):
    try:
        # Generate embeddings for multiple texts
        embeddings = model.encode(request.texts)
        # Convert to Python list for JSON serialization
        embeddings_list = embeddings.tolist()

        return BatchEmbeddingResponse(
            embeddings=embeddings_list,
            dimension=len(embeddings_list[0]) if embeddings_list else 0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}