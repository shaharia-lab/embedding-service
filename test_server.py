from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI(title="Code Embedding Service")

# Load model globally for reuse
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")


class EmbeddingRequest(BaseModel):
    text: str


class BatchEmbeddingRequest(BaseModel):
    texts: List[str]


class EmbeddingResponse(BaseModel):
    embedding: List[float]
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


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)