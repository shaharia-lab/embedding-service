from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Union, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer

app = FastAPI(title="Code Embedding Service")

# Define allowed models without the prefix
ALLOWED_MODELS = [
    'all-MiniLM-L6-v2',
    'all-mpnet-base-v2',
    'paraphrase-multilingual-MiniLM-L12-v2'
]

# Default model
DEFAULT_MODEL = 'all-MiniLM-L6-v2'

# Model prefix
MODEL_PREFIX = 'sentence-transformers/'

def get_full_model_path(model_id: str) -> str:
    """Convert short model ID to full path."""
    return f"{MODEL_PREFIX}{model_id}"

# Load model and tokenizer globally for reuse
current_model_id = DEFAULT_MODEL
model = SentenceTransformer(get_full_model_path(DEFAULT_MODEL))
tokenizer = AutoTokenizer.from_pretrained(get_full_model_path(DEFAULT_MODEL))

class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = DEFAULT_MODEL
    encoding_format: Literal["float"] = "float"

    @field_validator('model')
    def validate_model(cls, v):
        if v not in ALLOWED_MODELS:
            raise ValueError(
                f"Model '{v}' is not supported. Supported models are: {', '.join(ALLOWED_MODELS)}"
            )
        return v

    @field_validator('input')
    def validate_input(cls, v):
        if isinstance(v, str) and not v.strip():
            raise ValueError("Input string cannot be empty")
        if isinstance(v, list):
            if not v:
                raise ValueError("Input list cannot be empty")
            if not all(isinstance(item, str) and item.strip() for item in v):
                raise ValueError("All items in the input list must be non-empty strings")
        return v

class EmbeddingObject(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: List[float]
    index: int

    @field_validator('embedding')
    def validate_embedding(cls, v):
        if isinstance(v, np.ndarray):
            return [float(x) for x in v.flatten()]
        return v

class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[EmbeddingObject]
    model: str
    usage: Usage

def count_tokens(texts: List[str]) -> int:
    """Count the total number of tokens in a list of texts."""
    return sum(len(tokenizer.encode(text)) for text in texts)

def create_usage_info(texts: List[str]) -> Usage:
    """Create usage information with token counts."""
    prompt_tokens = count_tokens(texts)
    return Usage(
        prompt_tokens=prompt_tokens,
        total_tokens=prompt_tokens
    )

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    try:
        # Handle both single string and list of strings
        inputs = request.input if isinstance(request.input, list) else [request.input]

        # Check if we need to load a different model
        global model, tokenizer, current_model_id
        if request.model != current_model_id:
            full_model_path = get_full_model_path(request.model)
            model = SentenceTransformer(full_model_path)
            tokenizer = AutoTokenizer.from_pretrained(full_model_path)
            current_model_id = request.model

        # Generate embeddings
        embeddings = model.encode(inputs)

        # Convert to list format if it's a single embedding
        if len(inputs) == 1:
            embeddings = [embeddings]

        # Create embedding objects
        embedding_objects = []
        for idx, embedding in enumerate(embeddings):
            embedding_list = [float(x) for x in embedding.flatten()]
            obj = EmbeddingObject(
                embedding=embedding_list,
                index=idx
            )
            embedding_objects.append(obj)

        # Calculate token usage
        usage = create_usage_info(inputs)

        return EmbeddingResponse(
            data=embedding_objects,
            model=request.model,
            usage=usage
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}