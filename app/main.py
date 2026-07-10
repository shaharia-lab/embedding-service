import asyncio
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Union, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from transformers import AutoTokenizer

# Runtime configuration (overridable via environment)
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "8192"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "64"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "30"))
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "4"))
ENCODE_CHUNK_SIZE = int(os.environ.get("ENCODE_CHUNK_SIZE", "8"))
MAX_CONCURRENT_INFERENCE = int(os.environ.get("MAX_CONCURRENT_INFERENCE", "1"))

# Left unpinned, torch/BLAS grab every visible core even while idle
torch.set_num_threads(TORCH_NUM_THREADS)

app = FastAPI(title="Code Embedding Service")

# Define allowed models without the prefix
ALLOWED_MODELS = [
    'all-MiniLM-L6-v2',
    'all-mpnet-base-v2',
    'paraphrase-multilingual-MiniLM-L12-v2',
    'mixedbread-ai/mxbai-embed-large-v1'
]

# Default model
DEFAULT_MODEL = 'all-MiniLM-L6-v2'

# Model prefix
MODEL_PREFIX = 'sentence-transformers/'

def get_full_model_path(model_id: str) -> str:
    """Convert short model ID to full path.

    Model ids that already include an org prefix (i.e. contain a '/') are
    returned unchanged; bare ids are assumed to live under MODEL_PREFIX.
    """
    if '/' in model_id:
        return model_id
    return f"{MODEL_PREFIX}{model_id}"

# Load model and tokenizer globally for reuse
current_model_id = DEFAULT_MODEL
model = SentenceTransformer(get_full_model_path(DEFAULT_MODEL))
tokenizer = AutoTokenizer.from_pretrained(get_full_model_path(DEFAULT_MODEL))

# Guards the model/tokenizer globals: once encode() runs off the event loop,
# requests interleave and an unlocked swap is a real data race
model_lock = asyncio.Lock()

# CPU inference gains nothing from concurrent encodes sharing the same torch
# thread pool; the gate makes retries queue instead of stacking computations
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)


def _preload_models(model_ids: List[str]) -> None:
    """Warm the on-disk cache for extra models at startup so the first request
    that switches to them pays a ~2s disk load instead of a network download
    (which can exceed REQUEST_TIMEOUT_SECONDS). Instances are discarded —
    only the default model stays resident."""
    for model_id in model_ids:
        model_id = model_id.strip()
        if not model_id:
            continue
        if model_id not in ALLOWED_MODELS:
            print(f"WARNING: PRELOAD_MODELS entry '{model_id}' is not an allowed model; skipping")
            continue
        if model_id == current_model_id:
            continue
        full_model_path = get_full_model_path(model_id)
        SentenceTransformer(full_model_path)
        AutoTokenizer.from_pretrained(full_model_path)


_preload_models(os.environ.get("PRELOAD_MODELS", "").split(","))


@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    # Bounds client-observed latency only: cancellation cannot kill an
    # inference thread already running; the input limits keep that work bounded
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"detail": f"Request exceeded the {REQUEST_TIMEOUT_SECONDS}s timeout"},
        )


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
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Input string cannot be empty")
            if len(v) > MAX_INPUT_CHARS:
                raise ValueError(
                    f"Input exceeds the maximum length of {MAX_INPUT_CHARS} characters"
                )
        if isinstance(v, list):
            if not v:
                raise ValueError("Input list cannot be empty")
            if len(v) > MAX_BATCH_SIZE:
                raise ValueError(
                    f"Input list exceeds the maximum batch size of {MAX_BATCH_SIZE} items"
                )
            if not all(isinstance(item, str) and item.strip() for item in v):
                raise ValueError("All items in the input list must be non-empty strings")
            for item in v:
                if len(item) > MAX_INPUT_CHARS:
                    raise ValueError(
                        f"Input items exceed the maximum length of {MAX_INPUT_CHARS} characters"
                    )
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

def count_tokens(texts: List[str], active_tokenizer, max_seq_length: int) -> int:
    """Count the total number of tokens in a list of texts.

    Truncates to the model's max sequence length so the count reflects what
    the model actually processed (and never warns on over-length input).
    """
    return sum(
        len(active_tokenizer.encode(text, truncation=True, max_length=max_seq_length))
        for text in texts
    )

def create_usage_info(texts: List[str], active_tokenizer, max_seq_length: int) -> Usage:
    """Create usage information with token counts."""
    prompt_tokens = count_tokens(texts, active_tokenizer, max_seq_length)
    return Usage(
        prompt_tokens=prompt_tokens,
        total_tokens=prompt_tokens
    )

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest, http_request: Request):
    try:
        # Handle both single string and list of strings
        inputs = request.input if isinstance(request.input, list) else [request.input]

        # The timeout middleware can only cancel the request, never a thread
        # already computing — so the encode loop below re-checks this deadline
        # between chunks and abandons the rest itself
        deadline = asyncio.get_running_loop().time() + REQUEST_TIMEOUT_SECONDS

        # Snapshot a consistent (model, tokenizer) pair under the lock;
        # reloads run in a thread so the event loop stays responsive
        global model, tokenizer, current_model_id
        async with model_lock:
            if request.model != current_model_id:
                full_model_path = get_full_model_path(request.model)
                new_model = await asyncio.to_thread(SentenceTransformer, full_model_path)
                new_tokenizer = await asyncio.to_thread(
                    AutoTokenizer.from_pretrained, full_model_path
                )
                model, tokenizer, current_model_id = new_model, new_tokenizer, request.model
            active_model, active_tokenizer = model, tokenizer

        # Run inference off the event loop so healthchecks and other requests
        # are never blocked by a slow forward pass. Chunking bounds the work
        # orphaned by a timeout to a single chunk instead of the whole batch.
        async with inference_semaphore:
            embedding_rows = []
            for start in range(0, len(inputs), ENCODE_CHUNK_SIZE):
                if asyncio.get_running_loop().time() >= deadline:
                    raise HTTPException(
                        status_code=504,
                        detail=f"Request exceeded the {REQUEST_TIMEOUT_SECONDS}s timeout",
                    )
                if await http_request.is_disconnected():
                    raise HTTPException(status_code=499, detail="Client disconnected")
                chunk = inputs[start:start + ENCODE_CHUNK_SIZE]
                chunk_embeddings = await asyncio.to_thread(active_model.encode, chunk)
                embedding_rows.extend(np.atleast_2d(np.asarray(chunk_embeddings)))

        # Create embedding objects
        embedding_objects = []
        for idx, embedding in enumerate(embedding_rows):
            embedding_list = [float(x) for x in embedding.flatten()]
            obj = EmbeddingObject(
                embedding=embedding_list,
                index=idx
            )
            embedding_objects.append(obj)

        # Calculate token usage
        usage = create_usage_info(inputs, active_tokenizer, active_model.max_seq_length)

        return EmbeddingResponse(
            data=embedding_objects,
            model=request.model,
            usage=usage
        )
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
