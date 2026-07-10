FROM python:3.10-slim

# Add standard GitHub container labels
# https://github.com/opencontainers/image-spec/blob/main/annotations.md
LABEL org.opencontainers.image.source="https://github.com/shaharia-lab/embedding-service"
LABEL org.opencontainers.image.description="FastAPI service for generating text embeddings using sentence-transformers"
LABEL org.opencontainers.image.licenses="MIT"

# Add GitHub-specific labels
LABEL org.opencontainers.image.documentation="https://github.com/shaharia-lab/embedding-service/blob/master/README.md"
LABEL org.opencontainers.image.url="https://github.com/shaharia-lab/embedding-service"
LABEL org.opencontainers.image.vendor="GitHub"
LABEL org.opencontainers.image.title="Code Embedding Service"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Set longer timeout and retry mechanism for pip
RUN pip install --no-cache-dir --timeout 200 --retries 3 torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --timeout 200 --retries 3 -r requirements.txt

# Bake the default model into the image: boot needs no network access and the
# service is ready in seconds. Mounting a volume over /root/.cache/huggingface
# shadows this layer — only do that to persist additionally preloaded models.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app/ ./app/

EXPOSE 8000

# Pin BLAS/OpenMP threads so idle CPU stays bounded; tune per host.
# Each uvicorn worker loads its own model copy — lower UVICORN_WORKERS to 1
# on memory-constrained hosts, especially with mixedbread-ai/mxbai-embed-large-v1.
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    UVICORN_WORKERS=2

# Shell form so UVICORN_WORKERS expands; sh exec's uvicorn as PID 1
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS}