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

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]