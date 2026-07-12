# Embedding Service

> [!IMPORTANT]
> **This repository is archived and no longer maintained.** For running a blazing-fast local embedding service, we recommend using [Hugging Face Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference) instead.

A FastAPI-based service that provides text embeddings using various Sentence Transformer models. This service offers a simple API to generate embeddings for text inputs, supporting both single strings and batches.

## Features

- Multiple model support (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `paraphrase-multilingual-MiniLM-L12-v2`, and the 1024-dimension `mixedbread-ai/mxbai-embed-large-v1`)
- OpenAI-compatible API format
- Batched inference support
- Docker support
- Comprehensive test suite
- GitHub Actions CI/CD pipeline
- Token usage tracking

## Quick Start

### Option 1: Using Pre-built Docker Image

```bash
docker run -d --name embedding-service -p 8000:8000 ghcr.io/shaharia-lab/embedding-service:latest
```

### Option 2: Building Docker Image from Source

```bash
# Build the image
docker build -t embedding-service .

# Run the container
docker run -p 8000:8000 embedding-service
```

### Option 3: Local Development

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Run the server:
```bash
uvicorn app.main:app --reload
```

Once running, the API will be available at `http://localhost:8000`. You can visit `http://localhost:8000/docs` for interactive API documentation.

## API Usage

### Generate Embeddings

#### Using cURL

```bash
curl -X POST http://localhost:8000/v1/embeddings \
-H "Content-Type: application/json" \
-d '{
    "input": "Hello world",
    "model": "all-MiniLM-L6-v2"
}'
```

#### Using Python

```python
import requests

url = "http://localhost:8000/v1/embeddings"
payload = {
    "input": "Hello world",
    "model": "all-MiniLM-L6-v2"  # optional, defaults to all-MiniLM-L6-v2
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload)
embeddings = response.json()
```

### Batch Processing

```python
payload = {
    "input": ["Hello world", "Another text"],
    "model": "all-MiniLM-L6-v2"
}
```

### Higher-quality 1024-dimension embeddings

For higher-quality embeddings, use `mixedbread-ai/mxbai-embed-large-v1`, which
outputs 1024-dimension vectors. It is referenced by its full Hugging Face path
because it lives under the `mixedbread-ai` org rather than `sentence-transformers`:

```python
payload = {
    "input": "Hello world",
    "model": "mixedbread-ai/mxbai-embed-large-v1"
}
```

## Use OpenAI SDK

```js
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: "",
  baseURL: "http://localhost:8000/v1"
});

const embedding = await openai.embeddings.create({
  model: "all-MiniLM-L6-v2",
  input: "Your text string goes here",
  encoding_format: "float",
});

console.log(embedding);
```

## Configuration

All settings are environment variables with safe defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `UVICORN_WORKERS` | `2` | Number of uvicorn worker processes (Docker image only). **Each worker loads its own copy of the model** — set to `1` on memory-constrained hosts, especially when using `mixedbread-ai/mxbai-embed-large-v1` (~1.3 GB per worker). |
| `TORCH_NUM_THREADS` | `4` | Threads torch uses per worker for inference. |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | `4` | Caps BLAS/OpenMP threads so the service doesn't consume every visible core while idle. |
| `MAX_INPUT_CHARS` | `8192` | Maximum characters per input string; longer inputs are rejected with `422`. |
| `MAX_BATCH_SIZE` | `64` | Maximum items per input list; larger batches are rejected with `422`. |
| `REQUEST_TIMEOUT_SECONDS` | `30` | Requests exceeding this return `504` instead of hanging. The encode loop re-checks the deadline between chunks, so a timed-out request stops computing within one chunk instead of orphaning the whole batch. |
| `ENCODE_CHUNK_SIZE` | `8` | Inputs are encoded in sub-batches of this size, with timeout/disconnect checks between chunks. Bounds how much CPU work a cancelled request can leave running. |
| `MAX_CONCURRENT_INFERENCE` | `1` | Encodes running concurrently per worker. Additional requests queue (without burning CPU) instead of thrashing the shared torch thread pool — a client retrying timed-out requests can no longer stack computations. |
| `PRELOAD_MODELS` | *(empty)* | Comma-separated allowed model IDs whose files are downloaded to disk **during startup**, so the first request that switches to them loads from disk (~2s) instead of downloading over the network (which can exceed the request timeout). Only the default model stays in memory. |
| `HF_HUB_OFFLINE` | *(unset)* | Set to `1` in airgapped deployments (or when only baked/preloaded models are used): skips HuggingFace online checks, which otherwise stall boot for minutes when no network is reachable. |

The default model (`all-MiniLM-L6-v2`) is baked into the Docker image, so
containers boot in seconds with no network access. Example — also warm the
1024-dim model at boot:

```bash
docker run -p 8000:8000 -e PRELOAD_MODELS=mixedbread-ai/mxbai-embed-large-v1 \
  ghcr.io/shaharia-lab/embedding-service:latest
```

Input longer than the model's max sequence length is truncated server-side
before embedding, and the reported `prompt_tokens` reflects the truncated
(actually processed) token count.

### Container healthchecks

Point healthchecks at the dedicated `GET /health` endpoint (not `/docs`):

```yaml
healthcheck:
  test: ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)\""]
  interval: 30s
  timeout: 5s
  retries: 3
```

## API Documentation

Once the server is running, you can access:
- Interactive API documentation: http://localhost:8000/docs
- OpenAPI schema: http://localhost:8000/openapi.json

## Development

### Running Tests

```bash
# Run tests
pytest app/tests -v

# Run tests with coverage
pytest app/tests -v --cov=app --cov-report=term-missing
```

### Project Structure

```
embedding-service/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── tests/
│       ├── __init__.py
│       └── test_main.py
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
└── README.md
```

## CI/CD

The project uses GitHub Actions for:
- Running tests on pull requests and pushes to main
- Building and publishing Docker images on releases
- Automated testing and validation

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
