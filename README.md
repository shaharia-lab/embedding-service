# Embedding Service

A FastAPI-based service that provides text embeddings using various Sentence Transformer models. This service offers a simple API to generate embeddings for text inputs, supporting both single strings and batches.

## Features

- Multiple model support (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `paraphrase-multilingual-MiniLM-L12-v2`)
- OpenAI-compatible API format
- Batched inference support
- Docker support
- Comprehensive test suite
- GitHub Actions CI/CD pipeline
- Token usage tracking

## Quick Start

### Option 1: Using Pre-built Docker Image

```bash
docker run -p 8000:8000 ghcr.io/shaharia-lab/embedding-service:latest
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