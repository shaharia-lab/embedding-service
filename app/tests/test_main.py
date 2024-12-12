import pytest
from fastapi.testclient import TestClient
from app.main import app, ALLOWED_MODELS, DEFAULT_MODEL
import numpy as np
from unittest.mock import patch, MagicMock

client = TestClient(app)

# Mock a more realistic embedding size (384 dimensions)
MOCK_EMBEDDING = np.zeros(384)  # Create a zero vector of correct size

# Test data
VALID_INPUTS = [
    ("single string input", "Hello world"),
    ("list of strings", ["Hello", "world"]),
    ("longer text", "This is a longer piece of text that needs to be embedded"),
]

INVALID_INPUTS = [
    ("empty string", "", 422),  # Changed to 422 to match FastAPI's validation
    ("empty list", [], 422),  # Changed to 422 to match FastAPI's validation
    ("invalid input type", 123, 422),
    ("list with non-string", ["Hello", 123], 422),
]

MODEL_TESTS = [
    ("default model", DEFAULT_MODEL, 200),
    ("alternative valid model", "all-mpnet-base-v2", 200),
    ("invalid model", "invalid-model", 422),  # Changed to 422 to match FastAPI's validation
]


@pytest.fixture
def mock_sentence_transformer():
    with patch('app.main.SentenceTransformer') as mock:
        model_instance = MagicMock()

        # Return correct size embedding for both single and batch inputs
        def mock_encode(texts):
            if isinstance(texts, str):
                return MOCK_EMBEDDING
            return np.array([MOCK_EMBEDDING for _ in texts])

        model_instance.encode.side_effect = mock_encode
        mock.return_value = model_instance
        yield mock


@pytest.fixture
def mock_tokenizer():
    with patch('app.main.AutoTokenizer') as mock:
        tokenizer_instance = MagicMock()
        tokenizer_instance.encode.return_value = [1, 2, 3]  # Mock token ids
        mock.from_pretrained.return_value = tokenizer_instance
        yield mock


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.parametrize("test_name,input_text", VALID_INPUTS)
def test_valid_embedding_requests(test_name, input_text, mock_sentence_transformer, mock_tokenizer):
    response = client.post(
        "/v1/embeddings",
        json={"input": input_text, "model": DEFAULT_MODEL}
    )

    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "object" in data
    assert "data" in data
    assert "model" in data
    assert "usage" in data

    # Check model name
    assert data["model"] == DEFAULT_MODEL

    # Check embeddings
    embeddings = data["data"]
    if isinstance(input_text, list):
        assert len(embeddings) == len(input_text)
    else:
        assert len(embeddings) == 1

    # Check embedding format
    for emb in embeddings:
        assert "embedding" in emb
        assert "index" in emb
        assert len(emb["embedding"]) == len(MOCK_EMBEDDING)
        assert all(isinstance(x, float) for x in emb["embedding"])


@pytest.mark.parametrize("test_name,input_text,expected_status", INVALID_INPUTS)
def test_invalid_embedding_requests(test_name, input_text, expected_status, mock_sentence_transformer, mock_tokenizer):
    response = client.post(
        "/v1/embeddings",
        json={"input": input_text, "model": DEFAULT_MODEL}
    )
    assert response.status_code == expected_status


@pytest.mark.parametrize("test_name,model,expected_status", MODEL_TESTS)
def test_model_validation(test_name, model, expected_status, mock_sentence_transformer, mock_tokenizer):
    response = client.post(
        "/v1/embeddings",
        json={"input": "Test text", "model": model}
    )
    assert response.status_code == expected_status


def test_usage_calculation(mock_sentence_transformer, mock_tokenizer):
    input_text = "Test text"
    response = client.post(
        "/v1/embeddings",
        json={"input": input_text, "model": DEFAULT_MODEL}
    )

    assert response.status_code == 200
    data = response.json()

    # Check usage statistics
    assert "usage" in data
    usage = data["usage"]
    assert "prompt_tokens" in usage
    assert "total_tokens" in usage
    assert usage["prompt_tokens"] == usage["total_tokens"]
    assert isinstance(usage["prompt_tokens"], int)


def test_model_switching(mock_sentence_transformer, mock_tokenizer):
    # Test switching between different valid models
    for model in ALLOWED_MODELS[:2]:  # Test with first two models
        response = client.post(
            "/v1/embeddings",
            json={"input": "Test text", "model": model}
        )
        assert response.status_code == 200
        assert response.json()["model"] == model