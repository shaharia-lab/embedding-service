import pytest
from fastapi.testclient import TestClient
from app.main import app, ALLOWED_MODELS, DEFAULT_MODEL, MODEL_PREFIX, get_full_model_path
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
    ("1024-dim org-prefixed model", "mixedbread-ai/mxbai-embed-large-v1", 200),
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


def test_get_full_model_path_prepends_prefix_for_bare_id():
    assert get_full_model_path("all-MiniLM-L6-v2") == f"{MODEL_PREFIX}all-MiniLM-L6-v2"


def test_get_full_model_path_keeps_org_prefixed_id():
    model_id = "mixedbread-ai/mxbai-embed-large-v1"
    assert get_full_model_path(model_id) == model_id


def test_model_switching(mock_sentence_transformer, mock_tokenizer):
    # Test switching between different valid models
    for model in ALLOWED_MODELS[:2]:  # Test with first two models
        response = client.post(
            "/v1/embeddings",
            json={"input": "Test text", "model": model}
        )
        assert response.status_code == 200
        assert response.json()["model"] == model

# --- Issue #10 regressions: truncation, size limits, event-loop liveness, timeout, switch lock ---

import threading
import time
from contextlib import contextmanager

import app.main as app_main
from app.main import MAX_BATCH_SIZE, MAX_INPUT_CHARS


@contextmanager
def patched_default_model(encode_side_effect=None, encode_return=None, token_ids=None):
    """Patch the module-level model/tokenizer instances (not the classes) so
    the default-model path runs against mocks, restoring globals afterwards."""
    with patch.object(app_main, 'model') as mock_model, \
            patch.object(app_main, 'tokenizer') as mock_tok, \
            patch.object(app_main, 'current_model_id', DEFAULT_MODEL):
        mock_model.max_seq_length = 512
        if encode_side_effect is not None:
            mock_model.encode.side_effect = encode_side_effect
        else:
            mock_model.encode.return_value = (
                encode_return if encode_return is not None else np.array([MOCK_EMBEDDING])
            )
        mock_tok.encode.return_value = token_ids if token_ids is not None else [1, 2, 3]
        yield mock_model, mock_tok


def test_overlength_input_truncated_in_usage_count():
    long_text = "word " * 1200  # well past any max_seq_length once tokenized
    with patched_default_model(token_ids=list(range(512))) as (mock_model, mock_tok):
        response = client.post(
            "/v1/embeddings",
            json={"input": long_text, "model": DEFAULT_MODEL}
        )
        assert response.status_code == 200
        assert response.json()["usage"]["prompt_tokens"] == 512
        mock_tok.encode.assert_called_once_with(long_text, truncation=True, max_length=512)


def test_input_string_over_max_chars_rejected():
    response = client.post(
        "/v1/embeddings",
        json={"input": "a" * (MAX_INPUT_CHARS + 1), "model": DEFAULT_MODEL}
    )
    assert response.status_code == 422


def test_input_list_item_over_max_chars_rejected():
    response = client.post(
        "/v1/embeddings",
        json={"input": ["ok", "a" * (MAX_INPUT_CHARS + 1)], "model": DEFAULT_MODEL}
    )
    assert response.status_code == 422


def test_batch_over_max_size_rejected():
    response = client.post(
        "/v1/embeddings",
        json={"input": ["hi"] * (MAX_BATCH_SIZE + 1), "model": DEFAULT_MODEL}
    )
    assert response.status_code == 422


def test_batch_at_max_size_accepted():
    batch = ["hi"] * MAX_BATCH_SIZE
    with patched_default_model(encode_return=np.array([MOCK_EMBEDDING for _ in batch])):
        response = client.post(
            "/v1/embeddings",
            json={"input": batch, "model": DEFAULT_MODEL}
        )
        assert response.status_code == 200
        assert len(response.json()["data"]) == MAX_BATCH_SIZE


def test_health_responsive_while_encode_is_blocked():
    started = threading.Event()
    release = threading.Event()

    def blocked_encode(texts, **kwargs):
        started.set()
        assert release.wait(timeout=10), "encode was never released"
        return np.array([MOCK_EMBEDDING])

    with patched_default_model(encode_side_effect=blocked_encode):
        with TestClient(app) as live_client:
            result = {}

            def do_post():
                result['response'] = live_client.post(
                    "/v1/embeddings",
                    json={"input": "hello", "model": DEFAULT_MODEL}
                )

            worker = threading.Thread(target=do_post)
            worker.start()
            try:
                assert started.wait(timeout=10), "embedding request never reached encode"
                start = time.monotonic()
                health = live_client.get("/health")
                elapsed = time.monotonic() - start
                assert health.status_code == 200
                assert elapsed < 2, f"/health took {elapsed:.2f}s while encode was in flight"
            finally:
                release.set()
                worker.join(timeout=10)
            assert result['response'].status_code == 200


def test_slow_request_times_out_with_504(monkeypatch):
    monkeypatch.setattr(app_main, 'REQUEST_TIMEOUT_SECONDS', 0.2)

    def slow_encode(texts, **kwargs):
        time.sleep(1.0)
        return np.array([MOCK_EMBEDDING])

    with patched_default_model(encode_side_effect=slow_encode):
        with TestClient(app) as live_client:
            response = live_client.post(
                "/v1/embeddings",
                json={"input": "hello", "model": DEFAULT_MODEL}
            )
    assert response.status_code == 504


def test_concurrent_model_switches_stay_consistent(mock_tokenizer):
    def make_model(path):
        time.sleep(0.05)  # widen the switch window
        instance = MagicMock()
        instance._loaded_path = path
        instance.max_seq_length = 512
        instance.encode.return_value = np.array([MOCK_EMBEDDING])
        return instance

    with patch('app.main.SentenceTransformer', side_effect=make_model), \
            patch.object(app_main, 'model'), \
            patch.object(app_main, 'tokenizer'), \
            patch.object(app_main, 'current_model_id', DEFAULT_MODEL):
        with TestClient(app) as live_client:
            responses = []

            def post(model_id):
                responses.append(live_client.post(
                    "/v1/embeddings",
                    json={"input": "hello", "model": model_id}
                ))

            threads = [
                threading.Thread(target=post, args=(model_id,))
                for model_id in ("all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2")
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)

        assert len(responses) == 2
        assert all(r.status_code == 200 for r in responses)
        # the loaded model must always match what current_model_id claims
        assert app_main.model._loaded_path == get_full_model_path(app_main.current_model_id)


def test_preload_models_warms_only_valid_non_default_models():
    with patch('app.main.SentenceTransformer') as mock_st, \
            patch('app.main.AutoTokenizer') as mock_tok, \
            patch.object(app_main, 'current_model_id', DEFAULT_MODEL):
        app_main._preload_models([" all-mpnet-base-v2 ", "not-a-model", "", " ", DEFAULT_MODEL])
        mock_st.assert_called_once_with(f"{MODEL_PREFIX}all-mpnet-base-v2")
        mock_tok.from_pretrained.assert_called_once_with(f"{MODEL_PREFIX}all-mpnet-base-v2")


def test_preload_models_noop_on_empty_env():
    with patch('app.main.SentenceTransformer') as mock_st, \
            patch('app.main.AutoTokenizer') as mock_tok:
        app_main._preload_models("".split(","))  # mirrors unset PRELOAD_MODELS
        mock_st.assert_not_called()
        mock_tok.from_pretrained.assert_not_called()
