"""Tests for the tokenizer and the 384-dim embedding generator."""
import math

import pytest

from app.services import embedding_service
from app.services.embedding_service import (
    EMBEDDING_DIM,
    EmbeddingDimensionError,
    benchmark_embeddings,
    get_embedding,
    get_embeddings,
    validate_embedding,
)
from app.services.tokenizer import CLS_ID, PAD_ID, SEP_ID, split_subwords, tokenize
from app.services.vector_search import cosine_similarity

SAMPLE_TICKETS = [
    "Cannot log into my account after password reset",
    "Invoice shows the wrong billing amount for November",
    "App crashes with ERR_404 when opening settings",
    "How do I export my data to CSV?",
    "Payment failed twice but my card was charged",
    "SAML SSO login timeout error 500",
    "Please delete my account and personal data",
    "The dashboard is very slow to load reports",
    "Two-factor authentication code never arrives",
    "Refund request for duplicate subscription charge",
]
BATCH_OF_50 = [f"{text} (ticket #{i})" for i, text in enumerate(SAMPLE_TICKETS * 5)]


# --- Tokenizer ---------------------------------------------------------------------------
def test_tokenizer_adds_special_tokens_and_pads():
    encoded = tokenize("Reset my password", max_length=16)
    assert encoded.input_ids[0] == CLS_ID
    assert len(encoded.input_ids) == len(encoded.attention_mask) == 16
    real = sum(encoded.attention_mask)
    assert encoded.input_ids[real - 1] == SEP_ID
    assert encoded.input_ids[real:] == [PAD_ID] * (16 - real)
    assert encoded.attention_mask == [1] * real + [0] * (16 - real)


def test_tokenizer_truncates_long_text():
    encoded = tokenize("word " * 500, max_length=32)
    assert len(encoded.input_ids) == 32
    assert sum(encoded.attention_mask) == 32
    assert encoded.input_ids[-1] == SEP_ID


def test_tokenizer_is_deterministic():
    assert tokenize("Billing issue").input_ids == tokenize("Billing issue").input_ids


def test_subword_split():
    assert split_subwords("authentication") == ["auth", "##enti", "##cati", "##on"]
    assert split_subwords("sso") == ["sso"]


# --- Embedding dimensions ------------------------------------------------------------------
EDGE_CASES = {
    "empty": "",
    "whitespace": "   ",
    "single_char": "x",
    "very_long_1000_words": "word " * 1000,
    "unicode_emoji": "登录失败 🔐 émoji",
}


@pytest.mark.parametrize(
    "text",
    SAMPLE_TICKETS + list(EDGE_CASES.values()),
    ids=[f"ticket_{i}" for i in range(len(SAMPLE_TICKETS))] + list(EDGE_CASES.keys()),
)
def test_every_embedding_has_384_float_dimensions(text):
    vector = get_embedding(text)
    assert len(vector) == EMBEDDING_DIM == 384
    assert all(isinstance(v, float) for v in vector)
    assert all(math.isfinite(v) for v in vector)


def test_batch_of_50_has_uniform_shape():
    vectors = get_embeddings(BATCH_OF_50)
    assert len(vectors) == 50
    assert {len(v) for v in vectors} == {384}


def test_embeddings_are_unit_length():
    vector = get_embedding("Payment failed twice")
    assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0)


def test_empty_text_returns_zero_vector():
    assert get_embedding("") == [0.0] * 384


# --- Embedding behaviour --------------------------------------------------------------------
def test_embeddings_are_deterministic():
    assert get_embedding("Refund my order") == get_embedding("Refund my order")


def test_different_texts_give_different_embeddings():
    assert get_embedding("billing question") != get_embedding("server crashed")


def test_shared_words_score_higher_than_unrelated_text():
    base = get_embedding("billing question about my invoice")
    related = get_embedding("billing problem with invoice")
    unrelated = get_embedding("server crashed overnight")
    assert cosine_similarity(base, related) > cosine_similarity(base, unrelated)


# --- Dimension guard --------------------------------------------------------------------------
def test_validate_embedding_rejects_wrong_dimension():
    with pytest.raises(EmbeddingDimensionError):
        validate_embedding([0.1] * 383)
    with pytest.raises(EmbeddingDimensionError):
        validate_embedding([0.1] * 385)


def test_validate_embedding_rejects_nan():
    with pytest.raises(EmbeddingDimensionError):
        validate_embedding([float("nan")] + [0.0] * 383)


def test_get_embedding_raises_if_backend_returns_wrong_shape(monkeypatch):
    monkeypatch.setattr(embedding_service, "_mock_embedding", lambda text: [0.0] * 768)
    with pytest.raises(EmbeddingDimensionError, match="Expected 384 dimensions, got 768"):
        get_embedding("anything")


# --- Latency benchmark ---------------------------------------------------------------------------
def test_benchmark_batch_of_50_tickets():
    stats = benchmark_embeddings(BATCH_OF_50)
    print(f"\nEmbedding benchmark (50 tickets): {stats}")
    assert stats["count"] == 50
    assert stats["mean_ms"] > 0
    assert stats["p95_ms"] <= stats["max_ms"]
    assert stats["total_ms"] < 5000  # generous budget for the mock backend