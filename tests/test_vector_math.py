"""Unit tests for cosine similarity in app/services/vector_search.py.

Cosine similarity measures the angle between two vectors:
    cos(theta) = (a . b) / (|a| * |b|)
     1.0 -> same direction (perfect match)
     0.0 -> perpendicular (unrelated)
    -1.0 -> opposite direction
"""
import pytest

from app.services.vector_search import cosine_similarity, mock_text_embedding


# 1. Identical vectors -> perfect match
def test_vector_with_itself_is_perfect_match():
    vec = [0.5, 1.2, -3.0, 4.4]
    assert cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_sentence_embedding_with_itself_is_perfect_match():
    emb = mock_text_embedding("Password reset")
    assert cosine_similarity(emb, emb) == pytest.approx(1.0)


# 2. Opposite / unrelated vectors -> -1.0 / 0.0
def test_opposite_vectors_score_minus_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]) == pytest.approx(-1.0)


def test_orthogonal_vectors_score_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


# 3. Similar meaning -> high score (> 0.80)
def test_similar_vectors_score_high():
    # Two vectors pointing in nearly the same direction
    assert cosine_similarity([1.0, 2.0, 3.0], [1.1, 2.1, 2.9]) > 0.80


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mock_text_embedding is character-based, not semantic: "
        "'billing question' vs 'invoice issue' scores 0.7113, below 0.80. "
        "A real embedding model is needed for meaning-based similarity."
    ),
)
def test_similar_sentences_score_high_with_mock_embedding():
    score = cosine_similarity(
        mock_text_embedding("billing question"),
        mock_text_embedding("invoice issue"),
    )
    assert score > 0.80