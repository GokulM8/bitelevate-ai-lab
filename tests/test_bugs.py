"""Tests for AI-BUG-01: zero-vector division in vector search.

Finding: on the current main the bug does NOT reproduce. cosine_similarity
guards zero magnitudes (`or 1.0`). naive_cosine below shows the crash the
incident report describes.
"""
import math

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import get_db
from app.main import app
from app.models.domain import Base, KnowledgeDoc
from app.services.vector_search import cosine_similarity, mock_text_embedding


def naive_cosine(v1, v2):
    """Cosine similarity WITHOUT a zero guard: the bug as described in AI-BUG-01."""
    dot = sum(a * b for a, b in zip(v1, v2))
    return dot / (math.sqrt(sum(a * a for a in v1)) * math.sqrt(sum(b * b for b in v2)))


def test_reproduce_zero_division_without_guard():
    with pytest.raises(ZeroDivisionError):
        naive_cosine([0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0])


def test_reproduce_empty_or_punctuation_query():
    doc_vec = mock_text_embedding("billing question")

    # Empty / whitespace-only queries embed to the all-zero vector...
    for query in ["", "   "]:
        query_vec = mock_text_embedding(query)
        assert all(v == 0.0 for v in query_vec)
        # ...so an unguarded cosine divides by 0 (the AI-BUG-01 crash):
        with pytest.raises(ZeroDivisionError):
            naive_cosine(query_vec, doc_vec)
        # ...while the repo's cosine_similarity survives thanks to `or 1.0`:
        assert cosine_similarity(query_vec, doc_vec) == 0.0

    # Punctuation-only queries are NOT zero vectors in this mock embedding.
    for query in ["???", "!!! ... ???"]:
        query_vec = mock_text_embedding(query)
        assert any(v != 0.0 for v in query_vec)
        assert -1.0 <= cosine_similarity(query_vec, doc_vec) <= 1.0


def test_cosine_similarity_with_true_zero_vector_does_not_crash():
    assert cosine_similarity([0.0] * 4, [1.0, 2.0, 3.0, 4.0]) == 0.0
    assert cosine_similarity([0.0] * 4, [0.0] * 4) == 0.0


@pytest.fixture
def client():
    """TestClient wired to a throwaway in-memory DB containing one KB doc."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    seed = Session()
    seed.add(KnowledgeDoc(title="Billing FAQ", content="How to download invoices", category="BILLING"))
    seed.commit()
    seed.close()

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.mark.parametrize("query", ["???", "!!! ... ???", "", "   "])
def test_search_endpoint_survives_punctuation_and_empty_queries(client, query):
    response = client.get("/api/v1/knowledge-base/search", params={"query": query})
    assert response.status_code in (200, 422)  # a crash would surface as a 500
    if response.status_code == 200:
        assert all(-1.0 <= item["similarity_score"] <= 1.0 for item in response.json())