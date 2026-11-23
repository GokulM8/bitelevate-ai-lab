"""Embedding generator for SupportPulse with fixed 384-dimension output.

Pipeline (same shape as a real sentence-transformer):
    text -> tokenizer (input_ids + attention_mask)
         -> one vector per token
         -> mean pooling over real tokens (mask = 1)
         -> L2 normalization
         -> 384-dim list[float]

Backends (env var EMBEDDING_BACKEND):
- "mock" (default): zero-cost, deterministic, no downloads
- "sentence-transformers": local all-MiniLM-L6-v2 (also 384 dims);
  requires `pip install sentence-transformers`
"""
import math
import os
import random
import statistics
import time
from functools import lru_cache
from typing import Dict, List, Sequence

from app.services.tokenizer import CLS_ID, SEP_ID, tokenize

EMBEDDING_DIM = 384
ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingDimensionError(ValueError):
    """Raised when an embedding does not have exactly EMBEDDING_DIM floats."""


def _backend() -> str:
    return os.getenv("EMBEDDING_BACKEND", "mock").lower()


# --- Mock backend ------------------------------------------------------------------
@lru_cache(maxsize=50_000)
def _token_vector(token_id: int) -> tuple:
    """Deterministic pseudo-random vector for one token ID (same ID -> same vector)."""
    rng = random.Random(token_id)
    return tuple(rng.gauss(0.0, 1.0) for _ in range(EMBEDDING_DIM))


def _mock_embedding(text: str) -> List[float]:
    encoded = tokenize(text)
    content_ids = [
        tid for tid, mask in zip(encoded.input_ids, encoded.attention_mask)
        if mask == 1 and tid not in (CLS_ID, SEP_ID)
    ]
    if not content_ids:
        return [0.0] * EMBEDDING_DIM

    # Mean pooling: average the token vectors
    pooled = [0.0] * EMBEDDING_DIM
    for tid in content_ids:
        for i, value in enumerate(_token_vector(tid)):
            pooled[i] += value
    pooled = [v / len(content_ids) for v in pooled]

    # L2 normalization -> unit length, ready for cosine similarity
    norm = math.sqrt(sum(v * v for v in pooled)) or 1.0
    return [v / norm for v in pooled]


# --- Sentence-transformers backend ------------------------------------------------------
@lru_cache(maxsize=1)
def _load_st_model():
    from sentence_transformers import SentenceTransformer  # optional dependency
    return SentenceTransformer(ST_MODEL_NAME)


def _st_embedding(text: str) -> List[float]:
    vector = _load_st_model().encode(text or "", normalize_embeddings=True)
    return [float(v) for v in vector]


# --- Public API ---------------------------------------------------------------------------
def validate_embedding(vector: Sequence[float]) -> List[float]:
    """Ensure the vector is exactly EMBEDDING_DIM finite floats."""
    if len(vector) != EMBEDDING_DIM:
        raise EmbeddingDimensionError(
            f"Expected {EMBEDDING_DIM} dimensions, got {len(vector)}"
        )
    result = [float(v) for v in vector]
    if not all(math.isfinite(v) for v in result):
        raise EmbeddingDimensionError("Embedding contains NaN or infinite values")
    return result


def get_embedding(text: str) -> List[float]:
    """Return a 384-dimension embedding for the given text."""
    if _backend() == "sentence-transformers":
        vector = _st_embedding(text)
    else:
        vector = _mock_embedding(text)
    return validate_embedding(vector)


def get_embeddings(texts: Sequence[str]) -> List[List[float]]:
    """Embed a batch of texts; output shape is (len(texts), 384)."""
    return [get_embedding(t) for t in texts]


def benchmark_embeddings(texts: Sequence[str]) -> Dict[str, float]:
    """Time embedding generation for a batch and return latency stats in ms."""
    if not texts:
        raise ValueError("benchmark_embeddings needs at least one text")
    per_text_ms: List[float] = []
    start = time.perf_counter()
    for text in texts:
        t0 = time.perf_counter()
        get_embedding(text)
        per_text_ms.append((time.perf_counter() - t0) * 1000)
    total_ms = (time.perf_counter() - start) * 1000

    ordered = sorted(per_text_ms)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(texts),
        "total_ms": round(total_ms, 3),
        "mean_ms": round(statistics.mean(per_text_ms), 3),
        "p95_ms": round(ordered[p95_index], 3),
        "max_ms": round(ordered[-1], 3),
        "throughput_per_sec": round(len(texts) / (total_ms / 1000), 1) if total_ms else 0.0,
    }