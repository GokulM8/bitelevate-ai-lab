import math
from typing import List, Tuple

def mock_text_embedding(text: str) -> List[float]:
    """Generate deterministic pseudo-embedding vector for text similarity matching."""
    words = text.lower().split()
    vector = [0.0] * 16
    for idx, word in enumerate(words):
        for char_idx, char in enumerate(word):
            vector[(ord(char) + idx) % 16] += 1.0
    
    # Normalize vector
    magnitude = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / magnitude for v in vector]

import math

def cosine_similarity(vec1, vec2) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    # Guard: a zero-magnitude vector has no direction, so similarity is undefined.
    # Return 0.0 instead of dividing by zero.
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)