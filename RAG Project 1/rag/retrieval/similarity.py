"""
================================================================================
 SIMILARITY MEASURES  —  "how close are two vectors?"
================================================================================

Three common ways to compare two embedding vectors a and b:

  1. DOT PRODUCT        a . b = sum(a_i * b_i)
       Big when the vectors point the same way AND are long.

  2. COSINE SIMILARITY  cos(a, b) = (a . b) / (|a| * |b|)
       Only the ANGLE matters, length is ignored. Range -1 .. +1
         +1  = same direction  (same meaning)
          0  = unrelated (90 degrees)
       This is the standard for text embeddings.

  3. EUCLIDEAN DISTANCE |a - b| = sqrt(sum((a_i - b_i)^2))
       Straight-line distance: SMALLER = more similar.

When vectors are normalised to length 1 (OpenAI's are), all three give the
SAME ranking:  cos = dot,  and  |a-b|^2 = 2 - 2*cos.

Practical note about scores from text-embedding-3-*: cosine values for
"relevant" chunks are typically ~0.35-0.70, and unrelated text still scores
~0.05-0.25. Scores are RELATIVE — what matters is the ranking and the gap
between the top results and the rest, not a magic absolute number.
"""

from __future__ import annotations

import numpy as np


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to length 1, so a matrix multiplication gives cosines."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)
