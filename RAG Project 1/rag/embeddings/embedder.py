"""
================================================================================
 INDEXING STEP I4 (and QUERY STEP Q3)  —  EMBEDDINGS
================================================================================

An EMBEDDING is a list of numbers (a vector) that represents the MEANING of a
text. `text-embedding-3-small` turns any text into 1,536 numbers:

    "How many vacation days do I get?"  ->  [ 0.012, -0.044, 0.031, ... ]  (1536 numbers)
    "Annual leave is 24 working days"   ->  [ 0.015, -0.039, 0.027, ... ]

Texts with similar meaning get vectors that point in a similar DIRECTION, even
if they share no words ("vacation" vs "annual leave"). That is the superpower
of vector search compared to keyword search.

The SAME model must embed both the chunks (at index time) and the question
(at query time) — vectors from different models live in different "spaces"
and cannot be compared.
"""

from __future__ import annotations

import time

import numpy as np

from rag.config import RAGConfig
from rag.embeddings.cache import EmbeddingCache
from rag.embeddings.openai_client import get_client
from rag.logger import log
from rag.utils.usage import format_cost, usage


def _call_embeddings_api(texts: list[str], model: str, purpose: str) -> list[list[float]]:
    client = get_client()
    log.debug(f"POST /v1/embeddings  model={model}  inputs={len(texts)}")
    start = time.perf_counter()
    response = client.embeddings.create(model=model, input=texts)
    seconds = time.perf_counter() - start
    call = usage.record(purpose, model, response.usage.prompt_tokens, 0, seconds)
    log.detail(f"API call: {len(texts)} texts, {call.input_tokens:,} tokens, "
               f"{seconds * 1000:.0f} ms, cost {format_cost(call.cost)}")
    # The API returns items with an .index; sort to be safe and keep input order.
    return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]


def embed_texts(texts: list[str], cfg: RAGConfig, purpose: str) -> np.ndarray:
    """Embed many texts, using the disk cache and batching. Returns shape (n, dim)."""
    cache = EmbeddingCache(cfg.embedding_model) if cfg.use_embedding_cache else None
    vectors: list[list[float] | None] = [cache.get(t) if cache else None for t in texts]
    missing = [i for i, v in enumerate(vectors) if v is None]

    if cache:
        log.step(f"Embedding cache: [green]{len(texts) - len(missing)} HIT[/] / "
                 f"[yellow]{len(missing)} MISS[/]  (cache file holds {len(cache):,} vectors)")
        log.explain("A HIT means this exact text was embedded before — we reuse the vector for free.")

    if missing:
        batch_size = cfg.embedding_batch_size
        n_batches = (len(missing) + batch_size - 1) // batch_size
        if len(texts) > 1:
            log.step(f"Sending {len(missing)} texts to OpenAI in {n_batches} batch(es) of up to {batch_size}")
            log.explain("Batching = many texts in one HTTP request. Far faster than one request per chunk.")
        else:
            log.step("Sending the text to the OpenAI embeddings API")
        for b in range(n_batches):
            idxs = missing[b * batch_size:(b + 1) * batch_size]
            new_vectors = _call_embeddings_api([texts[i] for i in idxs], cfg.embedding_model, purpose)
            for i, vec in zip(idxs, new_vectors):
                vectors[i] = vec
                if cache:
                    cache.put(texts[i], vec)
        if cache:
            cache.save()

    return np.array(vectors, dtype=np.float32)


def embed_query(question: str, cfg: RAGConfig) -> np.ndarray:
    """Embed a single question. Returns shape (dim,)."""
    vector = embed_texts([question], cfg, purpose="embed query")[0]
    describe_vector("Question vector", vector)
    return vector


def describe_vector(label: str, vector: np.ndarray) -> None:
    preview = ", ".join(f"{x:+.4f}" for x in vector[:8])
    log.detail(f"{label}: {vector.shape[0]} dimensions, L2 norm = {np.linalg.norm(vector):.4f}")
    log.debug(f"first 8 values: [{preview}, ...]")
    log.explain("OpenAI embeddings come pre-normalised (length ~1.0), so cosine similarity = dot product.")
