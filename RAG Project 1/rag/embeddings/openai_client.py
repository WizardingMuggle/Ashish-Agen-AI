"""
================================================================================
 OPENAI CLIENT  —  one shared connection to the OpenAI API
================================================================================

All three kinds of API calls in this project go through this client:
    client.embeddings.create(...)        -> turn text into vectors
    client.chat.completions.create(...)  -> rewrite queries, rerank, answer

The API key is read from the OPENAI_API_KEY environment variable, which
`rag/config.py` loads from the `.env` file in the project root.
"""

from __future__ import annotations

import os

from openai import OpenAI

_client: OpenAI | None = None


class MissingAPIKeyError(RuntimeError):
    pass


def get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key or key.startswith("sk-your"):
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and paste your key into it."
            )
        # max_retries: the SDK automatically retries rate-limit / network errors.
        _client = OpenAI(api_key=key, max_retries=3, timeout=60)
    return _client


def is_reasoning_model(model: str) -> bool:
    """gpt-5 / o-series 'reasoning' models don't accept a custom temperature."""
    return model.startswith(("gpt-5", "o1", "o3", "o4"))
