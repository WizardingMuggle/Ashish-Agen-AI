"""
================================================================================
 TOKENS  —  how LLMs "see" text
================================================================================

LLMs and embedding models don't read characters or words — they read TOKENS.
A token is a frequent piece of text: " the", "ing", "Nim", "bus", "4471"...
Rough rule of thumb for English:  1 token ~ 4 characters ~ 0.75 words.

Why do we care?
  * Model limits (context window) are measured in tokens.
  * Price is per token.
  * Our chunk_size is in tokens, so chunks fit nicely into prompts.

We use `tiktoken`, OpenAI's own tokenizer library, so our counts match what
the API will bill us for.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=None)
def get_encoding(model: str) -> tiktoken.Encoding:
    """Return the tokenizer used by `model` (falls back to cl100k_base)."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Newer models may not be known to this tiktoken version yet.
        name = "o200k_base" if model.startswith(("gpt-4o", "gpt-4.1", "gpt-5", "o")) else "cl100k_base"
        return tiktoken.get_encoding(name)


def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    return len(get_encoding(model).encode(text))


def show_tokens(text: str, model: str = "text-embedding-3-small", limit: int = 25) -> str:
    """Visualise tokenisation, e.g.  'Error E-4471'  ->  |Error| E|-|447|1|"""
    enc = get_encoding(model)
    ids = enc.encode(text)[:limit]
    pieces = [enc.decode([i]) for i in ids]
    return "|" + "|".join(p.replace("\n", "\\n") for p in pieces) + "|"
