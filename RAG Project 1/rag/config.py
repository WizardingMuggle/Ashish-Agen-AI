"""
================================================================================
 CONFIGURATION  —  every "knob" of the RAG pipeline lives in this one file
================================================================================

Why one central config?
    A RAG system is a chain of ~10 steps, and EVERY step has parameters that
    change the final answer (chunk size, overlap, top_k, BM25 k1/b, fusion
    method, temperature, ...). Keeping them in one place means:
      * you can read this file and understand everything that is tunable
      * you can change a value while the bot is running with:   /set <key> <value>
        and immediately watch the logs to see what changed.

Some keys affect the *index* (how the document is chunked / embedded / indexed).
Changing those forces the document to be re-indexed. They are listed in
`REINDEX_KEYS` below. Every other key only affects how a *question* is answered.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

# The project root is the folder that contains `main.py`.
# We ONLY allow ingesting files that live inside this folder (the "workspace").
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read OPENAI_API_KEY (and optional model overrides) from the .env file.
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class RAGConfig:
    # ------------------------------------------------------------------ #
    # 1. DOCUMENT
    # ------------------------------------------------------------------ #
    # Path (relative to the project root) of the file to ingest.
    document_path: str = "data/nimbus_handbook.md"

    # ------------------------------------------------------------------ #
    # 2. CHUNKING  (affects the index)
    # ------------------------------------------------------------------ #
    # "recursive": split on paragraphs -> lines -> sentences -> words,
    #              so chunks follow the natural structure of the text.
    # "fixed":     cut every N tokens blindly (simple, but can cut sentences).
    chunk_strategy: str = "recursive"
    # Maximum size of a chunk, measured in TOKENS (not characters, not words).
    chunk_size: int = 250
    # How many tokens of the previous chunk are repeated at the start of the
    # next one, so a fact that sits on a boundary is not lost.
    chunk_overlap: int = 40
    # Prefix each chunk with its markdown section path before embedding, e.g.
    # "Section: Leave Policy > Sick Leave". Gives the chunk extra context.
    prepend_section_title: bool = True

    # ------------------------------------------------------------------ #
    # 3. EMBEDDINGS  (affects the index)
    # ------------------------------------------------------------------ #
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    # How many texts we send to the embeddings API in one request.
    embedding_batch_size: int = 64
    # Cache embeddings on disk (.rag_cache/) so re-runs don't pay twice.
    use_embedding_cache: bool = True

    # ------------------------------------------------------------------ #
    # 4. BM25 KEYWORD SEARCH  (affects the index)
    # ------------------------------------------------------------------ #
    # k1: how fast repeated occurrences of a word stop adding score
    #     (0 = only presence matters, 3 = repetitions matter a lot).
    bm25_k1: float = 1.5
    # b:  how strongly long chunks are penalised (0 = no length penalty, 1 = full).
    bm25_b: float = 0.75
    # Drop very common words ("the", "is", "of") before scoring.
    bm25_remove_stopwords: bool = True
    # Reduce words to a crude root ("policies" -> "policy", "drones" -> "drone").
    bm25_use_stemming: bool = True

    # ------------------------------------------------------------------ #
    # 5. RETRIEVAL
    # ------------------------------------------------------------------ #
    # "vector": meaning-based search with embeddings
    # "bm25":   keyword-based search
    # "hybrid": run both and fuse the two ranked lists (recommended)
    retrieval_mode: str = "hybrid"
    # How to merge the two lists in hybrid mode:
    # "rrf":      Reciprocal Rank Fusion — uses only the RANK positions
    # "weighted": min-max normalise both scores, then alpha*vector + (1-alpha)*bm25
    fusion_method: str = "rrf"
    # Weight of the vector score in "weighted" fusion (1.0 = pure vector).
    hybrid_alpha: float = 0.5
    # The "k" constant in RRF: score = 1 / (rrf_k + rank). Bigger = flatter.
    rrf_k: int = 60
    # How many candidates each retriever returns before fusion.
    candidates_per_retriever: int = 10
    # How many chunks finally go into the LLM prompt.
    top_k: int = 4
    # Drop chunks whose cosine similarity to the question is below this.
    # (Not applied in pure "bm25" mode, where there is no embedding.)
    similarity_threshold: float = 0.20

    # ------------------------------------------------------------------ #
    # 6. RERANKING (optional second opinion on the candidates)
    # ------------------------------------------------------------------ #
    # Ask the LLM to score every fused candidate 0-10 for relevance and
    # re-order them. Slower + costs tokens, but often more precise.
    use_reranker: bool = False

    # ------------------------------------------------------------------ #
    # 7. QUERY REWRITING (conversational memory for retrieval)
    # ------------------------------------------------------------------ #
    # Turn a follow-up like "and how long does it fly?" into a standalone
    # question "How long can the NX-90 Sparrow drone fly?" before searching.
    use_query_rewriting: bool = True

    # ------------------------------------------------------------------ #
    # 8. GENERATION
    # ------------------------------------------------------------------ #
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    # 0 = deterministic and factual; 1+ = more creative / varied.
    temperature: float = 0.2
    # Upper bound on the length of the answer (in tokens).
    max_answer_tokens: int = 600
    # Token budget for retrieved context inside the prompt.
    max_context_tokens: int = 2500
    # How many previous Q/A turns are sent to the LLM.
    history_turns: int = 3

    # ------------------------------------------------------------------ #
    # 9. LOGGING
    # ------------------------------------------------------------------ #
    # 0 = answers only | 1 = stage headers | 2 = + tables & explanations
    # 3 = + everything (full prompts, vectors, per-term BM25 math)
    verbosity: int = 3
    # Also write a full (verbosity 3) log of the session to logs/.
    log_to_file: bool = True

    # ------------------------------------------------------------------ #
    # Helpers used by the /set and /config commands
    # ------------------------------------------------------------------ #
    def set_value(self, key: str, raw_value: str) -> object:
        """Parse `raw_value` to the right type for `key`, validate and assign it."""
        field_types = {f.name: f.type for f in fields(self)}
        if key not in field_types:
            raise KeyError(f"Unknown setting '{key}'. Type /config to see all settings.")

        current = getattr(self, key)
        if isinstance(current, bool):
            lowered = raw_value.strip().lower()
            if lowered in ("true", "1", "yes", "on"):
                value: object = True
            elif lowered in ("false", "0", "no", "off"):
                value = False
            else:
                raise ValueError(f"'{key}' expects true/false, got '{raw_value}'")
        elif isinstance(current, int):
            value = int(raw_value)
        elif isinstance(current, float):
            value = float(raw_value)
        else:
            value = raw_value.strip()

        if key in CHOICES and value not in CHOICES[key]:
            raise ValueError(f"'{key}' must be one of {CHOICES[key]}")
        if key == "chunk_overlap" and value >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if key == "chunk_size" and value <= self.chunk_overlap:
            raise ValueError("chunk_size must be larger than chunk_overlap")

        setattr(self, key, value)
        return value

    def as_dict(self) -> dict[str, object]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# Settings that change how the document is indexed -> re-index when changed.
REINDEX_KEYS = {
    "document_path",
    "chunk_strategy",
    "chunk_size",
    "chunk_overlap",
    "prepend_section_title",
    "embedding_model",
    "bm25_k1",
    "bm25_b",
    "bm25_remove_stopwords",
    "bm25_use_stemming",
}

# Allowed values for "enum-like" string settings.
CHOICES: dict[str, tuple[object, ...]] = {
    "chunk_strategy": ("recursive", "fixed"),
    "retrieval_mode": ("vector", "bm25", "hybrid"),
    "fusion_method": ("rrf", "weighted"),
    "verbosity": (0, 1, 2, 3),
}

# Approximate prices in USD per 1 MILLION tokens: (input, output).
# Used only to print cost estimates in the logs — check openai.com/pricing
# for the current numbers. Unknown models are shown as "n/a".
PRICES_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}
