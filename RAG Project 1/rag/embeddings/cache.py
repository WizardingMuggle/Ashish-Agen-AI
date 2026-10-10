"""
================================================================================
 EMBEDDING CACHE  —  never pay twice for the same vector
================================================================================

An embedding depends ONLY on (model, exact input text). So we store them on
disk keyed by  sha256(model + text):

    .rag_cache/embeddings_text-embedding-3-small.json
    { "3f2a...": [0.0123, -0.0456, ...], ... }

Watch the logs: the first run shows "cache MISS" for every chunk (API call),
the second run shows "cache HIT" (no API call, no cost). Change chunk_size and
most chunks become new text -> misses again. That is a real-world lesson:
re-chunking a large corpus means re-embedding (and re-paying for) it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rag.config import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / ".rag_cache"


class EmbeddingCache:
    def __init__(self, model: str) -> None:
        self.path: Path = CACHE_DIR / f"embeddings_{model}.json"
        self._data: dict[str, list[float]] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        self.model = model

    def key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}::{text}".encode("utf-8")).hexdigest()

    def get(self, text: str) -> list[float] | None:
        return self._data.get(self.key(text))

    def put(self, text: str, vector: list[float]) -> None:
        self._data[self.key(text)] = vector

    def save(self) -> None:
        CACHE_DIR.mkdir(exist_ok=True)
        self.path.write_text(json.dumps(self._data), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._data)
