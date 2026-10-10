"""
================================================================================
 INDEXING STEP I5 (and QUERY STEP Q4)  —  VECTOR STORE + SEMANTIC SEARCH
================================================================================

A vector store keeps one vector per chunk and answers the question:
"which chunk vectors are closest to this question vector?"

Our store is deliberately simple: a NumPy matrix of shape (n_chunks, 1536).

        chunk 0 -> [ 0.01, -0.04, ...,  0.02]
        chunk 1 -> [-0.03,  0.05, ..., -0.01]
        ...
    question q  -> [ 0.02, -0.01, ...,  0.03]

    scores = matrix @ q      # ONE matrix multiplication = cosine with every chunk

This is "brute force" / exact search. It is perfect for thousands of chunks.
Production vector databases (FAISS, pgvector, Pinecone, Chroma, Qdrant...) use
approximate nearest-neighbour indexes (HNSW, IVF) to stay fast with millions of
vectors — but the idea is identical.
"""

from __future__ import annotations

import numpy as np

from rag.ingestion.chunker import Chunk
from rag.logger import Table, esc, log
from rag.retrieval.similarity import cosine_similarity, euclidean_distance, normalize_rows


class VectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        log.stage(
            "I5", "BUILD VECTOR INDEX",
            "Stack all chunk vectors into one matrix, normalised so that a dot product equals cosine similarity.",
        )
        self.chunks = chunks
        self.matrix = normalize_rows(vectors)
        log.kv("Vector index", {
            "matrix shape": f"{self.matrix.shape}  (chunks x dimensions)",
            "memory": f"{self.matrix.nbytes / 1024:.1f} KB (float32)",
            "search method": "exact brute-force cosine similarity (matrix @ query)",
        })
        if len(chunks) >= 2:
            # A quick look at how similar neighbouring chunks are to each other.
            neighbour = [float(self.matrix[i] @ self.matrix[i + 1]) for i in range(len(chunks) - 1)]
            log.detail(f"Average cosine between neighbouring chunks: {np.mean(neighbour):.3f} "
                       f"(min {np.min(neighbour):.3f}, max {np.max(neighbour):.3f})")
            log.explain("Neighbouring chunks share topic (and overlap text), so they are usually more "
                        "similar to each other than to random chunks.")
        log.success("Vector index ready")

    def all_scores(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity between the query and EVERY chunk, shape (n_chunks,)."""
        q = query_vector / max(np.linalg.norm(query_vector), 1e-12)
        return self.matrix @ q

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        log.stage(
            "Q4", "VECTOR SEARCH (semantic)",
            "Compare the question vector with every chunk vector using cosine similarity; keep the closest.",
        )
        scores = self.all_scores(query_vector)
        order = np.argsort(-scores)[:k]
        results = [(int(i), float(scores[i])) for i in order]

        log.kv("Score distribution over ALL chunks", {
            "max": f"{scores.max():.4f}",
            "mean": f"{scores.mean():.4f}",
            "min": f"{scores.min():.4f}",
            "gap top1 - mean": f"{scores.max() - scores.mean():.4f}",
        })
        log.explain("A big gap between the best chunk and the average means the question clearly matches "
                    "some part of the document. A tiny gap suggests the answer may not be in it.")

        log.bar_chart(f"Top {len(results)} chunks by cosine similarity",
                      [(f"chunk {i}", s) for i, s in results], max_value=max(scores.max(), 1e-9))

        table = Table(title="Vector search results", title_justify="left")
        table.add_column("rank", justify="right")
        table.add_column("chunk", justify="right")
        table.add_column("cosine", justify="right", style="green")
        table.add_column("section", style="cyan", max_width=30, no_wrap=True)
        table.add_column("preview", style="grey70", max_width=60, no_wrap=True)
        for rank, (i, s) in enumerate(results, start=1):
            c = self.chunks[i]
            table.add_row(str(rank), str(i), f"{s:.4f}", esc(c.section), esc(c.preview()))
        log.table(table)

        if results:
            # Worked example for the best chunk: show that the 3 measures agree.
            best = results[0][0]
            q, v = query_vector, self.matrix[best]
            log.debug(f"Worked example for chunk {best}: cosine={cosine_similarity(q, v):.4f}, "
                      f"dot={float(q @ v):.4f}, euclidean distance={euclidean_distance(q, v):.4f} "
                      f"(check: sqrt(2 - 2*cos) = {np.sqrt(max(2 - 2 * cosine_similarity(q, v), 0)):.4f})")
        return results
