"""
================================================================================
 QUERY STEPS Q3-Q8  —  THE RETRIEVER (puts the search steps together)
================================================================================

                   +--> Q3 embed question --> Q4 vector search --+
    question ------+                                             +--> Q6 fusion --> Q7 rerank --> Q8 filter
                   +--> Q5 BM25 keyword search ------------------+     (hybrid)     (optional)    (threshold
                                                                                                   + top_k)
`retrieval_mode` decides which branches run:
    "vector" -> only the top branch   (needs an embedding API call)
    "bm25"   -> only the bottom branch (free, no API call at all)
    "hybrid" -> both, then fusion
"""

from __future__ import annotations

from rag.config import RAGConfig
from rag.embeddings.embedder import embed_query
from rag.ingestion.chunker import Chunk
from rag.logger import Table, esc, log
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.hybrid import Candidate, fuse
from rag.retrieval.reranker import rerank
from rag.retrieval.vector_store import VectorStore


class Retriever:
    def __init__(self, chunks: list[Chunk], vector_store: VectorStore, bm25: BM25Index) -> None:
        self.chunks = chunks
        self.vector_store = vector_store
        self.bm25 = bm25

    def retrieve(self, query: str, cfg: RAGConfig) -> list[Candidate]:
        mode = cfg.retrieval_mode
        n = cfg.candidates_per_retriever
        use_vector = mode in ("vector", "hybrid")
        use_bm25 = mode in ("bm25", "hybrid")
        log.step(f"Retrieval mode: [bold]{mode}[/]  (vector={use_vector}, bm25={use_bm25}); "
                 f"{n} candidates per retriever")

        all_cosine = None
        vector_results: list[tuple[int, float]] = []
        if use_vector:
            log.stage("Q3", "EMBED THE QUESTION",
                      "Turn the question into a vector with the SAME model used for the chunks.")
            query_vector = embed_query(query, cfg)
            vector_results = self.vector_store.search(query_vector, n)
            all_cosine = self.vector_store.all_scores(query_vector)
        else:
            log.stage("Q3/Q4", "VECTOR SEARCH — SKIPPED", "retrieval_mode is 'bm25': no embedding, no API call.")

        bm25_results: list[tuple[int, float]] = []
        if use_bm25:
            bm25_results = self.bm25.search(query, n)
        else:
            log.stage("Q5", "BM25 SEARCH — SKIPPED", "retrieval_mode is 'vector'.")

        # ---- Q6: combine ------------------------------------------------
        if mode == "hybrid":
            candidates = fuse(vector_results, bm25_results, all_cosine, self.bm25.all_scores(query),
                              self.chunks, cfg)
        else:
            log.stage("Q6", "FUSION — SKIPPED", f"Only one retriever ran ({mode}); its ranking is used as-is.")
            results = vector_results if mode == "vector" else bm25_results
            candidates = []
            for rank, (cid, score) in enumerate(results, start=1):
                cand = Candidate(chunk_id=cid, fused_score=score)
                if mode == "vector":
                    cand.vector_score, cand.vector_rank = score, rank
                else:
                    cand.bm25_score, cand.bm25_rank = score, rank
                candidates.append(cand)

        # ---- Q7: optional reranking -------------------------------------
        if cfg.use_reranker:
            candidates = rerank(query, candidates, self.chunks, cfg)
        else:
            log.stage("Q7", "RERANKING — SKIPPED", "use_reranker is false. Try `/set use_reranker true`.")

        return self._filter(candidates, cfg)

    # ------------------------------------------------------------------ #
    def _filter(self, candidates: list[Candidate], cfg: RAGConfig) -> list[Candidate]:
        log.stage(
            "Q8", "FILTER & SELECT TOP-K",
            "Drop weak matches (similarity_threshold) and keep only the best top_k chunks for the prompt.",
        )
        kept, dropped = [], []
        for c in candidates:
            if c.vector_score is not None and c.vector_score < cfg.similarity_threshold:
                dropped.append(c)
            else:
                kept.append(c)

        if cfg.retrieval_mode == "bm25":
            log.detail("similarity_threshold not applied: there are no cosine scores in bm25 mode.")
        elif dropped:
            log.step(f"Dropped {len(dropped)} chunk(s) with cosine < {cfg.similarity_threshold}: "
                     f"{[(c.chunk_id, round(c.vector_score, 3)) for c in dropped]}")
            log.explain("These chunks matched a keyword but their meaning is far from the question.")
        else:
            log.detail(f"No chunk fell below similarity_threshold={cfg.similarity_threshold}")

        selected = kept[: cfg.top_k]
        if not selected:
            log.warn("No chunk passed the filters — the LLM will get NO context and should say it doesn't know.")
            return selected

        table = Table(title=f"Final context chunks (top_k={cfg.top_k})", title_justify="left")
        table.add_column("#", justify="right", style="bold")
        table.add_column("chunk", justify="right")
        table.add_column("cosine", justify="right", style="green")
        table.add_column("bm25", justify="right", style="yellow")
        table.add_column("rerank", justify="right", style="magenta")
        table.add_column("section", style="cyan", max_width=30, no_wrap=True)
        table.add_column("preview", style="grey70", max_width=55, no_wrap=True)
        fmt = lambda v, p: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
        for pos, c in enumerate(selected, start=1):
            ch = self.chunks[c.chunk_id]
            table.add_row(str(pos), str(c.chunk_id), fmt(c.vector_score, 4), fmt(c.bm25_score, 3),
                          fmt(c.rerank_score, 0), esc(ch.section), esc(ch.preview()))
        log.table(table, level=1)
        if len(kept) > cfg.top_k:
            log.detail(f"{len(kept) - cfg.top_k} more candidate(s) were cut by top_k={cfg.top_k}")
        return selected
