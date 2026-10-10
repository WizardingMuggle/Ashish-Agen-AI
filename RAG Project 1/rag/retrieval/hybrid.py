"""
================================================================================
 QUERY STEP Q6  —  HYBRID FUSION (combine vector + BM25 results)
================================================================================

Vector search and BM25 fail in DIFFERENT situations:

    question                         vector search        BM25
    -------------------------------  -------------------  -------------------
    "How many vacation days?"        finds "annual leave"  misses (no shared word)
    "What does E-4471 mean?"         often fuzzy           exact hit on "e-4471"

So we run both and merge the two ranked lists. Two classic methods:

1) RECIPROCAL RANK FUSION (RRF)  — fusion_method = "rrf"
       score(chunk) = SUM over lists of  1 / (rrf_k + rank_in_that_list)
   Only ranks are used, never raw scores, so it does not matter that cosine
   is ~0.5 while BM25 is ~7.3. A chunk ranked well by BOTH lists wins.
   With rrf_k=60: rank 1 -> 1/61 = 0.0164, rank 2 -> 1/62 = 0.0161 ...

2) WEIGHTED SCORE FUSION  — fusion_method = "weighted"
       norm(x) = (x - min) / (max - min)                (min-max to 0..1)
       score   = alpha * norm(cosine) + (1 - alpha) * norm(bm25)
   hybrid_alpha = 1.0 -> pure vector,  0.0 -> pure BM25.
   Uses the actual score gaps, but is sensitive to outliers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rag.config import RAGConfig
from rag.ingestion.chunker import Chunk
from rag.logger import Table, esc, log


@dataclass
class Candidate:
    chunk_id: int
    vector_score: float | None = None   # cosine similarity with the question
    vector_rank: int | None = None      # 1-based rank in the vector list (None = not in it)
    bm25_score: float | None = None
    bm25_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None   # filled by the optional LLM reranker
    rerank_reason: str = ""


def _min_max(values: np.ndarray) -> np.ndarray:
    span = values.max() - values.min()
    return np.zeros_like(values) if span == 0 else (values - values.min()) / span


def fuse(vector_results: list[tuple[int, float]], bm25_results: list[tuple[int, float]],
         all_cosine: np.ndarray, all_bm25: list[float], chunks: list[Chunk],
         cfg: RAGConfig) -> list[Candidate]:
    log.stage(
        "Q6", f"HYBRID FUSION ({cfg.fusion_method.upper()})",
        "Merge the semantic list and the keyword list into one ranking.",
    )
    vec_rank = {cid: r for r, (cid, _) in enumerate(vector_results, start=1)}
    bm_rank = {cid: r for r, (cid, _) in enumerate(bm25_results, start=1)}
    union = list(dict.fromkeys([c for c, _ in vector_results] + [c for c, _ in bm25_results]))

    in_both = [c for c in union if c in vec_rank and c in bm_rank]
    log.step(f"Vector list: {len(vec_rank)} chunks | BM25 list: {len(bm_rank)} chunks | "
             f"union: {len(union)} | found by BOTH: {len(in_both)} {esc(str(in_both))}")
    log.explain("Chunks found by both retrievers are usually the strongest candidates.")

    candidates = [
        Candidate(chunk_id=c, vector_score=float(all_cosine[c]), vector_rank=vec_rank.get(c),
                  bm25_score=float(all_bm25[c]), bm25_rank=bm_rank.get(c))
        for c in union
    ]

    if cfg.fusion_method == "rrf":
        k = cfg.rrf_k
        for cand in candidates:
            cand.fused_score = sum(1 / (k + r) for r in (cand.vector_rank, cand.bm25_rank) if r is not None)
        log.detail(f"RRF formula: 1/({k} + vector_rank) + 1/({k} + bm25_rank); a missing rank adds 0")
    else:
        alpha = cfg.hybrid_alpha
        norm_vec = _min_max(np.asarray(all_cosine, dtype=float))
        norm_bm = _min_max(np.asarray(all_bm25, dtype=float))
        for cand in candidates:
            cand.fused_score = alpha * norm_vec[cand.chunk_id] + (1 - alpha) * norm_bm[cand.chunk_id]
        log.detail(f"Weighted formula: {alpha} * minmax(cosine) + {1 - alpha:.2f} * minmax(bm25) "
                   f"(min/max taken over all {len(chunks)} chunks)")

    candidates.sort(key=lambda c: -c.fused_score)
    _log_fusion_table(candidates, chunks, cfg)
    return candidates


def _rank(r: int | None) -> str:
    return "[grey50]-[/]" if r is None else f"#{r}"


def _log_fusion_table(candidates: list[Candidate], chunks: list[Chunk], cfg: RAGConfig) -> None:
    table = Table(title="Fusion table (sorted by fused score)", title_justify="left")
    table.add_column("final", justify="right", style="bold")
    table.add_column("chunk", justify="right")
    table.add_column("vec rank", justify="right")
    table.add_column("cosine", justify="right", style="green")
    table.add_column("bm25 rank", justify="right")
    table.add_column("bm25", justify="right", style="yellow")
    table.add_column("FUSED", justify="right", style="bold cyan")
    table.add_column("section", style="cyan", max_width=28, no_wrap=True)
    for pos, c in enumerate(candidates, start=1):
        table.add_row(f"#{pos}", str(c.chunk_id), _rank(c.vector_rank), f"{c.vector_score:.4f}",
                      _rank(c.bm25_rank), f"{c.bm25_score:.3f}", f"{c.fused_score:.5f}",
                      esc(chunks[c.chunk_id].section))
    log.table(table)

    top = candidates[0] if candidates else None
    if top and cfg.fusion_method == "rrf":
        parts = [f"1/({cfg.rrf_k}+{r})" for r in (top.vector_rank, top.bm25_rank) if r is not None]
        log.debug(f"Worked example, chunk {top.chunk_id}: {' + '.join(parts)} = {top.fused_score:.5f}")
