"""
================================================================================
 QUERY STEP Q7  —  LLM RERANKING (optional, `/set use_reranker true`)
================================================================================

Retrievers are fast but shallow:
  * an embedding squeezes a whole chunk into ONE vector, and
  * BM25 only counts words.
Neither actually "reads" the chunk together with the question.

A RERANKER does: it looks at (question, chunk) pairs and judges relevance.
Here we use the chat model as the judge — it scores every candidate 0-10 —
then we re-sort. Production systems often use a dedicated cross-encoder
model for this (e.g. Cohere Rerank, bge-reranker), which is cheaper.

Trade-off: better precision, but one extra LLM call (latency + cost) per question.
Watch the "movement" column in the logs to see what the reranker changed.
"""

from __future__ import annotations

import json
import time

from rag.config import RAGConfig
from rag.embeddings.openai_client import get_client, is_reasoning_model
from rag.ingestion.chunker import Chunk
from rag.logger import Table, esc, log
from rag.retrieval.hybrid import Candidate
from rag.utils.usage import format_cost, usage

RERANK_PROMPT = """You are a relevance judge for a search engine.
Score how useful each passage is for answering the question, from 0 (irrelevant)
to 10 (directly contains the answer). Judge only relevance, not writing quality.

Question: {question}

Passages:
{passages}

Reply with JSON only, in this exact shape:
{{"scores": [{{"id": <passage id>, "score": <0-10>, "reason": "<max 12 words>"}}, ...]}}"""


def rerank(question: str, candidates: list[Candidate], chunks: list[Chunk],
           cfg: RAGConfig) -> list[Candidate]:
    log.stage(
        "Q7", "LLM RERANKING",
        "Ask the chat model to read each candidate next to the question and score its relevance 0-10.",
    )
    if not candidates:
        return candidates

    passages = "\n\n".join(f"[id={c.chunk_id}] {chunks[c.chunk_id].text[:1200]}" for c in candidates)
    prompt = RERANK_PROMPT.format(question=question, passages=passages)
    log.block("Reranking prompt", prompt, level=3, style="magenta")

    params: dict = {}
    if not is_reasoning_model(cfg.chat_model):
        params["temperature"] = 0
    start = time.perf_counter()
    response = get_client().chat.completions.create(
        model=cfg.chat_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        **params,
    )
    seconds = time.perf_counter() - start
    call = usage.record("rerank", cfg.chat_model, response.usage.prompt_tokens,
                        response.usage.completion_tokens, seconds)
    log.detail(f"Rerank call: {call.input_tokens:,} in / {call.output_tokens:,} out tokens, "
               f"{seconds * 1000:.0f} ms, cost {format_cost(call.cost)}")

    raw = response.choices[0].message.content or "{}"
    log.block("Raw reranker reply", raw, level=3, style="magenta")
    try:
        judged = {int(s["id"]): s for s in json.loads(raw).get("scores", [])}
    except (ValueError, KeyError, TypeError) as exc:
        log.warn(f"Could not parse reranker JSON ({exc}); keeping the fusion order.")
        return candidates

    old_position = {c.chunk_id: pos for pos, c in enumerate(candidates, start=1)}
    for c in candidates:
        entry = judged.get(c.chunk_id, {})
        c.rerank_score = float(entry.get("score", 0))
        c.rerank_reason = str(entry.get("reason", ""))
    # Sort by LLM score; ties keep the fusion order (Python's sort is stable).
    reranked = sorted(candidates, key=lambda c: -(c.rerank_score or 0))

    table = Table(title="Reranking result", title_justify="left")
    table.add_column("new", justify="right", style="bold")
    table.add_column("old", justify="right", style="grey62")
    table.add_column("movement", justify="center")
    table.add_column("chunk", justify="right")
    table.add_column("LLM score", justify="right", style="green")
    table.add_column("judge's reason", style="grey70", max_width=60)
    for new_pos, c in enumerate(reranked, start=1):
        delta = old_position[c.chunk_id] - new_pos
        move = "[green]^ +%d[/]" % delta if delta > 0 else ("[red]v %d[/]" % delta if delta < 0 else "=")
        table.add_row(f"#{new_pos}", f"#{old_position[c.chunk_id]}", move, str(c.chunk_id),
                      f"{c.rerank_score:.0f}/10", esc(c.rerank_reason))
    log.table(table)
    log.explain("The retrievers decide WHICH chunks are candidates; the reranker decides their ORDER.")
    return reranked
