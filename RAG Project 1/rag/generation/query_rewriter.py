"""
================================================================================
 QUERY STEP Q2  —  QUERY REWRITING (make follow-up questions searchable)
================================================================================

Conversation:
    You: How long can the Sparrow drone fly?
    Bot: About 38 minutes per battery.
    You: And what's its wind limit?          <-- what is "its"?

If we search the index for "And what's its wind limit?", neither the embedding
nor BM25 knows we mean the Sparrow drone. So before searching, we ask the LLM
to rewrite the follow-up into a STANDALONE question using the chat history:

    "What is the maximum wind speed the NX-90 Sparrow drone can handle?"

This is called "query condensation" or "conversational query rewriting".
It only runs when there IS history — the first question is used as-is.
Try `/set use_query_rewriting false` and ask a follow-up to see retrieval fail.
"""

from __future__ import annotations

import time

from rag.config import RAGConfig
from rag.embeddings.openai_client import get_client, is_reasoning_model
from rag.logger import esc, log
from rag.memory.chat_history import ChatHistory
from rag.utils.usage import format_cost, usage

REWRITE_PROMPT = """Given the conversation and a follow-up question, rewrite the follow-up
into a single standalone question that can be understood without the conversation.
Replace pronouns and vague references with the specific names they refer to.
If the question is already standalone, return it unchanged.
Return ONLY the rewritten question, nothing else.

Conversation:
{history}

Follow-up question: {question}
Standalone question:"""


def rewrite_query(question: str, history: ChatHistory, cfg: RAGConfig) -> str:
    log.stage(
        "Q2", "QUERY REWRITING",
        "Turn a follow-up question into a standalone one, so retrieval knows what 'it' or 'that' means.",
    )
    if not cfg.use_query_rewriting:
        log.step("Skipped: use_query_rewriting is false")
        return question
    if len(history) == 0:
        log.step("Skipped: no chat history yet, the question is already standalone")
        return question

    turns = history.recent(cfg.history_turns)
    history_text = "\n".join(f"User: {t.question}\nAssistant: {t.answer[:500]}" for t in turns)
    prompt = REWRITE_PROMPT.format(history=history_text, question=question)
    log.block("Rewrite prompt", prompt, level=3, style="magenta")

    params: dict = {}
    if not is_reasoning_model(cfg.chat_model):
        params["temperature"] = 0
    start = time.perf_counter()
    response = get_client().chat.completions.create(
        model=cfg.chat_model,
        messages=[{"role": "user", "content": prompt}],
        **params,
    )
    seconds = time.perf_counter() - start
    call = usage.record("rewrite query", cfg.chat_model, response.usage.prompt_tokens,
                        response.usage.completion_tokens, seconds)
    rewritten = (response.choices[0].message.content or question).strip()

    log.step(f"Original : [yellow]{esc(question)}[/]")
    log.step(f"Rewritten: [green]{esc(rewritten)}[/]")
    log.detail(f"Rewrite call: {call.input_tokens:,} in / {call.output_tokens:,} out tokens, "
               f"{seconds * 1000:.0f} ms, cost {format_cost(call.cost)}")
    log.explain("Only RETRIEVAL uses the rewritten question. The final answer still sees your original "
                "words plus the chat history.")
    return rewritten
