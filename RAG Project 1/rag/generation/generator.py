"""
================================================================================
 QUERY STEP Q10  —  GENERATION (the G in RAG)
================================================================================

We send the messages to the chat model and STREAM the answer: tokens are
printed as soon as they are generated instead of waiting for the full reply.
That is why ChatGPT-style UIs "type" their answers.

Parameters that change the output:
  * temperature        0.0 -> nearly deterministic, best for factual Q&A
                       1.0+ -> more varied wording (and more risk of drifting)
  * max_answer_tokens  hard cap on answer length (the answer is cut off if reached)

`stream_options={"include_usage": True}` asks the API to send token counts in
the final streamed message so we can log the exact cost.
"""

from __future__ import annotations

import re
import time

from rich.text import Text

from rag.config import RAGConfig
from rag.embeddings.openai_client import get_client, is_reasoning_model
from rag.logger import log
from rag.utils.usage import format_cost, usage


def generate_answer(messages: list[dict], cfg: RAGConfig) -> str:
    log.stage(
        "Q10", "GENERATE THE ANSWER (streaming)",
        "The LLM reads the prompt and writes an answer grounded in the retrieved chunks.",
    )
    params: dict = {"max_completion_tokens": cfg.max_answer_tokens}
    if is_reasoning_model(cfg.chat_model):
        log.detail(f"{cfg.chat_model} is a reasoning model: temperature is not supported, using the default.")
    else:
        params["temperature"] = cfg.temperature
    log.kv("Generation settings", {"model": cfg.chat_model, **params}, level=2)

    start = time.perf_counter()
    first_token_at: float | None = None
    parts: list[str] = []
    prompt_tokens = completion_tokens = 0
    finish_reason = None

    stream = get_client().chat.completions.create(
        model=cfg.chat_model, messages=messages, stream=True,
        stream_options={"include_usage": True}, **params,
    )
    log.always("")
    log.always("[bold green]Assistant:[/]")
    for event in stream:
        if event.usage:  # the very last event carries the token counts
            prompt_tokens = event.usage.prompt_tokens
            completion_tokens = event.usage.completion_tokens
        if not event.choices:
            continue
        choice = event.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        delta = choice.delta.content or ""
        if delta:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            parts.append(delta)
            # Print raw text (markup=False) so "[C3]" citations are not read as rich styles.
            log.console.print(delta, end="", markup=False, highlight=False)
    log.console.print()
    answer = "".join(parts)
    log.file_only(Text(answer))

    seconds = time.perf_counter() - start
    call = usage.record("answer", cfg.chat_model, prompt_tokens, completion_tokens, seconds)
    log.kv("Generation stats", {
        "time to first token": f"{((first_token_at or time.perf_counter()) - start) * 1000:.0f} ms",
        "total time": f"{seconds * 1000:.0f} ms",
        "prompt tokens": f"{prompt_tokens:,}",
        "answer tokens": f"{completion_tokens:,}",
        "finish reason": finish_reason,
        "cost": format_cost(call.cost),
    }, level=1)
    if finish_reason == "length":
        log.warn("The answer was cut off because it reached max_answer_tokens.")
    return answer


def cited_chunk_ids(answer: str) -> list[int]:
    """Find citations like [C3] or [C3, C7] in the answer."""
    return sorted({int(n) for n in re.findall(r"C(\d+)", " ".join(re.findall(r"\[([^\]]+)\]", answer)))})
