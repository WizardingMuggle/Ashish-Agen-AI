"""
================================================================================
 QUERY STEP Q9  —  PROMPT CONSTRUCTION ("augmentation", the A in RAG)
================================================================================

R  = Retrieval   (we found relevant chunks)
A  = Augmented   (we paste them into the prompt)      <-- this file
G  = Generation  (the LLM writes an answer from them)

The messages we send look like this:

    [system]     rules: answer ONLY from the context, cite chunk ids, say
                 "I don't know" when the context lacks the answer
    [user]       previous question         \
    [assistant]  previous answer            > chat history (last N turns)
    ...                                    /
    [user]       CONTEXT:
                 [C12] (section: Leave Policy > Annual Leave)
                 Every employee receives 24 working days ...
                 [C13] ...
                 QUESTION: How many vacation days do I get?

The system prompt is the main defence against HALLUCINATION: the model is told
to rely on the retrieved context rather than on its general knowledge.

A TOKEN BUDGET (max_context_tokens) caps how much context we paste in: chunks
are added in ranking order until the budget is full.
"""

from __future__ import annotations

from rag.config import RAGConfig
from rag.ingestion.chunker import Chunk
from rag.logger import Table, log
from rag.memory.chat_history import ChatHistory
from rag.retrieval.hybrid import Candidate
from rag.utils.tokens import count_tokens

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about the document "{source}".

Rules:
1. Answer ONLY using the information in the CONTEXT section of the user's message.
2. After every fact, cite the chunk it came from, like [C3]. Use only ids that appear in the context.
3. If the context does not contain the answer, say: "I couldn't find that in the document." Do not guess
   and do not use outside knowledge.
4. Be concise and precise. Quote exact numbers, names and codes from the context."""


def build_messages(question: str, selected: list[Candidate], chunks: list[Chunk],
                   history: ChatHistory, cfg: RAGConfig, source: str) -> tuple[list[dict], list[int]]:
    """Return (messages for the chat API, ids of the chunks that made it into the prompt)."""
    log.stage(
        "Q9", "BUILD THE PROMPT (augmentation)",
        "Paste the retrieved chunks + rules + chat history into the messages we send to the LLM.",
    )
    model = cfg.chat_model

    # 1) Context blocks, respecting the token budget.
    blocks, used_ids, used_tokens = [], [], 0
    for cand in selected:
        chunk = chunks[cand.chunk_id]
        block = f"[C{chunk.id}] (section: {chunk.section or 'n/a'})\n{chunk.text}"
        n = count_tokens(block, model)
        if used_tokens + n > cfg.max_context_tokens:
            log.warn(f"Chunk {chunk.id} ({n} tokens) does not fit in max_context_tokens="
                     f"{cfg.max_context_tokens} (already used {used_tokens}); skipped.")
            continue
        blocks.append(block)
        used_ids.append(chunk.id)
        used_tokens += n

    context = "\n\n".join(blocks) if blocks else "(no relevant context was found)"
    user_message = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    system_message = SYSTEM_PROMPT.format(source=source)
    history_messages = history.as_messages(cfg.history_turns)

    messages = [{"role": "system", "content": system_message}, *history_messages,
                {"role": "user", "content": user_message}]

    # 2) Explain where the tokens go.
    history_tokens = sum(count_tokens(m["content"], model) for m in history_messages)
    table = Table(title="Prompt token budget", title_justify="left")
    table.add_column("part")
    table.add_column("tokens", justify="right")
    table.add_column("notes", style="grey70")
    table.add_row("system prompt (rules)", str(count_tokens(system_message, model)), "fixed every request")
    table.add_row("chat history", str(history_tokens),
                  f"{len(history_messages) // 2} previous turn(s), history_turns={cfg.history_turns}")
    table.add_row("retrieved context", str(used_tokens),
                  f"chunks {used_ids}, budget max_context_tokens={cfg.max_context_tokens}")
    table.add_row("question", str(count_tokens(question, model)), "")
    total = sum(count_tokens(m["content"], model) for m in messages)
    table.add_row("[bold]TOTAL[/]", f"[bold]{total}[/]", "+ a few tokens of message formatting overhead")
    log.table(table)
    log.explain("The retrieved context is usually the biggest part of the prompt — that is what top_k and "
                "chunk_size control.")

    for m in messages:
        log.block(f"message role={m['role']}", m["content"], level=3,
                  style={"system": "red", "user": "blue", "assistant": "green"}[m["role"]])
    return messages, used_ids
