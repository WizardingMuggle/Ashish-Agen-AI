"""
================================================================================
 LLM  —  the one place that talks to the OpenAI API
================================================================================

In Project 1 the model only ever produced TEXT. An agent needs the model to
produce DECISIONS. The Chat Completions API supports this with "tools":

    request:   messages + a list of tool schemas (name, description, JSON Schema)
    response:  either   message.content     -> a normal text answer
               or       message.tool_calls  -> [{"name": "check_inventory",
                                                 "arguments": '{"query": "AX-12"}'}]

The model does NOT run anything. It only *asks* for a tool to be run. Our code
runs it and sends the result back as a {"role": "tool"} message. That
request -> run -> report-back cycle is the heart of every agent framework.

Every call made here is recorded in `usage`, so we can enforce the cost
budget and print the cost per step.
"""

from __future__ import annotations

import os
import time
from typing import Any

from openai import OpenAI

from agent.config import AgentConfig
from agent.logger import log
from agent.usage import format_cost, usage

_client: OpenAI | None = None


class MissingAPIKeyError(RuntimeError):
    pass


def get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key or key.startswith("sk-your"):
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and paste your key into it."
            )
        # max_retries: the SDK automatically retries rate-limit / network errors.
        _client = OpenAI(api_key=key, max_retries=3, timeout=60)
    return _client


def is_reasoning_model(model: str) -> bool:
    """gpt-5 / o-series 'reasoning' models don't accept a custom temperature."""
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def chat(
    messages: list[dict],
    cfg: AgentConfig,
    purpose: str,
    tools: list[dict] | None = None,
    tool_choice: str | None = None,
    response_format: dict | None = None,
) -> Any:
    """One chat completion. Returns `choices[0]` (has .message and .finish_reason)."""
    params: dict[str, Any] = {
        "model": cfg.chat_model,
        "messages": messages,
        "max_completion_tokens": cfg.max_output_tokens,
    }
    if not is_reasoning_model(cfg.chat_model):
        params["temperature"] = cfg.temperature
    if tools:
        params["tools"] = tools
        params["parallel_tool_calls"] = cfg.parallel_tool_calls
        if tool_choice:
            params["tool_choice"] = tool_choice
    if response_format:
        params["response_format"] = response_format

    start = time.perf_counter()
    response = get_client().chat.completions.create(**params)
    seconds = time.perf_counter() - start

    u = response.usage
    call = usage.record(purpose, cfg.chat_model, u.prompt_tokens, u.completion_tokens, seconds)
    choice = response.choices[0]
    log.detail(
        f"API call '{purpose}': {u.prompt_tokens:,} in + {u.completion_tokens:,} out tokens, "
        f"{seconds * 1000:.0f} ms, {format_cost(call.cost)}, finish_reason={choice.finish_reason}"
    )
    return choice


def assistant_message_to_dict(message: Any) -> dict:
    """Turn the SDK's message object back into a plain dict we can re-send.

    The assistant message that REQUESTED the tools must stay in the history,
    right before the tool results, or the API rejects the next call: every
    {"role": "tool"} message must answer a tool_call_id it has seen.
    """
    out: dict[str, Any] = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]
    return out
