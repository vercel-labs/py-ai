"""Temporal activities — all real I/O lives here.

Two activities:
  1. llm_stream_activity  — calls the real LLM, drains the stream, returns messages
  2. tool_call_activity   — calls the real tool function, returns the result

These run in the activity context (outside the workflow sandbox), so they can
do network I/O, access environment variables, etc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from temporalio import activity

import vercel_ai_sdk as ai
from vercel_ai_sdk.anthropic import AnthropicModel


# ── Serializable parameter / result types ────────────────────────


@dataclass
class LLMCallParams:
    """Input to the LLM activity. All fields are JSON-serializable."""

    messages: list[dict[str, Any]]  # [Message.model_dump(), ...]
    tool_schemas: list[dict[str, Any]]  # [{name, description, schema}, ...]


@dataclass
class LLMCallResult:
    """Output from the LLM activity."""

    messages: list[dict[str, Any]]  # final accumulated messages from the stream


@dataclass
class ToolCallParams:
    """Input to the tool call activity."""

    tool_name: str
    tool_args: dict[str, Any]


# ── Stash for original tool functions ────────────────────────────
# temporal_tool() puts originals here before replacing .fn with an
# activity-calling wrapper.  The activity looks them up from here.

_original_tool_fns: dict[str, Any] = {}


def register_original_fn(name: str, fn: Any) -> None:
    _original_tool_fns[name] = fn


def get_original_fn(name: str) -> Any:
    return _original_tool_fns[name]


# ── Activities ───────────────────────────────────────────────────


@activity.defn(name="llm_stream")
async def llm_stream_activity(params: LLMCallParams) -> LLMCallResult:
    """Call the LLM, drain the full stream, return the final messages.

    The real AnthropicModel is constructed here from environment variables.
    The stream is consumed fully — the workflow only sees the final result.
    """
    # Hardcoded for the example — in production, model config would be
    # passed through or resolved from a registry.
    llm = AnthropicModel(
        model="anthropic/claude-sonnet-4",
        base_url="https://ai-gateway.vercel.sh",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    # Reconstruct Message objects from serialized dicts
    messages = [ai.Message.model_validate(m) for m in params.messages]

    # Reconstruct Tool objects (schema-only, for the LLM to know what's available)
    tools = [
        ai.Tool(
            name=t["name"],
            description=t["description"],
            schema=t["schema"],
            fn=lambda: None,  # placeholder — not called, just for schema
        )
        for t in params.tool_schemas
    ]

    # Drain the stream, keep all emitted messages
    result_messages: list[ai.Message] = []
    async for msg in llm.stream(messages=messages, tools=tools or None):
        result_messages.append(msg)

    # Only return the final message (the one with is_done=True and all parts settled)
    # Earlier messages are intermediate streaming states of the same message.
    final = result_messages[-1] if result_messages else None
    if final is None:
        return LLMCallResult(messages=[])

    return LLMCallResult(messages=[final.model_dump()])


@activity.defn(name="tool_call")
async def tool_call_activity(params: ToolCallParams) -> Any:
    """Execute a tool function by name with the given arguments.

    Looks up the *original* function (not the temporal wrapper) from the
    stash populated by temporal_tool().
    """
    fn = get_original_fn(params.tool_name)
    return await fn(**params.tool_args)
