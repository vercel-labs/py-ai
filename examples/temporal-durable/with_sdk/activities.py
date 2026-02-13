"""Temporal activities — all real I/O lives here.

Activities run outside the workflow sandbox, so they can do network I/O,
access environment variables, etc.  No framework imports needed — these
are plain functions wrapped with @activity.defn.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from temporalio import activity

from vercel_ai_sdk.anthropic import AnthropicModel
import vercel_ai_sdk as ai


# ── Tool functions (plain Python) ─────────────────────────────────


def get_weather(city: str) -> str:
    return f"Sunny, 72F in {city}"


def get_population(city: str) -> int:
    return {"new york": 8_336_817, "los angeles": 3_979_576}.get(
        city.lower(), 1_000_000
    )


TOOL_FNS: dict[str, Any] = {
    "get_weather": get_weather,
    "get_population": get_population,
}


# ── Serializable parameter types ─────────────────────────────────


@dataclass
class LLMCallParams:
    messages: list[dict[str, Any]]
    tool_schemas: list[dict[str, Any]]


@dataclass
class LLMCallResult:
    messages: list[dict[str, Any]]  # list of serialized ai.Message


@dataclass
class ToolCallParams:
    tool_name: str
    tool_args: dict[str, Any]


# ── Activities ───────────────────────────────────────────────────


@activity.defn(name="llm_call")
async def llm_call_activity(params: LLMCallParams) -> LLMCallResult:
    """Call the LLM, drain the stream, return the final message."""
    llm = AnthropicModel(
        model="anthropic/claude-sonnet-4",
        base_url="https://ai-gateway.vercel.sh",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    messages = [ai.Message.model_validate(m) for m in params.messages]

    # Build Tool objects with schema only (fn is not called activity-side).
    # TODO: framework should expose ToolSchema or let stream() accept dicts.
    async def _noop() -> None: ...

    tools = [
        ai.Tool(
            name=t["name"], description=t["description"], schema=t["schema"], fn=_noop
        )
        for t in params.tool_schemas
    ]

    final = None
    async for msg in llm.stream(messages=messages, tools=tools or None):
        final = msg

    if final is None:
        return LLMCallResult(messages=[])
    return LLMCallResult(messages=[final.model_dump()])


@activity.defn(name="tool_call")
async def tool_call_activity(params: ToolCallParams) -> Any:
    """Execute a tool function by name."""
    fn = TOOL_FNS[params.tool_name]
    return fn(**params.tool_args)
