"""Temporal activities — all real I/O lives here.

Activities run outside the workflow sandbox, so they can do network I/O,
access environment variables, etc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from temporalio import activity

import vercel_ai_sdk as ai
from vercel_ai_sdk.anthropic import AnthropicModel


# ── Tool functions ───────────────────────────────────────────────
# Same as examples/samples/custom_loop.py — plain @ai.tool functions.


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72F in {city}"


@ai.tool
async def get_population(city: str) -> int:
    """Get population of a city."""
    return {"new york": 8_336_817, "los angeles": 3_979_576}.get(
        city.lower(), 1_000_000
    )


TOOLS: dict[str, ai.Tool] = {
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
    messages: list[dict[str, Any]]


@dataclass
class ToolCallParams:
    tool_name: str
    tool_args: dict[str, Any]


# ── Activities ───────────────────────────────────────────────────


@activity.defn(name="llm_stream")
async def llm_stream_activity(params: LLMCallParams) -> LLMCallResult:
    """Call the LLM, drain the stream, return the final message."""
    llm = AnthropicModel(
        model="anthropic/claude-sonnet-4",
        base_url="https://ai-gateway.vercel.sh",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    messages = [ai.Message.model_validate(m) for m in params.messages]
    tools = [
        ai.Tool(
            name=t["name"],
            description=t["description"],
            schema=t["schema"],
            fn=lambda: None,
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
    tool = TOOLS[params.tool_name]
    return await tool.fn(**params.tool_args)
