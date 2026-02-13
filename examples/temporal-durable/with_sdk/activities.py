"""Temporal activities — all real I/O lives here.

Each tool is its own activity with a plain function signature.
The LLM activity uses ToolSchema (no dummy fn) and llm.buffer()
(no manual drain loop).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from temporalio import activity

import vercel_ai_sdk as ai
from vercel_ai_sdk.anthropic import AnthropicModel


# ── Tool activities (one per tool, plain functions) ───────────────


@activity.defn(name="get_weather")
async def get_weather_activity(city: str) -> str:
    return f"Sunny, 72F in {city}"


@activity.defn(name="get_population")
async def get_population_activity(city: str) -> int:
    return {"new york": 8_336_817, "los angeles": 3_979_576}.get(
        city.lower(), 1_000_000
    )


# ── LLM activity ─────────────────────────────────────────────────


@dataclass
class LLMCallParams:
    messages: list[dict[str, Any]]
    tool_schemas: list[dict[str, Any]]


@dataclass
class LLMCallResult:
    message: dict[str, Any]  # serialized ai.Message


@activity.defn(name="llm_call")
async def llm_call_activity(params: LLMCallParams) -> LLMCallResult:
    """Call the LLM, drain the stream, return the final message."""
    llm = AnthropicModel(
        model="anthropic/claude-sonnet-4",
        base_url="https://ai-gateway.vercel.sh",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    messages = [ai.Message.model_validate(m) for m in params.messages]
    tools = [ai.ToolSchema.model_validate(t) for t in params.tool_schemas]

    result = await llm.buffer(messages, tools)
    return LLMCallResult(message=result.model_dump())
