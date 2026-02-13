"""Temporal activities — all real I/O lives here.

Uses the anthropic SDK directly. No framework.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import anthropic
from temporalio import activity

MODEL = "anthropic/claude-sonnet-4"


# ── Tool functions ───────────────────────────────────────────────


def get_weather(city: str) -> str:
    return f"Sunny, 72F in {city}"


def get_population(city: str) -> int:
    return {"new york": 8_336_817, "los angeles": 3_979_576}.get(
        city.lower(), 1_000_000
    )


TOOL_FNS = {
    "get_weather": get_weather,
    "get_population": get_population,
}

# Anthropic tool schemas — what the LLM sees.
TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "get_population",
        "description": "Get population of a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
]


# ── Serializable parameter types ─────────────────────────────────


@dataclass
class LLMCallParams:
    messages: list[dict[str, Any]]


@dataclass
class LLMCallResult:
    response: dict[str, Any]  # raw Anthropic response


@dataclass
class ToolCallParams:
    tool_name: str
    tool_args: dict[str, Any]


# ── Activities ───────────────────────────────────────────────────


@activity.defn(name="llm_call")
async def llm_call_activity(params: LLMCallParams) -> LLMCallResult:
    """Call the Anthropic API, return the full response."""
    client = anthropic.AsyncAnthropic(
        base_url="https://ai-gateway.vercel.sh",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )
    response = await client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system="Answer questions using the weather and population tools.",
        messages=params.messages,
        tools=TOOL_SCHEMAS,
    )
    return LLMCallResult(response=response.model_dump())


@activity.defn(name="tool_call")
async def tool_call_activity(params: ToolCallParams) -> Any:
    """Execute a tool function by name."""
    fn = TOOL_FNS[params.tool_name]
    return fn(**params.tool_args)
