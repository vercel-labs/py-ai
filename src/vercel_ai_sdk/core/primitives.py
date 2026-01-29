"""Raw building blocks for LLM streaming and tool execution."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from . import messages as messages_
from . import tools as tools_


async def stream_llm(
    llm: Any,  # LanguageModel, but avoiding circular import
    messages: list[messages_.Message],
    tools: list[tools_.Tool] | None = None,
) -> AsyncGenerator[messages_.Message, None]:
    """Raw LLM streaming - yields messages, no Runtime interaction."""
    async for msg in llm.stream(messages=messages, tools=tools):
        yield msg


async def execute_tool(tool: tools_.Tool, args: dict[str, Any]) -> Any:
    """Raw tool execution."""
    return await tool.fn(**args)


def extract_tool_calls(message: messages_.Message) -> list[dict[str, Any]]:
    """Extract tool calls from a completed message."""
    tool_calls = []
    for part in message.parts:
        if isinstance(part, messages_.ToolPart):
            tool_calls.append(
                {
                    "tool_call_id": part.tool_call_id,
                    "tool_name": part.tool_name,
                    "tool_args": json.loads(part.tool_args)
                    if isinstance(part.tool_args, str)
                    else part.tool_args,
                }
            )
    return tool_calls
