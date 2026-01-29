"""Raw building blocks for LLM streaming and tool execution."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from . import messages as messages_
from . import tools as tools_
from . import decorators as decorators_
from . import step as step_

if TYPE_CHECKING:
    from . import runtime as runtime_


async def raw_stream_llm(
    llm: Any,  # LanguageModel, but avoiding circular import
    messages: list[messages_.Message],
    tools: list[tools_.Tool] | None = None,
) -> AsyncGenerator[messages_.Message, None]:
    """Raw LLM streaming - yields messages, no Runtime interaction."""
    async for msg in llm.stream(messages=messages, tools=tools):
        yield msg


async def execute_tool(
    tool_call: step_.ToolCall,
    tools: list[tools_.Tool],
    message: messages_.Message | None = None,
) -> Any:
    """
    Execute a single tool call and optionally update the message.
    
    If message is provided, updates the tool part with the result.
    Can be wrapped with @workflow.step for durability.
    Use with asyncio.gather() for parallel execution.
    
    Example:
        await asyncio.gather(*(
            ai.execute_tool(tc, tools, result.last_message)
            for tc in result.tool_calls
        ))
    """
    tool_fn = next((t for t in tools if t.name == tool_call.tool_name), None)
    if tool_fn is None:
        raise ValueError(f"Tool not found: {tool_call.tool_name}")
    
    result = await tool_fn.fn(**tool_call.tool_args)
    
    # Update message if provided
    if message is not None:
        tool_part = message.get_tool_part(tool_call.tool_call_id)
        if tool_part:
            tool_part.status = "result"
            tool_part.result = result
    
    return result


# --- User-facing streaming functions ---


@decorators_.stream
async def stream_step(
    llm: "runtime_.LanguageModel",
    messages: list[messages_.Message],
    tools: list[tools_.Tool] | None = None,
    label: str | None = None,
) -> AsyncGenerator[messages_.Message, None]:
    """
    Single LLM call that streams to Runtime.
    
    Returns StepResult with .tool_calls, .text, .last_message when awaited.
    Can be wrapped with @workflow.step for durability.
    """
    async for msg in llm.stream(messages=messages, tools=tools):
        msg.label = label
        yield msg
