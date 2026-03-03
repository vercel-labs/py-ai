"""Agent logic for the chat demo.

Demonstrates human-in-the-loop tool approval using ToolApproval hooks.
Every tool call is gated behind user approval before execution.
"""

import asyncio
from typing import Any

import vercel_ai_sdk as ai


@ai.tool
async def talk_to_mothership(question: str) -> str:
    """Contact the mothership for important decisions."""
    return f"Mothership says: {question} -> Soon."


def get_llm() -> ai.LanguageModel:
    """Create the LLM instance."""
    return ai.ai_gateway.GatewayModel(model="anthropic/claude-opus-4.6")


TOOLS: list[ai.Tool[..., Any]] = [talk_to_mothership]


async def _execute_with_approval(
    tc: ai.ToolPart, message: ai.Message | None = None
) -> None:
    """Execute a tool call only after the user grants approval.

    Creates a ToolApproval hook that suspends execution until the
    frontend responds with an approve/reject decision.
    """
    approval = await ai.ToolApproval.create(  # type: ignore[attr-defined]
        f"approve_{tc.tool_call_id}",
        metadata={"tool_name": tc.tool_name, "tool_args": tc.tool_args},
    )

    if approval.granted:
        await ai.execute_tool(tc, message=message)
    else:
        tc.set_error("Tool call was denied by the user.")


async def graph(
    llm: ai.LanguageModel,
    messages: list[ai.Message],
    tools: list[ai.Tool[..., Any]],
) -> ai.StreamResult:
    """Agent graph with human-in-the-loop tool approval.

    Loops: stream LLM -> request approval -> execute tools -> repeat.
    The ToolApproval hook suspends execution and emits an approval-
    request event on the SSE stream.  The frontend displays Approve /
    Reject buttons and sends the decision back on the next request.
    """
    local_messages = list(messages)

    while True:
        result = await ai.stream_step(llm, local_messages, tools)

        if not result.tool_calls:
            return result

        last_msg = result.last_message
        assert last_msg is not None
        local_messages.append(last_msg)

        await asyncio.gather(
            *(_execute_with_approval(tc, message=last_msg) for tc in result.tool_calls)
        )
