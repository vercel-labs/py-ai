"""Agent logic for the chat demo.

Demonstrates human-in-the-loop tool approval using ToolApproval hooks.
Every tool call is gated behind user approval before execution.
"""

import asyncio
from typing import Any

import ai


@ai.tool
async def talk_to_mothership(question: str) -> str:
    """Contact the mothership for important decisions."""
    return f"Mothership says: {question} -> Soon."


MODEL = ai.Model(
    id="anthropic/claude-opus-4.6",
    adapter="ai-gateway-v3",
    provider="ai-gateway",
)

TOOLS: list[ai.Tool[..., Any]] = [talk_to_mothership]


async def _execute_with_approval(
    tc: ai.ToolPart, message: ai.Message | None = None
) -> ai.ToolPart:
    """Execute a tool call only after the user grants approval.

    Creates a ToolApproval hook that suspends execution until the
    frontend responds with an approve/reject decision.
    Returns the updated (immutable) ToolPart with the result.
    """
    approval = await ai.ToolApproval.create(  # type: ignore[attr-defined]
        f"approve_{tc.tool_call_id}",
        metadata={"tool_name": tc.tool_name, "tool_args": tc.tool_args},
    )

    if approval.granted:
        return await ai.execute_tool(tc, message=message)
    return tc.with_error("Tool call was denied by the user.")


chat_agent = ai.agent(
    model=MODEL,
    system="",
    tools=TOOLS,
)


@chat_agent.loop
async def graph(
    agent: ai.Agent,
    messages: list[ai.Message],
) -> ai.StreamResult:
    """Agent graph with human-in-the-loop tool approval.

    Loops: stream LLM -> request approval -> execute tools -> repeat.
    The ToolApproval hook suspends execution and emits an approval-
    request event on the SSE stream.  The frontend displays Approve /
    Reject buttons and sends the decision back on the next request.
    """
    local_messages = list(messages)

    while True:
        result = await ai.stream_step(agent.model, local_messages, agent.tools)

        if not result.tool_calls:
            return result

        last_msg = result.last_message
        assert last_msg is not None

        updated_parts = await asyncio.gather(
            *(_execute_with_approval(tc, message=last_msg) for tc in result.tool_calls)
        )
        updated_msg = last_msg
        for updated_tc in updated_parts:
            updated_msg = updated_msg.replace(updated_tc)
        local_messages.append(updated_msg)
