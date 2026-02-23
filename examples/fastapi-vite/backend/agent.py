"""Agent logic for the chat demo."""

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


async def graph(
    llm: ai.LanguageModel,
    messages: list[ai.Message],
    tools: list[ai.Tool[..., Any]],
) -> ai.StreamResult:
    """
    Agent graph: stream LLM, execute tools, repeat until done.

    This is a plain async function that goes through the Runtime queue
    via stream_loop. When hooks are added later, they slot in here
    between tool calls — no structural change needed.
    """
    return await ai.stream_loop(llm, messages, tools)
