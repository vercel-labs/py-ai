"""Streaming from inside a tool via an async generator.

An async generator tool yields messages that flow through the runtime
sink to the consumer in real time. The final yielded message's text
becomes the tool result.
"""

import asyncio
from collections.abc import AsyncGenerator

import ai
from ai.agents import agent, tool


@tool  # type: ignore[arg-type]  # async generator tools are supported at runtime
async def talk_to_mothership(question: str) -> AsyncGenerator[ai.Message]:
    """Ask the mothership a question. Streams progress back to the caller."""
    for step in ["Connecting...", "Transmitting...", "Awaiting response..."]:
        yield ai.Message(
            role="assistant",
            parts=[ai.TextPart(text=step, state="done")],
            label="tool_progress",
        )
        await asyncio.sleep(0.3)

    # The final yielded message's text is returned as the tool result.
    yield ai.Message(
        role="assistant",
        parts=[ai.TextPart(text="The mothership says: Soon.", state="done")],
    )


async def main() -> None:
    model = ai.Model(
        id="anthropic/claude-sonnet-4-20250514",
        adapter="ai-gateway-v3",
        provider="ai-gateway",
    )

    my_agent = agent(tools=[talk_to_mothership])

    messages = [
        ai.system_message("Use the mothership tool when asked about the future."),
        ai.user_message("When will the robots take over?"),
    ]

    async for msg in my_agent.run(model, messages):
        if msg.label == "tool_progress":
            print(f"  [{msg.text}]")
        elif msg.text_delta:
            print(msg.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
