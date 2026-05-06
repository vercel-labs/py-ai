"""Streaming from inside a tool via an async generator.

An async generator tool yields values that flow through the runtime
sink to the consumer in real time as ``PartialToolCallResult`` events.
The tool's ``aggregator`` decides how the yielded values are combined
into the final tool result that goes back to the model.
"""

import asyncio
from collections.abc import AsyncGenerator

import ai


@ai.tool(aggregator=ai.LastAggregator)
async def talk_to_mothership(question: str) -> AsyncGenerator[str]:
    """Ask the mothership a question. Streams progress back to the caller."""
    for step in ["Connecting...", "Transmitting...", f"Asking: {question!r}..."]:
        yield step
        await asyncio.sleep(0.3)

    # LastAggregator keeps only the final yield, which becomes the tool result.
    yield "The mothership says: Soon."


async def main() -> None:
    model = ai.ai_gateway("anthropic/claude-sonnet-4")

    my_agent = ai.agent(tools=[talk_to_mothership])

    messages = [
        ai.system_message("Use the mothership tool when asked about the future."),
        ai.user_message("When will the robots take over?"),
    ]

    async for event in my_agent.run(model, messages):
        if isinstance(event, ai.PartialToolCallResult):
            print(f"  [{event.value}]")
        elif isinstance(event, ai.TextDelta):
            print(event.chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
