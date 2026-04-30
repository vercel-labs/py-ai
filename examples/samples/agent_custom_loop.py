"""Custom loop: manual control over streaming and tool execution."""

import asyncio
from collections.abc import AsyncGenerator

import ai


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72F in {city}"


@ai.tool
async def get_population(city: str) -> int:
    """Get population of a city."""
    return {"new york": 8_336_817, "tokyo": 13_960_000}.get(city.lower(), 1_000_000)


async def main() -> None:
    model = ai.ai_gateway("anthropic/claude-sonnet-4")

    tools = [get_weather, get_population]
    my_agent = ai.agent(tools=tools)

    @my_agent.loop
    async def custom(context: ai.Context) -> AsyncGenerator[ai.Event]:
        """Stream, execute tools with logging, repeat."""
        while True:
            s = ai.models.stream(context.model, context.messages, tools=context.tools)
            async for event in s:
                yield event

            # Yield the assistant message for silent history collection.
            if s.message is not None:
                yield s.message

            tool_calls = context.resolve(s.tool_calls)
            if not tool_calls:
                return

            print(
                f"\n  [calling {len(tool_calls)} tool(s): "
                f"{', '.join(tc.name for tc in tool_calls)}]"
            )
            # Each resolved tool call exposes tc.fn and tc.kwargs, and
            # tc(**overrides) lets you adjust arguments before execution.

            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(tc()) for tc in tool_calls]

            yield ai.tool_result(*(t.result() for t in tasks))

    async for event in my_agent.run(
        model,
        [ai.user_message("Compare the weather and population of New York and Tokyo.")],
    ):
        if isinstance(event, ai.TextDelta):
            print(event.chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
