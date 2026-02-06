import asyncio
import os
from collections.abc import AsyncGenerator

import rich

import vercel_ai_sdk as ai


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Simulated API call
    await asyncio.sleep(0.3)
    return f"Sunny, 72°F in {city}"


@ai.tool
async def get_population(city: str) -> int:
    """Get population of a city."""
    await asyncio.sleep(0.3)
    populations = {
        "new york": 8_336_817,
        "los angeles": 3_979_576,
        "chicago": 2_693_976,
    }
    return populations.get(city.lower(), 1_000_000)


@ai.stream
async def custom_stream_step(
    llm: ai.LanguageModel,
    messages: list[ai.Message],
    tools: list[ai.Tool],
    label: str | None = None,
) -> AsyncGenerator[ai.Message, None]:
    marker = ai.Message(
        role="assistant",
        parts=[ai.TextPart(text="start of a custom step", state="done")],
        label="custom_loop_marker",
    )
    yield marker
    async for msg in llm.stream(messages=messages, tools=tools):
        msg.label = label
        yield msg


async def agent(llm: ai.LanguageModel, user_query: str, runtime: ai.Runtime):
    """
    Custom agent loop with manual tool execution.

    Note: When implementing a custom loop (instead of using ai.stream_loop),
    you need to emit tool result messages manually using runtime.put_message()
    after tool execution. This ensures the UI adapter sees both the pending
    and result states of tool calls.
    """
    tools = [get_weather, get_population]

    messages = ai.make_messages(
        system="You are a robot assistant. Use the get_weather and get_population tools to answer questions.",
        user=user_query,
    )

    while True:
        result = await custom_stream_step(llm, messages, tools, label="agent")

        if not result.tool_calls:
            return result

        messages.append(result.last_message)

        await asyncio.gather(*(tc.execute() for tc in result.tool_calls))

        # Emit the message with tool results so the UI sees status="result"
        # (the stream_step already yielded the message with status="pending")
        if result.last_message:
            await runtime.put_message(result.last_message.model_copy(deep=True))


async def main():
    llm = ai.anthropic.AnthropicModel(
        model="anthropic/claude-sonnet-4",
        base_url="https://ai-gateway.vercel.sh",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    # Track tool calls we've already printed to avoid duplicates
    printed_tool_calls: set[str] = set()

    async for msg in ai.run(
        agent, llm, "What's the weather and population of New York and Los Angeles?"
    ):
        if msg.label == "custom_loop_marker":
            rich.print(f"\n[red]{msg.text}[/red]")
            printed_tool_calls.clear()  # Reset for new step
            continue

        # Show streaming text
        if msg.text_delta:
            rich.print(f"[blue]{msg.text_delta}[/blue]", end="", flush=True)

        # Show tool status (only print each tool call once)
        for part in msg.parts:
            if isinstance(part, ai.ToolPart):
                tool_id = part.tool_call_id
                if part.status == "pending" and part.state == "done":
                    if tool_id not in printed_tool_calls:
                        printed_tool_calls.add(tool_id)
                        rich.print(
                            f"\n[yellow]→ Calling {part.tool_name}({part.tool_args})[/yellow]"
                        )
                elif part.status == "result":
                    result_key = f"{tool_id}:result"
                    if result_key not in printed_tool_calls:
                        printed_tool_calls.add(result_key)
                        rich.print(f"[green]✓ {part.tool_name} = {part.result}[/green]")


if __name__ == "__main__":
    asyncio.run(main())
