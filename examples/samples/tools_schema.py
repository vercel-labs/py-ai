"""Tools — pass tool schemas to the model."""

import asyncio

import ai

model = ai.ai_gateway("anthropic/claude-sonnet-4")

# Define a tool schema — anything matching the ToolLike protocol works.
get_weather = ai.ToolSchema(
    name="get_weather",
    description="Get the current weather for a city.",
    param_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The city name"},
        },
        "required": ["city"],
    },
    return_type=str,
)

messages = [ai.user_message("What's the weather in Tokyo?")]


async def main() -> None:
    # Stream with tools — the model may emit tool calls
    async for msg in await ai.stream(model, messages, tools=[get_weather]):
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)

        if msg.is_done:
            for tc in msg.tool_calls:
                print(f"\nTool call: {tc.tool_name}({tc.tool_args})")
    print()


if __name__ == "__main__":
    asyncio.run(main())
