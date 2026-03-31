"""Tools — pass tool schemas to the model."""

import asyncio

from vercel_ai_sdk import models2 as m
from vercel_ai_sdk.types import messages as messages_
from vercel_ai_sdk.types import tools as tools_

model = m.Model(
    id="anthropic/claude-sonnet-4",
    api="ai-gateway",
    provider="ai-gateway",
)

# Define a tool schema — anything matching the ToolLike protocol works.
get_weather = tools_.ToolSchema(
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

messages = [
    messages_.Message(
        role="user",
        parts=[messages_.TextPart(text="What's the weather in Tokyo?")],
    ),
]


async def main() -> None:
    # Stream with tools — the model may emit tool calls
    async for msg in m.stream(model, messages, tools=[get_weather]):
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)

        for tc in msg.tool_calls:
            if tc.state == "done":
                print(f"\nTool call: {tc.tool_name}({tc.tool_args})")
    print()


if __name__ == "__main__":
    asyncio.run(main())
