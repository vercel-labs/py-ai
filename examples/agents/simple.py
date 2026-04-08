"""Simplest agent: model + tool, default loop handles everything."""

import asyncio

import vercel_ai_sdk as ai
from vercel_ai_sdk.agents import agent, tool


@tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72F in {city}"


async def main() -> None:
    model = ai.Model(
        id="anthropic/claude-sonnet-4-20250514",
        adapter="ai-gateway-v3",
        provider="ai-gateway",
    )

    my_agent = agent(
        system="You are a helpful weather assistant.",
        tools=[get_weather],
    )

    async for msg in my_agent.run(
        model, [ai.user_message("What's the weather in Tokyo?")]
    ):
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
