"""Minimal agent using Collector.stream() with get_root()."""

import asyncio
import os
import random

from proto_sdk import core
from proto_sdk.openai import OpenAIModel
from rich.console import Console

import dotenv

dotenv.load_dotenv()

console = Console()


@core.tool
async def roll_dice(sides: int = 6) -> dict:
    """Roll a dice with the specified number of sides."""
    return {"result": random.randint(1, sides)}


@core.tool
async def get_weather(location: str) -> dict:
    """Get the current weather for a location."""
    return {"temperature": 72, "condition": "sunny", "location": location}


async def main():
    llm = OpenAIModel(
        model="anthropic/claude-sonnet-4.5",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
        base_url="https://ai-gateway.vercel.sh/v1",
    )

    messages = [
        core.Message(
            role="user",
            parts=[core.TextPart(text="Roll a d20 and check Tokyo weather.")],
        )
    ]

    root = core.get_root(llm, messages, tools=[roll_dice, get_weather])

    async for msg in core.Collector().stream(root):
        console.print(msg)
        if msg.is_done:
            print()


if __name__ == "__main__":
    asyncio.run(main())
