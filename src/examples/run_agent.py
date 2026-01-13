"""Minimal example of using the LanguageModel to stream a completion with tools."""

import asyncio
import os

from proto_sdk import core
from proto_sdk.openai import OpenAIModel

import dotenv

dotenv.load_dotenv()


@core.tool
async def get_weather(location: str) -> dict:
    """Get the current weather for a location."""
    return {"temperature": 72, "condition": "sunny", "location": location}


@core.tool
async def add_numbers(a: int, b: int) -> dict:
    """Add two numbers together."""
    return {"result": a + b}


async def main():
    model = "anthropic/claude-sonnet-4.5"
    base_url = "https://ai-gateway.vercel.sh/v1"
    api_key = os.environ.get("AI_GATEWAY_API_KEY")
    model = OpenAIModel(model=model, api_key=api_key, base_url=base_url)
    tools = [get_weather, add_numbers]

    messages = [
        core.Message(
            role="user",
            parts=[core.TextPart(text="What's the weather in San Francisco?")],
        )
    ]

    print("Streaming response...")
    async for msg in model.stream(messages, tools):
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)
        if msg.is_done:
            print("\n---")
            print(f"Final message parts: {msg.parts}")


if __name__ == "__main__":
    asyncio.run(main())
