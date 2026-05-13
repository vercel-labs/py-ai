"""Explicit provider — bring your own auth and base URL."""

import asyncio
import os

import ai

provider = ai.get_provider(
    "vercel",
    base_url="https://ai-gateway.vercel.sh/v3/ai",
    api_key=os.environ["AI_GATEWAY_API_KEY"],
)

model = ai.Model("anthropic/claude-sonnet-4", provider=provider)

messages = [ai.user_message("Hello!")]


async def main() -> None:
    try:
        async with ai.stream(model, messages) as s:
            async for event in s:
                if isinstance(event, ai.events.TextDelta):
                    print(event.chunk, end="", flush=True)
        print()
    finally:
        await provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
