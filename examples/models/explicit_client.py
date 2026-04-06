"""Explicit client — bring your own auth and base URL."""

import asyncio
import os

import vercel_ai_sdk as ai
from vercel_ai_sdk import models as m

model = m.Model(
    id="anthropic/claude-sonnet-4",
    adapter="ai-gateway-v3",
    provider="ai-gateway",
)

# Explicit client — useful for custom auth, proxies, or self-hosted gateways.
client = m.Client(
    base_url="https://ai-gateway.vercel.sh/v3/ai",
    api_key=os.environ["AI_GATEWAY_API_KEY"],
    headers={"X-Custom-Header": "example"},
)

messages = [ai.user_message("Hello!")]


async def main() -> None:
    try:
        async for msg in m.stream(model, messages, client=client):
            if msg.text_delta:
                print(msg.text_delta, end="", flush=True)
        print()
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
