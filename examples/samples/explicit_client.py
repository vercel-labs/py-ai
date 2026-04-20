"""Explicit client — bring your own auth and base URL."""

import asyncio
import os

import ai

# Explicit client — useful for custom auth, proxies, or self-hosted gateways.
client = ai.Client(
    base_url="https://ai-gateway.vercel.sh/v3/ai",
    api_key=os.environ["AI_GATEWAY_API_KEY"],
    headers={"X-Custom-Header": "example"},
)

model = ai.ai_gateway("anthropic/claude-sonnet-4", client=client)

messages = [ai.user_message("Hello!")]


async def main() -> None:
    try:
        async for msg in await ai.models.stream(model, messages):
            for ev in msg.deltas:
                if isinstance(ev.part, ai.TextPart):
                    print(ev.chunk, end="", flush=True)
        print()
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
