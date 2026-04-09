"""Direct adapter call — bypass the registry, call the adapter function directly."""

import asyncio
import os

import ai
from ai import models as m
from ai.models import ai_gateway as ai_gateway_v3

model = m.Model(
    id="anthropic/claude-sonnet-4",
    adapter="ai-gateway-v3",
    provider="ai-gateway",
)

client = m.Client(
    base_url="https://ai-gateway.vercel.sh/v3/ai",
    api_key=os.environ["AI_GATEWAY_API_KEY"],
)

messages = [ai.user_message("Say hello in three languages.")]


async def main() -> None:
    # Call the adapter function directly — no registry lookup, no auto-client.
    # This is the lowest level of the API.
    try:
        async for msg in ai_gateway_v3.stream(client, model, messages):
            if msg.text_delta:
                print(msg.text_delta, end="", flush=True)
        print()
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
