"""Direct adapter call — bypass the registry, call the adapter function directly."""

import asyncio
import os

from vercel_ai_sdk import models2 as m
from vercel_ai_sdk.models2.ai_gateway import adapter as ai_gateway_v3
from vercel_ai_sdk.types import messages as messages_

model = m.Model(
    id="anthropic/claude-sonnet-4",
    adapter="ai-gateway-v3",
    provider="ai-gateway",
)

client = m.Client(
    base_url="https://ai-gateway.vercel.sh/v3/ai",
    api_key=os.environ["AI_GATEWAY_API_KEY"],
)

messages = [
    messages_.Message(
        role="user",
        parts=[messages_.TextPart(text="Say hello in three languages.")],
    ),
]


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
