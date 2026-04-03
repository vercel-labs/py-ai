"""Basic streaming — print text deltas as they arrive."""

import asyncio

from vercel_ai_sdk import models2 as m
from vercel_ai_sdk.types import messages as messages_

model = m.Model(
    id="anthropic/claude-sonnet-4",
    adapter="ai-gateway-v3",
    provider="ai-gateway",
)

messages = [
    messages_.Message(role="system", parts=[messages_.TextPart(text="Be concise.")]),
    messages_.Message(
        role="user",
        parts=[
            messages_.TextPart(text="Explain why the sky is blue in two sentences.")
        ],
    ),
]


async def main() -> None:
    async for msg in m.stream(model, messages):
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
