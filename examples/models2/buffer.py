"""Buffered response — drain the stream, get the final message."""

import asyncio

from vercel_ai_sdk import models2 as m
from vercel_ai_sdk.types import messages as messages_

model = m.Model(
    id="anthropic/claude-sonnet-4",
    api="ai-gateway",
    provider="ai-gateway",
)

messages = [
    messages_.Message(
        role="user",
        parts=[messages_.TextPart(text="What is 2 + 2?")],
    ),
]


async def main() -> None:
    result = await m.buffer(m.stream(model, messages))
    print(result.text)
    if result.usage:
        print(
            f"tokens: {result.usage.input_tokens} in, {result.usage.output_tokens} out"
        )


if __name__ == "__main__":
    asyncio.run(main())
