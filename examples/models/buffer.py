"""Buffered response — drain the stream, get the final message."""

import asyncio

import vercel_ai_sdk as ai
from vercel_ai_sdk import models as m

model = m.Model(
    id="anthropic/claude-sonnet-4",
    adapter="ai-gateway-v3",
    provider="ai-gateway",
)

messages = [ai.user_message("What is 2 + 2?")]


async def main() -> None:
    result = await m.buffer(m.stream(model, messages))
    print(result.text)
    if result.usage:
        print(
            f"tokens: {result.usage.input_tokens} in, {result.usage.output_tokens} out"
        )


if __name__ == "__main__":
    asyncio.run(main())
