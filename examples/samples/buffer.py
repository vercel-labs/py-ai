"""Buffered response — drain the stream, get the final message."""

import asyncio

import ai
from ai import models as m

model = m.Model(
    id="anthropic/claude-sonnet-4",
    adapter="ai-gateway-v3",
    provider="ai-gateway",
)

messages = [ai.user_message("What is 2 + 2?")]


async def main() -> None:
    s = await m.stream(model, messages)
    result = await m.buffer(s)  # type: ignore[arg-type]  # StreamResult is async-iterable
    print(result.text)
    if result.usage:
        print(
            f"tokens: {result.usage.input_tokens} in, {result.usage.output_tokens} out"
        )


if __name__ == "__main__":
    asyncio.run(main())
