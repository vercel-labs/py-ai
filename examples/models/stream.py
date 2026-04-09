"""Basic streaming — print text deltas as they arrive."""

import asyncio

import ai
from ai import models as m

model = m.Model(
    id="anthropic/claude-sonnet-4",
    adapter="ai-gateway-v3",
    provider="ai-gateway",
)

messages = [
    ai.system_message("Be concise."),
    ai.user_message("Explain why the sky is blue in two sentences."),
]


async def main() -> None:
    async for msg in m.stream(model, messages):
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
