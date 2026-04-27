"""Basic streaming — print text deltas as they arrive."""

import asyncio

import ai

model = ai.ai_gateway("anthropic/claude-sonnet-4")

messages = [
    ai.system_message("Be concise."),
    ai.user_message("Explain why the sky is blue in two sentences."),
]


async def main() -> None:
    async for event in ai.stream(model, messages):
        if isinstance(event, ai.TextDelta):
            print(event.chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
