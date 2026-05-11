"""Gemini direct - stream from Google's Gemini API."""

import asyncio
import sys

import ai

if ai.google.client().api_key is None:
    print("[SKIP] GOOGLE_API_KEY or GEMINI_API_KEY not set")
    sys.exit(0)

model = ai.google("gemini-3.1-flash-lite")

messages = [
    ai.system_message("Be concise."),
    ai.user_message("Explain why the sky is blue in two sentences."),
]


async def main() -> None:
    async with ai.stream(model, messages) as s:
        async for event in s:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
