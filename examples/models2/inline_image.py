"""Inline image generation — LLM that outputs images alongside text.

Models like Gemini 3 Pro Image can generate images as part of their
language model response. The images arrive as FileParts in the streamed
Message.
"""

import asyncio
import base64
import pathlib

from vercel_ai_sdk import models2 as m
from vercel_ai_sdk.types import messages as messages_

# This is a language model that can also output images inline.
model = m.Model(
    id="google/gemini-3-pro-image",
    api="ai-gateway",
    provider="ai-gateway",
    capabilities=("text", "image"),
)

messages = [
    messages_.Message(
        role="system",
        parts=[
            messages_.TextPart(
                text=(
                    "You are an anime art assistant. When asked to draw or create "
                    "an image, generate it in a soft pastel anime style."
                )
            ),
        ],
    ),
    messages_.Message(
        role="user",
        parts=[
            messages_.TextPart(
                text=(
                    "Draw an anime girl with long silver hair and violet eyes, "
                    "sitting in a field of cherry blossoms at sunset."
                )
            ),
        ],
    ),
]


async def main() -> None:
    last_msg: messages_.Message | None = None

    # Stream — text deltas arrive as usual, images arrive as FileParts
    async for msg in m.stream(model, messages):
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)
        last_msg = msg

    print()

    # Check for images in the final message
    if last_msg and last_msg.images:
        for i, img in enumerate(last_msg.images):
            filename = f"inline_{i}.png"
            data = (
                img.data if isinstance(img.data, bytes) else base64.b64decode(img.data)
            )
            pathlib.Path(filename).write_bytes(data)
            print(f"Saved {filename} ({img.media_type}, {len(data)} bytes)")
    else:
        print("No images were generated in this response.")


if __name__ == "__main__":
    asyncio.run(main())
