"""Inline image generation — LLM that outputs images alongside text.

Models like Gemini 3 Pro Image can generate images as part of their
language model response. The images arrive as FileParts in the streamed
Message.
"""

import asyncio
import base64
import pathlib

import ai

model = ai.ai_gateway("google/gemini-3-pro-image")

messages = [
    ai.system_message(
        "You are an art assistant. When asked to draw or create an image, generate it."
    ),
    ai.user_message("Draw a cat sitting in a field of cherry blossoms at sunset."),
]


async def main() -> None:
    last_msg: ai.Message | None = None

    # Stream — text deltas arrive as usual, images arrive as FileParts
    async for msg in await ai.stream(model, messages):
        for ev in msg.deltas:
            if isinstance(ev.part, ai.TextPart):
                print(ev.chunk, end="", flush=True)
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
