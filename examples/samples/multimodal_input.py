"""Multimodal input — send a local image to the model and ask about it."""

import asyncio
import pathlib

import ai

model = ai.ai_gateway("anthropic/claude-sonnet-4")

# Load a local image file (replace with your own path).
image_path = pathlib.Path("sample_image.jpg")
image_data = image_path.read_bytes()

messages = [
    ai.user_message(
        "Describe this image in detail.",
        ai.file_part(image_data, media_type="image/jpeg"),
    ),
]


async def main() -> None:
    async for msg in await ai.stream(model, messages):
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
