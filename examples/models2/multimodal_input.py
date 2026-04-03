"""Multimodal input — send a local image to the model and ask about it."""

import asyncio
import pathlib

from vercel_ai_sdk import models2 as m
from vercel_ai_sdk.types import messages as messages_

model = m.Model(
    id="anthropic/claude-sonnet-4",
    adapter="ai-gateway-v3",
    provider="ai-gateway",
)

# Load a local image file (replace with your own path)
image_path = pathlib.Path("sample_image.jpg")
image_data = image_path.read_bytes()

messages = [
    messages_.Message(
        role="user",
        parts=[
            messages_.TextPart(text="Describe this image in detail."),
            messages_.FilePart(data=image_data, media_type="image/jpeg"),
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
