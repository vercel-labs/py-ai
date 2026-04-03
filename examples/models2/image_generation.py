"""Image generation — dedicated image model via generate()."""

import asyncio
import base64
import pathlib

from vercel_ai_sdk import models2 as m
from vercel_ai_sdk.types import messages as messages_

model = m.Model(
    id="google/imagen-4.0-generate-001",
    adapter="ai-gateway-v3",
    provider="ai-gateway",
    capabilities=("image",),
)

messages = [
    messages_.Message(
        role="user",
        parts=[
            messages_.TextPart(
                text=(
                    "Anime girl with twin tails and cat ears, wearing a "
                    "sailor school uniform, striking a victory pose in front "
                    "of a futuristic Tokyo skyline at night, neon lights "
                    "reflecting in her eyes, digital art style"
                )
            ),
        ],
    ),
]


async def main() -> None:
    result = await m.generate(model, messages, m.ImageParams(n=2, aspect_ratio="16:9"))

    print(f"Generated {len(result.images)} image(s)")
    for i, img in enumerate(result.images):
        filename = f"generated_{i}.png"
        data = img.data if isinstance(img.data, bytes) else base64.b64decode(img.data)
        pathlib.Path(filename).write_bytes(data)
        print(f"  {filename}: {img.media_type}, {len(data)} bytes")

    if result.usage:
        print(
            f"Usage: {result.usage.input_tokens} input, "
            f"{result.usage.output_tokens} output tokens"
        )


if __name__ == "__main__":
    asyncio.run(main())
