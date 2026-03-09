"""Dedicated image generation model (Imagen 4).

Uses the ImageModel interface to generate images via the AI Gateway's
/image-model endpoint. Unlike language models, dedicated image models
are optimized purely for image generation with parameters like size,
aspect ratio, and seed.

Usage:
    uv run examples/samples/image_gen_dedicated.py
"""

import asyncio
import base64
import pathlib

import vercel_ai_sdk as ai


async def main() -> None:
    model = ai.ai_gateway.GatewayImageModel(
        model="google/imagen-4.0-generate-001",
    )

    # Generate two images of an anime girl character
    msg = await model.generate(
        ai.make_messages(
            user=(
                "Anime girl with twin tails and cat ears, wearing a "
                "sailor school uniform, striking a victory pose in front "
                "of a futuristic Tokyo skyline at night, neon lights "
                "reflecting in her eyes, digital art style"
            ),
        ),
        n=2,
        aspect_ratio="16:9",
    )

    print(f"Generated {len(msg.images)} images")
    for i, img in enumerate(msg.images):
        filename = f"neko_girl_{i}.png"
        data = img.data if isinstance(img.data, bytes) else base64.b64decode(img.data)
        pathlib.Path(filename).write_bytes(data)
        print(f"  {filename}: {img.media_type}, {len(data)} bytes")

    if msg.usage:
        print(
            f"Usage: {msg.usage.input_tokens} input, "
            f"{msg.usage.output_tokens} output tokens"
        )


if __name__ == "__main__":
    asyncio.run(main())
