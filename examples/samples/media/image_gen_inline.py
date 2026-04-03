"""Inline image generation via a language model (Gemini 3 Pro Image).

Models like Gemini 3 Pro Image and GPT-5 can generate images alongside
text as part of their language model response. The images arrive as
FileParts in the streamed Message.

Usage:
    uv run examples/samples/media/image_gen_inline.py
"""

import asyncio
import base64
import pathlib

import vercel_ai_sdk as ai


async def main() -> None:
    # Gemini 3 Pro Image is a language model that can output images inline
    model = ai.Model(
        id="google/gemini-3-pro-image",
        adapter="ai-gateway-v3",
        provider="ai-gateway",
    )

    prompt = (
        "Draw an anime girl with long silver hair and violet eyes, "
        "sitting in a field of cherry blossoms at sunset. "
        "She's wearing a traditional kimono and reading a book."
    )

    my_agent = ai.agent(
        model=model,
        system=(
            "You are an anime art assistant. When asked to draw or create "
            "an image, generate it in a soft pastel anime style with "
            "detailed backgrounds and expressive characters."
        ),
        tools=[],
    )

    async for msg in my_agent.run(ai.make_messages(user=prompt)):
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)

    print()

    # The final message may contain both text and images
    if msg.images:
        for i, img in enumerate(msg.images):
            filename = f"sakura_girl_{i}.png"
            data = (
                img.data if isinstance(img.data, bytes) else base64.b64decode(img.data)
            )
            pathlib.Path(filename).write_bytes(data)
            print(f"Saved {filename} ({img.media_type}, {len(data)} bytes)")
    else:
        print("No images were generated in this response.")


if __name__ == "__main__":
    asyncio.run(main())
