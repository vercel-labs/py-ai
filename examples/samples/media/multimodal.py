"""Multimodal input example: send an image URL to the model.

Usage:
    uv run examples/samples/media/multimodal.py
"""

import asyncio

import vercel_ai_sdk as ai

IMAGE_URL = (
    "https://4kwallpapers.com/images/wallpapers/hatsune-miku-3840x2160-15479.jpg"
)


async def main() -> None:
    model = ai.Model(
        id="anthropic/claude-opus-4.6",
        adapter="ai-gateway-v3",
        provider="ai-gateway",
    )

    my_agent = ai.agent(model=model, tools=[])

    async for msg in my_agent.run(
        [
            ai.Message(
                role="user",
                parts=[
                    ai.TextPart(text="What's in this image? Be concise."),
                    ai.FilePart.from_url(IMAGE_URL),
                ],
            )
        ]
    ):
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
