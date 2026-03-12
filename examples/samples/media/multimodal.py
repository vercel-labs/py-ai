"""Multimodal input example: send an image URL to the model.

Usage:
    uv run examples/samples/media/multimodal.py
"""

import asyncio

import vercel_ai_sdk as ai

IMAGE_URL = (
    "https://4kwallpapers.com/images/wallpapers/hatsune-miku-3840x2160-15479.jpg"
)


async def agent(llm: ai.LanguageModel, user_query: str) -> ai.StreamResult:
    return await ai.stream_loop(
        llm,
        messages=[
            ai.Message(
                role="user",
                parts=[
                    ai.TextPart(text=user_query),
                    ai.FilePart.from_url(IMAGE_URL),
                ],
            )
        ],
        tools=[],
    )


async def main() -> None:
    llm = ai.ai_gateway.GatewayModel(model="anthropic/claude-opus-4.6")

    async for msg in ai.run(agent, llm, "What's in this image? Be concise."):
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
