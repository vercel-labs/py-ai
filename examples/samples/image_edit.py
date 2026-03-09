"""Image editing with a dedicated image model.

Demonstrates sending an input image to be edited/transformed by the
image model. The input image is passed as a FilePart in the user
message, and the model returns the edited version.

Usage:
    uv run examples/samples/image_edit.py
"""

import asyncio
import base64
import pathlib

import vercel_ai_sdk as ai


async def main() -> None:
    model = ai.ai_gateway.GatewayImageModel(
        model="openai/gpt-image-1",
    )

    # Load an existing image to use as input for editing.
    # In practice you would load a real image file:
    #   image_data = pathlib.Path("my_photo.png").read_bytes()
    #   input_image = ai.FilePart.from_bytes(image_data, media_type="image/png")
    input_image = ai.FilePart.from_url(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        media_type="image/jpeg",
    )

    # Ask the model to transform the photo into anime style
    msg = await model.generate(
        [
            ai.Message(
                role="user",
                parts=[
                    ai.TextPart(
                        text=(
                            "Transform this photo into a soft watercolor "
                            "anime style. Turn the cat into an anime girl "
                            "with cat ears and a tail, sitting in the same "
                            "pose. Add cherry blossom petals falling gently "
                            "in the background."
                        )
                    ),
                    input_image,
                ],
            )
        ],
        size="1024x1024",
    )

    print(f"Generated {len(msg.images)} edited image(s)")
    for i, img in enumerate(msg.images):
        filename = f"catgirl_edit_{i}.png"
        data = img.data if isinstance(img.data, bytes) else base64.b64decode(img.data)
        pathlib.Path(filename).write_bytes(data)
        print(f"  {filename}: {img.media_type}, {len(data)} bytes")


if __name__ == "__main__":
    asyncio.run(main())
