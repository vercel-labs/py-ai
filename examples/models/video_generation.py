"""Video generation — dedicated video model via generate()."""

import asyncio
import base64
import pathlib

import ai
from ai import models as m

model = m.Model(
    id="google/veo-3.0-generate-001",
    adapter="ai-gateway-v3",
    provider="ai-gateway",
    capabilities=("video",),
)

messages = [
    ai.user_message(
        "An anime girl with long pink hair and a flowing white "
        "dress stands on a hilltop at golden hour. A warm breeze "
        "lifts her hair as she releases a paper lantern into the "
        "sunset sky. Soft cel-shaded anime art style, warm palette."
    ),
]


async def main() -> None:
    print("Generating video (this may take a minute or two)...")

    result = await m.generate(
        model,
        messages,
        m.VideoParams(aspect_ratio="16:9", duration=8),
    )

    print(f"Generated {len(result.videos)} video(s)")
    for i, vid in enumerate(result.videos):
        ext = "mp4" if "mp4" in vid.media_type else "webm"
        filename = f"generated_{i}.{ext}"
        data = vid.data if isinstance(vid.data, bytes) else base64.b64decode(vid.data)
        pathlib.Path(filename).write_bytes(data)
        print(f"  {filename}: {vid.media_type}, {len(data)} bytes")


if __name__ == "__main__":
    asyncio.run(main())
