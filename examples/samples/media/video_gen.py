"""Video generation with Veo 3.

Uses the VideoModel interface to generate videos via the AI Gateway's
/video-model endpoint. The gateway handles the long-running generation
process (which can take minutes) and returns the result via SSE.

Usage:
    uv run examples/samples/media/video_gen.py
"""

import asyncio
import base64
import pathlib

import vercel_ai_sdk as ai


async def main() -> None:
    model = ai.ai_gateway.GatewayVideoModel(
        model="google/veo-3.0-generate-001",
    )

    # Generate a short anime-style video clip
    print("Generating video (this may take a minute or two)...")
    msg = await model.generate(
        ai.make_messages(
            user=(
                "An anime girl with long pink hair and a flowing white "
                "dress stands on a hilltop at golden hour. A warm breeze "
                "lifts her hair as she releases a paper lantern into the "
                "sunset sky. The camera slowly pulls back to reveal dozens "
                "of lanterns rising over a countryside village below. "
                "Soft cel-shaded anime art style, warm palette."
            ),
        ),
        aspect_ratio="16:9",
        duration=8,
    )

    print(f"Generated {len(msg.videos)} video(s)")
    for i, vid in enumerate(msg.videos):
        ext = "mp4" if "mp4" in vid.media_type else "webm"
        filename = f"lantern_girl_{i}.{ext}"
        data = vid.data if isinstance(vid.data, bytes) else base64.b64decode(vid.data)
        pathlib.Path(filename).write_bytes(data)
        print(f"  {filename}: {vid.media_type}, {len(data)} bytes")


if __name__ == "__main__":
    asyncio.run(main())
