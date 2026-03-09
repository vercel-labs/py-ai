"""Abstract base class for video generation models.

Video models accept messages (with text prompts and optional input images
for image-to-video) and return a :class:`Message` containing
:class:`FilePart`\\s for each generated video.

Usage::

    model = ai.ai_gateway.GatewayVideoModel(model="google/veo-3.0-generate-001")
    msg = await model.generate(
        ai.make_messages(user="A cat walking on a beach at sunset"),
        aspect_ratio="16:9",
        duration=5,
    )
    for vid in msg.videos:
        print(vid.media_type, len(vid.data))
"""

from __future__ import annotations

import abc
from typing import Any

from . import messages as messages_


class VideoModel(abc.ABC):
    """Abstract video generation model.

    Accepts :class:`Message`\\s as input and returns a :class:`Message`
    containing generated videos as :class:`FilePart`\\s.
    """

    @abc.abstractmethod
    async def generate(
        self,
        messages: list[messages_.Message],
        *,
        n: int = 1,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        duration: float | None = None,
        fps: int | None = None,
        seed: int | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> messages_.Message:
        """Generate videos from the given messages.

        Args:
            messages: Input messages containing the text prompt and optional
                input image (as a :class:`FilePart`) for image-to-video.
            n: Number of videos to generate.
            aspect_ratio: Aspect ratio (e.g. ``"16:9"``).
            resolution: Video resolution (e.g. ``"1920x1080"``).
            duration: Duration in seconds.
            fps: Frames per second.
            seed: Random seed for reproducible generation.
            provider_options: Provider-specific options.

        Returns:
            A :class:`Message` with ``role="assistant"`` containing
            :class:`FilePart`\\s for each generated video.
        """
        ...
