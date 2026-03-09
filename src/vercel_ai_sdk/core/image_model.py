"""Abstract base class for image generation models.

Image models accept messages (with text prompts and optional input images)
and return a :class:`Message` containing :class:`FilePart`\\s for each
generated image.

Usage::

    model = ai.ai_gateway.GatewayImageModel(model="google/imagen-4.0-generate-001")
    msg = await model.generate(
        ai.make_messages(user="A sunset over Tokyo"),
        n=2,
        aspect_ratio="16:9",
    )
    for img in msg.images:
        print(img.media_type, len(img.data))
"""

from __future__ import annotations

import abc
from typing import Any

from . import messages as messages_


def extract_prompt(messages: list[messages_.Message]) -> str:
    """Extract a text prompt from user messages.

    Concatenates all :class:`TextPart` content from ``user`` and ``system``
    messages into a single prompt string.
    """
    parts: list[str] = []
    for msg in messages:
        if msg.role in ("user", "system"):
            for p in msg.parts:
                if isinstance(p, messages_.TextPart):
                    parts.append(p.text)
    return " ".join(parts)


def extract_input_images(
    messages: list[messages_.Message],
) -> list[messages_.FilePart]:
    """Extract input images from user messages (for image editing)."""
    images: list[messages_.FilePart] = []
    for msg in messages:
        if msg.role == "user":
            for p in msg.parts:
                if isinstance(p, messages_.FilePart) and p.media_type.startswith(
                    "image/"
                ):
                    images.append(p)
    return images


class ImageModel(abc.ABC):
    """Abstract image generation model.

    Accepts :class:`Message`\\s as input and returns a :class:`Message`
    containing generated images as :class:`FilePart`\\s.
    """

    @abc.abstractmethod
    async def generate(
        self,
        messages: list[messages_.Message],
        *,
        n: int = 1,
        size: str | None = None,
        aspect_ratio: str | None = None,
        seed: int | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> messages_.Message:
        """Generate images from the given messages.

        Args:
            messages: Input messages containing the text prompt and optional
                input images (as :class:`FilePart`\\s) for editing.
            n: Number of images to generate.
            size: Image dimensions (e.g. ``"1024x1024"``).
            aspect_ratio: Aspect ratio (e.g. ``"16:9"``).
            seed: Random seed for reproducible generation.
            provider_options: Provider-specific options.

        Returns:
            A :class:`Message` with ``role="assistant"`` containing
            :class:`FilePart`\\s for each generated image.
        """
        ...
