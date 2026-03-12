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
from typing import Any, override

from . import media_model as media_model_
from . import messages as messages_


class ImageModel(media_model_.MediaModel):
    """Abstract image generation model.

    Accepts :class:`Message`\\s as input and returns a :class:`Message`
    containing generated images as :class:`FilePart`\\s.

    Adapter authors implement :meth:`make_request`; the framework handles
    parsing messages and assembling the response :class:`Message`.
    """

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
        prompt = self._extract_prompt(messages)
        input_files = self._extract_input_files(messages)
        result = await self.make_request(
            prompt,
            input_files,
            n=n,
            size=size,
            aspect_ratio=aspect_ratio,
            seed=seed,
            provider_options=provider_options,
        )
        return self._build_message(result)

    @override
    @abc.abstractmethod
    async def make_request(
        self,
        prompt: str,
        input_files: list[messages_.FilePart],
        *,
        n: int = 1,
        size: str | None = None,
        aspect_ratio: str | None = None,
        seed: int | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> media_model_.MediaResult:
        """Adapter-specific image generation.

        Args:
            prompt: Text prompt extracted from messages.
            input_files: File parts from user messages (for editing).
            n: Number of images to generate.
            size: Image dimensions (e.g. ``"1024x1024"``).
            aspect_ratio: Aspect ratio (e.g. ``"16:9"``).
            seed: Random seed for reproducible generation.
            provider_options: Provider-specific options.

        Returns:
            A :class:`MediaResult` with generated image files.
        """
        ...
