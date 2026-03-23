"""ImageModel — abstract image generation model."""

from __future__ import annotations

import abc
from typing import Any, override

from ...types import messages as messages_
from .media.base import MediaModel, MediaResult


class ImageModel(MediaModel):
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
        """Generate images from the given messages."""
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
    ) -> MediaResult:
        """Adapter-specific image generation."""
        ...
