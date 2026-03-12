"""Media generation model abstractions (image, video).

Provides :class:`MediaModel` (shared base), :class:`ImageModel`, and
:class:`VideoModel` along with the :class:`MediaResult` return type.

The base class owns the common pipeline steps that every adapter would
otherwise duplicate:

* **Input** -- extract a text prompt and input files from messages.
* **Output** -- wrap the adapter's :class:`MediaResult` into a
  :class:`Message` with ``role="assistant"``.

Subclasses define the public ``generate()`` signature with
media-type-specific parameters and delegate to the adapter's
``make_request()`` method.

Usage::

    # Image model
    model = ai.ai_gateway.GatewayImageModel(
        model="google/imagen-4.0-generate-001",
    )
    msg = await model.generate(
        ai.make_messages(user="A sunset over Tokyo"),
        n=2, aspect_ratio="16:9",
    )
    for img in msg.images:
        print(img.media_type, len(img.data))

    # Video model
    model = ai.ai_gateway.GatewayVideoModel(
        model="google/veo-3.0-generate-001",
    )
    msg = await model.generate(
        ai.make_messages(user="A cat on a beach"),
        aspect_ratio="16:9", duration=5,
    )
    for vid in msg.videos:
        print(vid.media_type, len(vid.data))
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, override

from .. import messages as messages_

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MediaResult:
    """Raw result returned by an adapter's ``make_request()`` method.

    The framework wraps this into a :class:`Message` automatically.
    """

    files: list[messages_.FilePart]
    usage: messages_.Usage | None = None


# ---------------------------------------------------------------------------
# MediaModel -- shared base
# ---------------------------------------------------------------------------


class MediaModel(abc.ABC):
    """Abstract base for media generation models.

    Owns the shared pipeline steps that every adapter would otherwise
    duplicate:

    * **Input** -- extract a text prompt and input files from
      :class:`Message` objects.
    * **Output** -- wrap the adapter's :class:`MediaResult` into a
      :class:`Message` with ``role="assistant"``.

    Subclasses (:class:`ImageModel`, :class:`VideoModel`) define the
    public ``generate()`` signature with media-type-specific parameters
    and delegate to the adapter's ``make_request()`` method.
    """

    @staticmethod
    def _extract_prompt(messages: list[messages_.Message]) -> str:
        """Concatenate all :class:`TextPart` texts from user/system messages."""
        parts: list[str] = []
        for msg in messages:
            if msg.role in ("user", "system"):
                for p in msg.parts:
                    if isinstance(p, messages_.TextPart):
                        parts.append(p.text)
        return " ".join(parts)

    @staticmethod
    def _extract_input_files(
        messages: list[messages_.Message],
    ) -> list[messages_.FilePart]:
        """Collect all :class:`FilePart` objects from user messages."""
        files: list[messages_.FilePart] = []
        for msg in messages:
            if msg.role == "user":
                for p in msg.parts:
                    if isinstance(p, messages_.FilePart):
                        files.append(p)
        return files

    @staticmethod
    def _build_message(result: MediaResult) -> messages_.Message:
        """Wrap adapter output into a :class:`Message`."""
        return messages_.Message(
            role="assistant",
            parts=list(result.files),
            usage=result.usage,
        )

    @abc.abstractmethod
    async def make_request(
        self,
        prompt: str,
        input_files: list[messages_.FilePart],
        *,
        n: int = 1,
        provider_options: dict[str, Any] | None = None,
    ) -> MediaResult:
        """Adapter-specific generation logic.

        Receives already-parsed inputs and returns a :class:`MediaResult`.
        The framework calls this from ``generate()`` and wraps the result
        into a :class:`Message`.
        """
        ...


# ---------------------------------------------------------------------------
# ImageModel
# ---------------------------------------------------------------------------


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
        """Generate images from the given messages.

        Args:
            messages: Input messages containing the text prompt and
                optional input images (as :class:`FilePart`\\s) for
                editing.
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
    ) -> MediaResult:
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


# ---------------------------------------------------------------------------
# VideoModel
# ---------------------------------------------------------------------------


class VideoModel(MediaModel):
    """Abstract video generation model.

    Accepts :class:`Message`\\s as input and returns a :class:`Message`
    containing generated videos as :class:`FilePart`\\s.

    Adapter authors implement :meth:`make_request`; the framework handles
    parsing messages and assembling the response :class:`Message`.
    """

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
            messages: Input messages containing the text prompt and
                optional input image (as a :class:`FilePart`) for
                image-to-video.
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
        prompt = self._extract_prompt(messages)
        input_files = self._extract_input_files(messages)
        result = await self.make_request(
            prompt,
            input_files,
            n=n,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration=duration,
            fps=fps,
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
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        duration: float | None = None,
        fps: int | None = None,
        seed: int | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> MediaResult:
        """Adapter-specific video generation.

        Args:
            prompt: Text prompt extracted from messages.
            input_files: File parts from user messages (e.g. input
                image for image-to-video).
            n: Number of videos to generate.
            aspect_ratio: Aspect ratio (e.g. ``"16:9"``).
            resolution: Video resolution (e.g. ``"1920x1080"``).
            duration: Duration in seconds.
            fps: Frames per second.
            seed: Random seed for reproducible generation.
            provider_options: Provider-specific options.

        Returns:
            A :class:`MediaResult` with generated video files.
        """
        ...
