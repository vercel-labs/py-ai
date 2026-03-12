"""Base class for media generation models (image, video).

Provides shared input-parsing (extract prompt text and input files from
messages) and output-assembly (wrap adapter results into a :class:`Message`).

Subclasses :class:`ImageModel` and :class:`VideoModel` define the concrete
``generate()`` / ``make_request()`` signatures with media-type-specific
parameters.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any

from . import messages as messages_


@dataclasses.dataclass
class MediaResult:
    """Raw result returned by an adapter's ``make_request()`` method.

    The framework wraps this into a :class:`Message` automatically.
    """

    files: list[messages_.FilePart]
    usage: messages_.Usage | None = None


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
