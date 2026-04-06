"""Composable message construction helpers.

Convenience functions for building Message objects without manually
constructing Part lists.  Each ``*_message`` function accepts a mix of
plain strings (auto-wrapped in :class:`TextPart`) and existing
:class:`Part` objects, returning a single :class:`Message`.
"""

from __future__ import annotations

from .messages import (
    FilePart,
    HookPart,
    Message,
    Part,
    ReasoningPart,
    StructuredOutputPart,
    TextPart,
    ToolPart,
)

_PART_TYPES = (
    TextPart,
    ToolPart,
    ReasoningPart,
    HookPart,
    StructuredOutputPart,
    FilePart,
)

# A value that can appear as message content: bare strings become TextPart.
PartLike = str | Part


def _coerce_parts(args: tuple[PartLike, ...]) -> list[Part]:
    parts: list[Part] = []
    for arg in args:
        if isinstance(arg, str):
            parts.append(TextPart(text=arg))
        elif isinstance(arg, _PART_TYPES):
            parts.append(arg)
        else:
            raise TypeError(f"Expected str or Part, got {type(arg).__name__}")
    return parts


def system_message(*content: PartLike) -> Message:
    """Create a system message.

    >>> ai.system_message("You are a helpful robot.")
    """
    return Message(role="system", parts=_coerce_parts(content))


def user_message(*content: PartLike) -> Message:
    """Create a user message from strings and/or Part objects.

    >>> ai.user_message("Describe this image:", ai.file_part(url))
    """
    return Message(role="user", parts=_coerce_parts(content))


def assistant_message(*content: PartLike) -> Message:
    """Create an assistant message from strings and/or Part objects.

    >>> ai.assistant_message(ai.thinking("hmm"), "Hello!")
    """
    return Message(role="assistant", parts=_coerce_parts(content))


def file_part(
    data: str | bytes,
    *,
    media_type: str | None = None,
    filename: str | None = None,
) -> FilePart:
    """Create a :class:`FilePart` from a URL string or raw bytes.

    Dispatches to :meth:`FilePart.from_url` (for ``str``) or
    :meth:`FilePart.from_bytes` (for ``bytes``), with automatic
    media-type detection.
    """
    if isinstance(data, str):
        return FilePart.from_url(data, media_type=media_type)
    return FilePart.from_bytes(data, media_type=media_type, filename=filename)


def thinking(text: str, *, signature: str | None = None) -> ReasoningPart:
    """Create a :class:`ReasoningPart`.

    Useful for replaying conversation history that includes model reasoning.
    """
    return ReasoningPart(text=text, signature=signature)
