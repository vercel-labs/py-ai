"""Shared transport types — the universal interchange format.

Message, Part types, Usage, and tool schema protocols used across
both the models and agents layers.
"""

from .messages import (
    FilePart,
    HookPart,
    Message,
    Part,
    PartClosed,
    PartDelta,
    PartOpened,
    ReasoningPart,
    StreamState,
    StructuredOutputPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    Usage,
    generate_id,
)
from .stream import StreamResultLike
from .tools import ToolLike, ToolSchema

__all__ = [
    "FilePart",
    "HookPart",
    "Message",
    "Part",
    "PartClosed",
    "PartDelta",
    "PartOpened",
    "ReasoningPart",
    "StreamResultLike",
    "StreamState",
    "StructuredOutputPart",
    "TextPart",
    "ToolCallPart",
    "ToolLike",
    "ToolResultPart",
    "ToolSchema",
    "Usage",
    "generate_id",
]
