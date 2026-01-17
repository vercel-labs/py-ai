from __future__ import annotations

import dataclasses
import uuid
from typing import Any, Literal


@dataclasses.dataclass
class TextPart:
    text: str
    type: Literal["text"] = "text"


@dataclasses.dataclass
class ToolCallPart:
    tool_call_id: str
    tool_name: str
    tool_args: str
    type: Literal["tool_call"] = "tool_call"


@dataclasses.dataclass
class ToolResultPart:
    tool_call_id: str
    result: dict[str, Any]
    type: Literal["tool_result"] = "tool_result"


@dataclasses.dataclass
class ReasoningPart:
    reasoning: str
    type: Literal["reasoning"] = "reasoning"
    # Anthropic's thinking blocks include a signature for cache/verification.
    # This must be preserved and sent back in multi-turn conversations.
    signature: str | None = None


Part = TextPart | ToolCallPart | ToolResultPart | ReasoningPart


def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclasses.dataclass
class ToolCallDelta:
    """Represents a streaming delta for tool call arguments."""

    tool_call_id: str
    tool_name: str
    args_delta: str


@dataclasses.dataclass
class Message:
    role: Literal["user", "assistant", "system", "tool"]
    parts: list[Part]
    id: str = dataclasses.field(default_factory=_gen_id)
    is_done: bool = False
    text_delta: str = ""
    reasoning_delta: str = ""
    tool_call_deltas: list[ToolCallDelta] = dataclasses.field(default_factory=list)
    label: str | None = None

    @property
    def text(self) -> str:
        for part in self.parts:
            if isinstance(part, TextPart):
                return part.text
        return ""

    @property
    def reasoning(self) -> str:
        for part in self.parts:
            if isinstance(part, ReasoningPart):
                return part.reasoning
        return ""
