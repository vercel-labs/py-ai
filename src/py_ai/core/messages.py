from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal

import pydantic


class TextPart(pydantic.BaseModel):
    text: str
    type: Literal["text"] = "text"


class ToolPart(pydantic.BaseModel):
    tool_call_id: str
    tool_name: str
    tool_args: str
    status: Literal["pending", "result"] = "pending"
    result: dict[str, Any] | None = None
    type: Literal["tool"] = "tool"


class ReasoningPart(pydantic.BaseModel):
    reasoning: str
    type: Literal["reasoning"] = "reasoning"
    # Anthropic's thinking blocks include a signature for cache/verification.
    # This must be preserved and sent back in multi-turn conversations.
    signature: str | None = None


Part = Annotated[
    TextPart | ToolPart | ReasoningPart,
    pydantic.Field(discriminator="type"),
]


def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


class ToolDelta(pydantic.BaseModel):
    tool_call_id: str
    tool_name: str
    args_delta: str


class Message(pydantic.BaseModel):
    role: Literal["user", "assistant", "system"]
    parts: list[Part]
    id: str = pydantic.Field(default_factory=_gen_id)
    is_done: bool = False
    text_delta: str = ""
    reasoning_delta: str = ""
    tool_deltas: list[ToolDelta] = pydantic.Field(default_factory=list)
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

    def get_tool_part(self, tool_call_id: str) -> ToolPart | None:
        for part in self.parts:
            if isinstance(part, ToolPart) and part.tool_call_id == tool_call_id:
                return part
        return None
