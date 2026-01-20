from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal

import pydantic


class TextPart(pydantic.BaseModel):
    text: str
    type: Literal["text"] = "text"


class ToolCallPart(pydantic.BaseModel):
    tool_call_id: str
    tool_name: str
    tool_args: str
    type: Literal["tool_call"] = "tool_call"


class ToolResultPart(pydantic.BaseModel):
    tool_call_id: str
    result: dict[str, Any]
    type: Literal["tool_result"] = "tool_result"


class ReasoningPart(pydantic.BaseModel):
    reasoning: str
    type: Literal["reasoning"] = "reasoning"
    # Anthropic's thinking blocks include a signature for cache/verification.
    # This must be preserved and sent back in multi-turn conversations.
    signature: str | None = None


Part = Annotated[
    TextPart | ToolCallPart | ToolResultPart | ReasoningPart,
    pydantic.Field(discriminator="type"),
]


def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


class ToolCallDelta(pydantic.BaseModel):
    tool_call_id: str
    tool_name: str
    args_delta: str


class Message(pydantic.BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    parts: list[Part]
    id: str = pydantic.Field(default_factory=_gen_id)
    is_done: bool = False
    text_delta: str = ""
    reasoning_delta: str = ""
    tool_call_deltas: list[ToolCallDelta] = pydantic.Field(default_factory=list)
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
