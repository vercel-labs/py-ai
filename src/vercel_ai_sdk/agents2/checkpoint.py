from __future__ import annotations

from typing import Any

import pydantic

from ..types import messages as messages_
from . import streams as streams_


class StepEvent(pydantic.BaseModel):
    """A completed @stream step."""

    index: int
    messages: list[messages_.Message]

    def to_stream_result(self) -> streams_.StreamResult:
        return streams_.StreamResult(messages=list(self.messages))


class ToolEvent(pydantic.BaseModel):
    """A completed tool execution."""

    tool_call_id: str
    result: Any
    status: str = "result"  # "result" | "error"


class HookEvent(pydantic.BaseModel):
    """A resolved hook."""

    label: str
    resolution: dict[str, Any]


class PendingHookInfo(pydantic.BaseModel):
    """A hook that was suspended but not resolved when the run ended."""

    label: str
    hook_type: str
    metadata: dict[str, Any] = {}


class Checkpoint(pydantic.BaseModel):
    steps: list[StepEvent] = []
    tools: list[ToolEvent] = []
    hooks: list[HookEvent] = []
    pending_hooks: list[PendingHookInfo] = []
