from __future__ import annotations

import dataclasses
from typing import Any

from . import messages as messages_
from . import streams as streams_


@dataclasses.dataclass
class StepEvent:
    """A completed @stream step."""

    index: int
    messages: list[dict[str, Any]]  # Message.model_dump() for each

    def to_stream_result(self) -> streams_.StreamResult:
        return streams_.StreamResult(
            messages=[messages_.Message.model_validate(m) for m in self.messages]
        )


@dataclasses.dataclass
class ToolEvent:
    """A completed tool execution."""

    tool_call_id: str
    result: Any


@dataclasses.dataclass
class HookEvent:
    """A resolved hook."""

    label: str
    resolution: dict[str, Any]


@dataclasses.dataclass
class Checkpoint:
    steps: list[StepEvent] = dataclasses.field(default_factory=list)
    tools: list[ToolEvent] = dataclasses.field(default_factory=list)
    hooks: list[HookEvent] = dataclasses.field(default_factory=list)

    def serialize(self) -> dict[str, Any]:
        return {
            "steps": [{"index": s.index, "messages": s.messages} for s in self.steps],
            "tools": [
                {"tool_call_id": t.tool_call_id, "result": t.result} for t in self.tools
            ],
            "hooks": [
                {"label": h.label, "resolution": h.resolution} for h in self.hooks
            ],
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> Checkpoint:
        return cls(
            steps=[StepEvent(**s) for s in data.get("steps", [])],
            tools=[ToolEvent(**t) for t in data.get("tools", [])],
            hooks=[HookEvent(**h) for h in data.get("hooks", [])],
        )
