"""Agent-layer event types.

The model layer emits ``StreamStart`` / ``StreamEnd`` plus block-level
deltas.  The agent layer adds ``ToolCallResult`` (tool execution outcomes)
and ``HookEvent`` (human-in-the-loop suspension points).

These types live here (rather than in ``ai.types.events``) because they
are an agent-runtime concern, not part of the public model-streaming
event vocabulary.
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Any, Literal

import pydantic

from .. import types


class Aggregator[Item, Result, ModelResult]:
    @abc.abstractmethod
    def feed(self, item: Item) -> None: ...

    @abc.abstractmethod
    def snapshot(self) -> Result: ...

    @abc.abstractmethod
    def to_model_output(self) -> ModelResult: ...


class PartialToolCallResult(pydantic.BaseModel):
    """Emitted when tool calls or other yield_from callers yield values."""

    # id: str = pydantic.Field(default_factory=generate_id)
    tool_call_id: str | None = None
    tool_name: str | None = None
    label: object = None
    value: Any = None

    def key(self) -> object:
        return (self.tool_call_id, self.label)

    aggregator_factory: Callable[[], Aggregator[Any, Any, Any]] | None = pydantic.Field(
        default=None, exclude=True, repr=False
    )

    kind: Literal["partial_tool_call_result"] = "partial_tool_call_result"


class ToolCallResult(pydantic.BaseModel):
    """Emitted after tool calls execute — carries the result message."""

    message: types.Message
    results: Sequence[types.ToolResultPart]

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["tool_call_result"] = "tool_call_result"


class HookEvent(pydantic.BaseModel):
    """Emitted when a hook suspends, resolves, or is cancelled."""

    message: types.Message
    hook: types.HookPart[Any]

    kind: Literal["hook"] = "hook"


AgentMessageEvent = types.Event | ToolCallResult | HookEvent

AgentEvent = types.Event | ToolCallResult | HookEvent | PartialToolCallResult

TerminalEvent = types.StreamEnd | ToolCallResult | HookEvent


__all__ = [
    "AgentEvent",
    "HookEvent",
    "TerminalEvent",
    "ToolCallResult",
    "PartialToolCallResult",
]
