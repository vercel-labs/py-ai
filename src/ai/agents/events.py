"""Agent-layer event types.

The model layer emits ``StreamStart`` / ``StreamEnd`` plus block-level
deltas.  The agent layer adds ``ToolCallResult`` (tool execution outcomes)
and ``HookEvent`` (human-in-the-loop suspension points).

These types live here (rather than in ``ai.types.events``) because they
are an agent-runtime concern, not part of the public model-streaming
event vocabulary.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import pydantic

from .. import types


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


AgentEvent = types.Event | ToolCallResult | HookEvent

TerminalEvent = types.StreamEnd | ToolCallResult | HookEvent


__all__ = [
    "AgentEvent",
    "HookEvent",
    "TerminalEvent",
    "ToolCallResult",
]
