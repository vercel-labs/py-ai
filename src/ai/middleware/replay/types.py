"""Persisted replay state for serverless agent re-entry."""

from __future__ import annotations

from typing import Any, Literal

import pydantic


class ReplayMetadata(pydantic.BaseModel):
    """Debug-friendly metadata about the recorded run."""

    model_id: str
    tool_names: list[str] = pydantic.Field(default_factory=list)
    input_message_count: int
    last_assistant_message_id: str | None = None


class RecordedModelStep(pydantic.BaseModel):
    """All streamed assistant snapshots produced by a single model call."""

    ordinal: int
    messages: list[dict[str, Any]] = pydantic.Field(default_factory=list)


class RecordedToolResult(pydantic.BaseModel):
    """The tool message returned by a single tool execution."""

    ordinal: int
    tool_call_id: str
    tool_name: str
    message: dict[str, Any]


class PendingHookInfo(pydantic.BaseModel):
    """A hook that interrupted execution and must be resumed later."""

    label: str
    hook_type: str
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)


class ReplayState(pydantic.BaseModel):
    """Persisted event log for a resumable session run."""

    version: Literal[1] = 1
    session_id: str
    fingerprint: str
    model_steps: list[RecordedModelStep] = pydantic.Field(default_factory=list)
    tool_results: list[RecordedToolResult] = pydantic.Field(default_factory=list)
    pending_hooks: list[PendingHookInfo] = pydantic.Field(default_factory=list)
    metadata: ReplayMetadata
