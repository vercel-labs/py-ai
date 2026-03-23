"""Telemetry for tracing agent runs.

Enable the built-in OpenTelemetry handler::

    import vercel_ai_sdk as ai

    ai.telemetry.enable()

Or bring your own handler::

    class MyHandler:
        def handle(self, event: ai.telemetry.TelemetryEvent) -> None:
            match event:
                case ai.telemetry.ToolCallFinishEvent(tool_name=name):
                    print(f"tool {name} done")

    ai.telemetry.enable(MyHandler())

Emit custom events from user code::

    ai.telemetry.handle(my_custom_event)
"""

from __future__ import annotations

import contextvars
import dataclasses
import time
import uuid
from typing import Any, Protocol, runtime_checkable

from ..agents import streams as streams_
from ..types import messages as messages_

# ── Protocol ───────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """Base class for all telemetry events."""

    pass


@dataclasses.dataclass(frozen=True, slots=True)
class RunStartEvent(TelemetryEvent):
    """Emitted when ``ai.run()`` begins executing."""

    pass


@dataclasses.dataclass(frozen=True, slots=True)
class RunFinishEvent(TelemetryEvent):
    """Emitted when ``ai.run()`` completes (success or error)."""

    usage: messages_.Usage | None
    error: BaseException | None


@dataclasses.dataclass(frozen=True, slots=True)
class StepStartEvent(TelemetryEvent):
    """Emitted when a ``@stream``-decorated step begins processing."""

    step_index: int


@dataclasses.dataclass(frozen=True, slots=True)
class StepFinishEvent(TelemetryEvent):
    """Emitted when a ``@stream``-decorated step finishes."""

    step_index: int
    result: streams_.StreamResult


@dataclasses.dataclass(frozen=True, slots=True)
class ToolCallStartEvent(TelemetryEvent):
    """Emitted before a tool is executed."""

    tool_name: str
    tool_call_id: str
    args: str  # raw JSON


@dataclasses.dataclass(frozen=True, slots=True)
class ToolCallFinishEvent(TelemetryEvent):
    """Emitted after a tool execution completes."""

    tool_name: str
    tool_call_id: str
    result: Any
    error: str | None
    duration_ms: float


@runtime_checkable
class Handler(Protocol):
    """Protocol for telemetry integrations.

    Implement a single ``handle`` method and use ``match`` to dispatch
    on event types you care about.  Unknown events are silently ignored.
    """

    def handle(self, event: TelemetryEvent) -> None: ...


# ── Internal state ────────────────────────────────────────────────


class _NoopHandler:
    __slots__ = ()

    def handle(self, event: TelemetryEvent) -> None:
        pass


_NOOP = _NoopHandler()

_handler: contextvars.ContextVar[Handler] = contextvars.ContextVar(
    "telemetry_handler", default=_NOOP
)

_run_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "telemetry_run_id", default=""
)


# ── Public API ────────────────────────────────────────────────────


def enable(handler: Handler | None = None) -> None:
    """Enable telemetry.

    - Called with no arguments: use the built-in OTel handler (requires
      ``opentelemetry-api``; raises ``ImportError`` if not installed).
    - Called with a handler: use that handler.
    """
    if handler is None:
        from .otel import OtelHandler

        _handler.set(OtelHandler())
    else:
        _handler.set(handler)


def disable() -> None:
    """Revert to the no-op handler."""
    _handler.set(_NOOP)


def handle(event: TelemetryEvent) -> None:
    """Dispatch an event to the active handler."""
    _handler.get().handle(event)


def get_run_id() -> str:
    """Return the current run ID (empty string outside a run)."""
    return _run_id.get()


def start_run() -> contextvars.Token[str]:
    """Generate a new run ID and set it in the context. Returns the token for reset."""
    return _run_id.set(uuid.uuid4().hex[:16])


def end_run(token: contextvars.Token[str]) -> None:
    """Reset the run ID context."""
    _run_id.reset(token)


def time_ms() -> float:
    """Monotonic timestamp in milliseconds."""
    return time.monotonic() * 1000
