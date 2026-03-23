"""Cross-cutting observability — telemetry event types and handler API.

Usage::

    import vercel_ai_sdk as ai

    ai.telemetry.enable()          # OTel handler (default)
    ai.telemetry.enable(handler)   # custom handler
    ai.telemetry.disable()
"""

from .events import (
    Handler,
    RunFinishEvent,
    RunStartEvent,
    StepFinishEvent,
    StepStartEvent,
    TelemetryEvent,
    ToolCallFinishEvent,
    ToolCallStartEvent,
    disable,
    enable,
    end_run,
    get_run_id,
    handle,
    start_run,
    time_ms,
)
from .otel import OtelHandler

__all__ = [
    # API
    "enable",
    "disable",
    "handle",
    "get_run_id",
    "start_run",
    "end_run",
    "time_ms",
    # Protocol
    "Handler",
    "TelemetryEvent",
    # Events
    "RunStartEvent",
    "RunFinishEvent",
    "StepStartEvent",
    "StepFinishEvent",
    "ToolCallStartEvent",
    "ToolCallFinishEvent",
    # Handler
    "OtelHandler",
]
