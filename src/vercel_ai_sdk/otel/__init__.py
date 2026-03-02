"""OpenTelemetry telemetry handler for the AI SDK.

Maps agent lifecycle events to OTel spans following the ``gen_ai.*``
semantic conventions.

Requires the ``opentelemetry-api`` package::

    pip install opentelemetry-api

Usage::

    import vercel_ai_sdk as ai

    # Uses OtelHandler by default
    ai.telemetry.enable()

    # Or configure explicitly
    from vercel_ai_sdk.otel import OtelHandler
    ai.telemetry.enable(OtelHandler(record_inputs=False))
"""

from __future__ import annotations

import json
from typing import Any

from opentelemetry import context, trace
from opentelemetry.trace import Span, StatusCode, Tracer, TracerProvider

from ..core.telemetry import (
    RunFinishEvent,
    RunStartEvent,
    StepFinishEvent,
    StepStartEvent,
    TelemetryEvent,
    ToolCallFinishEvent,
    ToolCallStartEvent,
    get_run_id,
)

_SCOPE_NAME = "vercel-ai-sdk"
_SCOPE_VERSION = "0.1.0"


def _select_attributes(
    attrs: dict[str, Any],
    *,
    record_inputs: bool,
    record_outputs: bool,
) -> dict[str, Any]:
    """Filter attribute dict respecting privacy flags.

    Values may be plain values, ``{"input": callable}`` or
    ``{"output": callable}`` dicts.  Callables are only evaluated when
    the corresponding flag is True.
    """
    result: dict[str, Any] = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, dict):
            if "input" in value:
                if not record_inputs:
                    continue
                resolved = (
                    value["input"]() if callable(value["input"]) else value["input"]
                )
                if resolved is not None:
                    result[key] = resolved
                continue
            if "output" in value:
                if not record_outputs:
                    continue
                resolved = (
                    value["output"]() if callable(value["output"]) else value["output"]
                )
                if resolved is not None:
                    result[key] = resolved
                continue
        result[key] = value
    return result


class OtelHandler:
    """OpenTelemetry implementation of :class:`TelemetryHandler`.

    Creates spans following the ``gen_ai.*`` semantic conventions:

    - ``ai.run`` — root span for the entire ``ai.run()`` call
    - ``ai.stream`` — child span per ``@stream`` step
    - ``ai.tool`` — child span per tool execution

    Parameters
    ----------
    tracer_provider:
        Custom ``TracerProvider``.  Falls back to the global provider.
    record_inputs:
        Record prompt messages and tool arguments on spans.
    record_outputs:
        Record completions and tool results on spans.
    function_id:
        Logical identifier for grouping traces.
    metadata:
        Custom key-value pairs added to all spans.
    """

    def __init__(
        self,
        *,
        tracer_provider: TracerProvider | None = None,
        record_inputs: bool = True,
        record_outputs: bool = True,
        function_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if tracer_provider is not None:
            self._tracer: Tracer = tracer_provider.get_tracer(
                _SCOPE_NAME, _SCOPE_VERSION
            )
        else:
            self._tracer = trace.get_tracer(_SCOPE_NAME, _SCOPE_VERSION)

        self._record_inputs = record_inputs
        self._record_outputs = record_outputs
        self._function_id = function_id
        self._metadata = metadata or {}

        # Span state (per-run, managed by lifecycle events)
        self._root_span: Span | None = None
        self._root_context: Any = None
        self._step_spans: dict[int, tuple[Span, Any]] = {}
        self._tool_spans: dict[str, Span] = {}

    # ── helpers ────────────────────────────────────────────────────

    def _base_attributes(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        run_id = get_run_id()
        if run_id:
            attrs["ai.run.id"] = run_id
        if self._function_id:
            attrs["ai.telemetry.functionId"] = self._function_id
        for k, v in self._metadata.items():
            attrs[f"ai.telemetry.metadata.{k}"] = v
        return attrs

    def _select(self, attrs: dict[str, Any]) -> dict[str, Any]:
        return _select_attributes(
            attrs,
            record_inputs=self._record_inputs,
            record_outputs=self._record_outputs,
        )

    # ── single dispatch ───────────────────────────────────────────

    def handle(self, event: TelemetryEvent) -> None:
        match event:
            case RunStartEvent():
                self._on_run_start(event)
            case RunFinishEvent():
                self._on_run_finish(event)
            case StepStartEvent():
                self._on_step_start(event)
            case StepFinishEvent():
                self._on_step_finish(event)
            case ToolCallStartEvent():
                self._on_tool_call_start(event)
            case ToolCallFinishEvent():
                self._on_tool_call_finish(event)

    # ── run lifecycle ─────────────────────────────────────────────

    def _on_run_start(self, event: RunStartEvent) -> None:
        attrs = self._select(
            {
                **self._base_attributes(),
                "ai.operationId": "ai.run",
            }
        )
        self._root_span = self._tracer.start_span("ai.run", attributes=attrs)
        self._root_context = trace.set_span_in_context(
            self._root_span, context.get_current()
        )

    def _on_run_finish(self, event: RunFinishEvent) -> None:
        span = self._root_span
        if span is None:
            return

        attrs: dict[str, Any] = {}
        if event.usage:
            attrs["gen_ai.usage.input_tokens"] = event.usage.input_tokens
            attrs["gen_ai.usage.output_tokens"] = event.usage.output_tokens

        if attrs:
            span.set_attributes(self._select(attrs))

        if event.error is not None:
            span.set_status(StatusCode.ERROR, str(event.error))
            if isinstance(event.error, Exception):
                span.record_exception(event.error)

        span.end()
        self._root_span = None
        self._root_context = None

    # ── step lifecycle ────────────────────────────────────────────

    def _on_step_start(self, event: StepStartEvent) -> None:
        parent_ctx = self._root_context
        attrs = self._select(
            {
                **self._base_attributes(),
                "ai.operationId": "ai.stream",
                "ai.step.index": event.step_index,
            }
        )
        span = self._tracer.start_span(
            "ai.stream",
            attributes=attrs,
            context=parent_ctx,
        )
        ctx = trace.set_span_in_context(
            span,
            parent_ctx if parent_ctx is not None else context.get_current(),
        )
        self._step_spans[event.step_index] = (span, ctx)

    def _on_step_finish(self, event: StepFinishEvent) -> None:
        entry = self._step_spans.pop(event.step_index, None)
        if entry is None:
            return
        span, _ctx = entry

        attrs: dict[str, Any] = {}
        usage = event.result.usage
        if usage:
            attrs["gen_ai.usage.input_tokens"] = usage.input_tokens
            attrs["gen_ai.usage.output_tokens"] = usage.output_tokens

        text = event.result.text
        if text:
            attrs["ai.response.text"] = {"output": lambda: text}

        finish_reason = None
        if event.result.last_message and not event.result.tool_calls:
            finish_reason = "stop"
        elif event.result.tool_calls:
            finish_reason = "tool_calls"
        if finish_reason:
            attrs["ai.response.finishReason"] = finish_reason
            attrs["gen_ai.response.finish_reasons"] = [finish_reason]

        if attrs:
            span.set_attributes(self._select(attrs))

        span.end()

    # ── tool lifecycle ────────────────────────────────────────────

    def _on_tool_call_start(self, event: ToolCallStartEvent) -> None:
        # Find parent step context (use the most recent active step)
        parent_ctx = None
        if self._step_spans:
            latest_idx = max(self._step_spans)
            _, parent_ctx = self._step_spans[latest_idx]
        if parent_ctx is None:
            parent_ctx = self._root_context

        attrs = self._select(
            {
                "ai.operationId": "ai.tool",
                "ai.toolCall.name": event.tool_name,
                "ai.toolCall.id": event.tool_call_id,
                "gen_ai.tool.name": event.tool_name,
                "gen_ai.tool.call.id": event.tool_call_id,
                "ai.toolCall.args": {"input": lambda: event.args},
                "gen_ai.tool.call.arguments": {"input": lambda: event.args},
            }
        )
        span = self._tracer.start_span(
            "ai.tool",
            attributes=attrs,
            context=parent_ctx,
        )
        self._tool_spans[event.tool_call_id] = span

    def _on_tool_call_finish(self, event: ToolCallFinishEvent) -> None:
        span = self._tool_spans.pop(event.tool_call_id, None)
        if span is None:
            return

        attrs: dict[str, Any] = {
            "ai.toolCall.durationMs": event.duration_ms,
        }

        if event.error is not None:
            span.set_status(StatusCode.ERROR, event.error)
        else:
            try:
                result_str = json.dumps(event.result)
            except (TypeError, ValueError):
                result_str = str(event.result)
            attrs["ai.toolCall.result"] = {"output": lambda: result_str}
            attrs["gen_ai.tool.call.result"] = {"output": lambda: result_str}

        span.set_attributes(self._select(attrs))
        span.end()


__all__ = ["OtelHandler"]
