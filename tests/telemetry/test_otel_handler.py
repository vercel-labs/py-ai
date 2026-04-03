"""OTel handler: span names, hierarchy, and attributes."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import vercel_ai_sdk as ai
from vercel_ai_sdk.telemetry.otel import OtelHandler

from ..conftest import MOCK_MODEL, mock_llm, text_msg, tool_msg


@pytest.fixture
def spans() -> Generator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    ai.telemetry.enable(OtelHandler(tracer_provider=provider))
    yield exporter
    ai.telemetry.disable()
    provider.shutdown()


@ai.tool
async def double(x: int) -> int:
    """Double a number."""
    return x * 2


@pytest.mark.asyncio
async def test_text_only_spans(spans: InMemorySpanExporter) -> None:
    """Text-only run produces ai.run > ai.stream span hierarchy."""

    async def root(model: ai.Model) -> ai.StreamResult:
        return await ai.stream_loop(
            model, messages=ai.make_messages(user="Hi"), tools=[]
        )

    mock_llm([[text_msg("Hello!")]])
    result = ai.run(root, MOCK_MODEL)
    [m async for m in result]

    finished = spans.get_finished_spans()
    names = [s.name for s in finished]
    assert "ai.run" in names
    assert "ai.stream" in names

    run_span = next(s for s in finished if s.name == "ai.run")
    stream_span = next(s for s in finished if s.name == "ai.stream")

    # ai.stream is a child of ai.run
    assert stream_span.parent is not None
    assert stream_span.parent.span_id == run_span.context.span_id

    # run_id attribute is set
    assert run_span.attributes is not None
    assert run_span.attributes.get("ai.run.id") != ""


@pytest.mark.asyncio
async def test_tool_call_spans(spans: InMemorySpanExporter) -> None:
    """Tool-calling run produces ai.tool spans with correct attributes."""

    async def root(model: ai.Model) -> ai.StreamResult:
        return await ai.stream_loop(
            model, messages=ai.make_messages(user="Double 5"), tools=[double]
        )

    mock_llm(
        [
            [tool_msg(tc_id="tc-1", name="double", args='{"x": 5}')],
            [text_msg("10")],
        ]
    )
    result = ai.run(root, MOCK_MODEL)
    [m async for m in result]

    finished = spans.get_finished_spans()
    names = [s.name for s in finished]
    assert names.count("ai.stream") == 2
    assert "ai.tool" in names

    tool_span = next(s for s in finished if s.name == "ai.tool")
    assert tool_span.attributes is not None
    assert tool_span.attributes.get("gen_ai.tool.name") == "double"
    assert tool_span.attributes.get("gen_ai.tool.call.id") == "tc-1"

    # ai.tool is a child of ai.run (tools execute between steps)
    run_span = next(s for s in finished if s.name == "ai.run")
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == run_span.context.span_id
