from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Any, cast

import pydantic
import pytest

import ai
from ai import models
from ai.models.openai import openai
from ai.types import events as events_
from ai.types import messages as messages_

from ...conftest import MOCK_MODEL, MOCK_PROVIDER, MockProvider, mock_llm, text_msg


class _MockStreamParams(pydantic.BaseModel):
    value: str


async def test_stream_aggregates_registered_adapter_events() -> None:
    mock = mock_llm([[text_msg("Hello world")]])

    stream = models.stream(MOCK_MODEL, [ai.user_message("Hi")])
    deltas: list[str] = []
    async for event in stream:
        if isinstance(event, events_.TextDelta):
            deltas.append(event.chunk)

    assert mock.call_count == 1
    assert stream.text == "Hello world"
    assert "".join(deltas) == "Hello world"


async def test_stream_tool_end_includes_aggregated_tool_call() -> None:
    async def _tool_stream(
        client: models.Client,
        model: models.Model[pydantic.BaseModel],
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[events_.Event]:
        yield events_.StreamStart()
        yield events_.ToolStart(tool_call_id="tc-1", tool_name="weather")
        yield events_.ToolDelta(tool_call_id="tc-1", chunk='{"city"')
        yield events_.ToolDelta(tool_call_id="tc-1", chunk=':"SF"}')
        yield events_.ToolEnd(
            tool_call_id="tc-1",
            tool_call=messages_.DUMMY_TOOL_CALL,
        )
        yield events_.StreamEnd()

    models.register_stream("mock", _tool_stream)

    stream = models.stream(MOCK_MODEL, [ai.user_message("Check weather")])
    tool_end: events_.ToolEnd | None = None
    async for event in stream:
        if isinstance(event, events_.ToolEnd):
            tool_end = event

    assert tool_end is not None
    assert tool_end.tool_call.tool_call_id == "tc-1"
    assert tool_end.tool_call.tool_name == "weather"
    assert tool_end.tool_call.tool_args == '{"city":"SF"}'
    assert stream.tool_calls == [tool_end.tool_call]


async def test_stream_uses_explicit_model_client() -> None:
    received_clients: list[models.Client] = []

    async def _spy_stream(
        client: models.Client,
        model: models.Model[pydantic.BaseModel],
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[events_.Event]:
        received_clients.append(client)
        yield events_.StreamStart()
        yield events_.StreamEnd()

    models.register_stream("mock", _spy_stream)

    explicit = models.Client(base_url="https://custom.test", api_key="sk-custom")
    model = models.Model[pydantic.BaseModel](
        id="mock-model",
        adapter="mock",
        provider=MOCK_PROVIDER,
        client=explicit,
    )
    stream = models.stream(model, [ai.user_message("Hi")])
    async for _ in stream:
        pass

    assert received_clients == [explicit]


async def test_stream_forwards_output_type_and_request_params() -> None:
    received_output_types: list[type[pydantic.BaseModel] | None] = []
    received_params: list[Any] = []

    class Answer(pydantic.BaseModel):
        value: str

    async def _spy_stream(
        client: models.Client,
        model: models.Model[pydantic.BaseModel],
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[events_.Event]:
        received_output_types.append(output_type)
        received_params.append(kwargs.get("params"))
        yield events_.StreamStart()
        yield events_.StreamEnd()

    models.register_stream("mock", _spy_stream)

    params = _MockStreamParams(value="ok")
    stream = models.stream(
        MOCK_MODEL,
        [ai.user_message("Hi")],
        output_type=Answer,
        params=params,
    )
    async for _ in stream:
        pass

    assert received_output_types == [Answer]
    assert received_params == [params]


def test_normalize_params_rejects_non_pydantic_value() -> None:
    """``stream(...)`` rejects raw dicts (and anything not a BaseModel)."""
    with pytest.raises(TypeError, match="pydantic BaseModel"):
        models.stream(
            openai("gpt-5.4"),
            [ai.user_message("Hi")],
            params=cast(Any, {"reasoning_effort": "high"}),
        )


async def test_generate_dispatches_to_registered_adapter() -> None:
    provider = MockProvider(adapter="mock-generate")
    model = models.Model[pydantic.BaseModel](
        id="generate-model",
        adapter="mock-generate",
        provider=provider,
    )
    sentinel = messages_.Message(
        role="assistant",
        parts=[messages_.FilePart(data=b"\x89PNG", media_type="image/png")],
    )
    called = False

    async def _generate(
        client: models.Client,
        model: models.Model[pydantic.BaseModel],
        messages: list[messages_.Message],
        params: Any = None,
    ) -> messages_.Message:
        nonlocal called
        called = True
        return sentinel

    models.register_generate("mock-generate", _generate)

    result = await models.generate(
        model,
        [ai.user_message("A cat")],
        models.ImageParams(n=1),
    )

    assert called
    assert result is sentinel


class _CheckProvider(MockProvider):
    def __init__(self) -> None:
        super().__init__(adapter="mock-check")
        self.received_client: models.Client | None = None

    async def check(
        self,
        client: models.Client,
        model: models.Model[pydantic.BaseModel],
    ) -> bool:
        self.received_client = client
        return False


async def test_check_connection_delegates_to_model_provider() -> None:
    provider = _CheckProvider()
    explicit = models.Client(base_url="https://check.test", api_key="sk-check")
    model = provider("mock-model", client=explicit)

    result = await models.check_connection(model)

    assert result is False
    assert provider.received_client is explicit


async def test_stream_replays_last_assistant_with_tool_calls() -> None:
    """If the last message is an assistant turn with tool calls, no LLM call."""
    called = False

    async def _spy_stream(
        client: models.Client,
        model: models.Model[pydantic.BaseModel],
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[events_.Event]:
        nonlocal called
        called = True
        yield events_.StreamStart()
        yield events_.StreamEnd()

    models.register_stream("mock", _spy_stream)

    assistant_msg = messages_.Message(
        role="assistant",
        parts=[
            messages_.TextPart(id="t1", text="calling tools"),
            messages_.ToolCallPart(
                tool_call_id="tc-1",
                tool_name="weather",
                tool_args='{"city":"SF"}',
            ),
        ],
    )

    stream = models.stream(
        MOCK_MODEL,
        [ai.user_message("Hi"), assistant_msg],
    )
    events: list[events_.Event] = [event async for event in stream]

    assert called is False, "should not have hit the LLM"
    # Stream.message is seeded from the original turn — text and tool
    # calls are both preserved.
    assert stream.text == "calling tools"
    assert len(stream.tool_calls) == 1
    assert stream.tool_calls[0].tool_call_id == "tc-1"
    assert stream.tool_calls[0].tool_args == '{"city":"SF"}'
    # Replay only emits ToolEnd events, flagged for agent.run to drop.
    tool_ends = [e for e in events if isinstance(e, events_.ToolEnd)]
    assert len(tool_ends) == 1
    assert tool_ends[0].replay is True
    assert tool_ends[0].tool_call.tool_call_id == "tc-1"
    # No ToolStart/Delta/text events are re-emitted.
    assert not any(isinstance(e, events_.ToolStart) for e in events)
    assert not any(isinstance(e, events_.TextDelta) for e in events)


def test_tool_end_replay_flag_excluded_from_json() -> None:
    """The replay flag is internal — it must not appear in serialized output."""
    ev = events_.ToolEnd(
        tool_call_id="tc-1",
        tool_call=messages_.DUMMY_TOOL_CALL,
        replay=True,
    )
    dumped = ev.model_dump()
    assert "replay" not in dumped
    dumped_json = ev.model_dump(mode="json")
    assert "replay" not in dumped_json


async def test_stream_does_not_replay_when_assistant_has_no_tool_calls() -> None:
    """Bare assistant text doesn't trigger replay — fall through to LLM."""
    called = False

    async def _spy_stream(
        client: models.Client,
        model: models.Model[pydantic.BaseModel],
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[events_.Event]:
        nonlocal called
        called = True
        yield events_.StreamStart()
        yield events_.StreamEnd()

    models.register_stream("mock", _spy_stream)

    assistant_text_only = messages_.Message(
        role="assistant",
        parts=[messages_.TextPart(text="just talking")],
    )

    stream = models.stream(MOCK_MODEL, [assistant_text_only])
    async for _ in stream:
        pass

    assert called is True
