"""Public ai.stream() and ai.generate() wrappers — end-to-end through mock adapters."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pydantic

import ai
from ai import models
from ai.types import events as events_
from ai.types import messages as messages_

from ..conftest import (
    MOCK_MODEL,
    MOCK_PROVIDER,
    MockProvider,
    collect_messages,
    mock_llm,
    text_msg,
)


# Module-level model so StructuredOutputPart can resolve it by FQN.
class _Recipe(pydantic.BaseModel):
    name: str
    steps: list[str]


# ---------------------------------------------------------------------------
# stream() tests
# ---------------------------------------------------------------------------


async def test_stream_basic() -> None:
    """ai.models.stream() yields deltas and exposes .text after iteration."""
    mock = mock_llm([[text_msg("Hello world")]])

    s = models.stream(MOCK_MODEL, [ai.user_message("Hi")])
    deltas: list[str] = []
    async for event in s:
        if isinstance(event, events_.TextDelta):
            deltas.append(event.chunk)

    assert mock.call_count == 1
    assert s.text == "Hello world"
    assert "".join(deltas) == "Hello world"


async def test_stream_preserves_existing_turn_ids() -> None:
    """ai.stream() stamps only inputs without a turn_id; older turns survive."""
    mock = mock_llm([[text_msg("reply")]])

    old = ai.user_message("earlier")
    old = old.model_copy(update={"turn_id": "prev"})
    fresh = ai.user_message("latest")

    s = models.stream(MOCK_MODEL, [old, fresh])
    yielded = await collect_messages(s)

    assert mock.call_count == 1
    # First yielded is the old input — unchanged.
    assert yielded[0].turn_id == "prev"
    # Fresh input was stamped with the current turn's id.
    assert yielded[1].turn_id is not None
    assert yielded[1].turn_id != "prev"
    # Response shares the current turn id.
    response_ids = [m.turn_id for m in yielded if m.role == "assistant"]
    assert response_ids and all(tid == yielded[1].turn_id for tid in response_ids)


async def test_stream_accepts_explicit_turn_id() -> None:
    """Explicit turn_id kwarg is used verbatim."""
    mock_llm([[text_msg("ok")]])
    fresh = ai.user_message("hi")

    s = models.stream(MOCK_MODEL, [fresh], turn_id="custom-turn")
    yielded = await collect_messages(s)

    assert s.turn_id == "custom-turn"
    assert yielded[0].turn_id == "custom-turn"
    assert yielded[-1].turn_id == "custom-turn"


async def test_stream_with_explicit_client() -> None:
    """Model with explicit client= forwards it to the adapter."""
    received_clients: list[models.Client] = []

    async def _spy_stream(
        client: models.Client,
        model: models.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[events_.Event]:
        received_clients.append(client)
        msg = messages_.Message(
            id="m1",
            role="assistant",
            parts=[messages_.TextPart(text="ok")],
        )
        yield events_.MessageStart(message=msg.model_copy(update={"parts": []}))
        yield events_.MessageEnd(message=msg)

    models.register_stream("mock", _spy_stream)

    explicit = models.Client(base_url="https://custom.test", api_key="sk-custom")
    explicit_model = models.Model(
        id="mock-model", adapter="mock", provider=MOCK_PROVIDER, client=explicit
    )
    s = models.stream(explicit_model, [ai.user_message("Hi")])
    async for _ in s:
        pass

    assert len(received_clients) == 1
    assert received_clients[0] is explicit


async def test_stream_with_output_type() -> None:
    """ai.models.stream(output_type=...) exposes .output after iteration."""
    json_text = '{"name": "Pancakes", "steps": ["Mix", "Cook"]}'

    # Register an adapter that emits text + a StructuredOutputPart, mimicking
    # what a fully-wired adapter would do when output_type is requested.
    async def _structured_stream(
        client: models.Client,
        model: models.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[events_.Event]:
        text_part = messages_.TextPart(text=json_text)
        parts: list[messages_.Part] = [text_part]
        if output_type is not None:
            import json

            parts.append(
                messages_.StructuredOutputPart(
                    data=json.loads(json_text),
                    output_type_name=f"{output_type.__module__}.{output_type.__qualname__}",
                )
            )
        msg = messages_.Message(id="m1", role="assistant", parts=parts)
        yield events_.MessageStart(message=msg.model_copy(update={"parts": []}))
        yield events_.TextStart(block_id=text_part.id)
        yield events_.TextDelta(block_id=text_part.id, chunk=json_text)
        yield events_.TextEnd(block_id=text_part.id)
        yield events_.MessageEnd(message=msg)

    models.register_stream("mock", _structured_stream)

    s = models.stream(
        MOCK_MODEL, [ai.user_message("Give me a recipe")], output_type=_Recipe
    )
    async for _ in s:
        pass

    assert s.output is not None
    assert isinstance(s.output, _Recipe)
    assert s.output.name == "Pancakes"
    assert s.output.steps == ["Mix", "Cook"]


# ---------------------------------------------------------------------------
# generate() tests
# ---------------------------------------------------------------------------

_MOCK_GEN_PROVIDER = MockProvider(adapter="mock-gen")

GENERATE_MODEL = models.Model(
    id="gen-model",
    adapter="mock-gen",
    provider=_MOCK_GEN_PROVIDER,
)


async def test_generate_dispatches_to_registered_adapter() -> None:
    """ai.generate() resolves the adapter from the registry and returns its Message."""
    sentinel = messages_.Message(
        role="assistant",
        parts=[messages_.FilePart(data=b"\x89PNG", media_type="image/png")],
    )
    called = False

    async def _mock_gen(
        client: models.Client,
        model: models.Model,
        messages: list[messages_.Message],
        params: Any = None,
    ) -> messages_.Message:
        nonlocal called
        called = True
        return sentinel

    models.register_generate("mock-gen", _mock_gen)

    result = await models.generate(
        GENERATE_MODEL, [ai.user_message("A cat")], models.ImageParams(n=1)
    )
    assert called
    assert result is sentinel
