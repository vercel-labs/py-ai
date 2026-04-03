from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pydantic

import vercel_ai_sdk as ai
from vercel_ai_sdk import models2
from vercel_ai_sdk.types import messages as messages_

# A fixed Model used in tests — adapter="mock" dispatches to the mock adapter.
MOCK_MODEL = models2.Model(id="mock-model", adapter="mock", provider="mock")

# Register a dummy provider so _auto_client() doesn't error for provider="mock".
models2._PROVIDER_DEFAULTS["mock"] = ("http://mock.test", "MOCK_API_KEY")


class MockAdapter:
    """Mock stream adapter that yields pre-configured response sequences.

    Each call to the adapter pops the next response list and yields the
    messages through a StreamHandler (matching real adapter behavior).
    Tracks ``call_count`` for assertions.
    """

    def __init__(self, responses: list[list[messages_.Message]]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.call_count = 0

    async def stream(
        self,
        client: models2.Client,
        model: models2.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[messages_.Message]:
        if self._call_index >= len(self._responses):
            raise RuntimeError("MockAdapter: no more responses configured")
        self.call_count += 1
        seq = self._responses[self._call_index]
        self._call_index += 1

        from vercel_ai_sdk.models2.core.helpers import streaming as streaming_

        handler = streaming_.StreamHandler()

        for msg in seq:
            for i, part in enumerate(msg.parts):
                if isinstance(part, messages_.TextPart):
                    bid = f"text-{i}"
                    yield handler.handle_event(streaming_.TextStart(block_id=bid))
                    if part.text:
                        yield handler.handle_event(
                            streaming_.TextDelta(block_id=bid, delta=part.text)
                        )
                    yield handler.handle_event(streaming_.TextEnd(block_id=bid))

                elif isinstance(part, messages_.ReasoningPart):
                    bid = f"reasoning-{i}"
                    yield handler.handle_event(streaming_.ReasoningStart(block_id=bid))
                    if part.text:
                        yield handler.handle_event(
                            streaming_.ReasoningDelta(block_id=bid, delta=part.text)
                        )
                    yield handler.handle_event(
                        streaming_.ReasoningEnd(block_id=bid, signature=part.signature)
                    )

                elif isinstance(part, messages_.ToolPart):
                    yield handler.handle_event(
                        streaming_.ToolStart(
                            tool_call_id=part.tool_call_id,
                            tool_name=part.tool_name,
                        )
                    )
                    if part.tool_args:
                        yield handler.handle_event(
                            streaming_.ToolArgsDelta(
                                tool_call_id=part.tool_call_id,
                                delta=part.tool_args,
                            )
                        )
                    yield handler.handle_event(
                        streaming_.ToolEnd(tool_call_id=part.tool_call_id)
                    )

        yield handler.handle_event(streaming_.MessageDone())


def mock_llm(responses: list[list[messages_.Message]]) -> MockAdapter:
    """Create a MockAdapter and register it in the models2 adapter registry.

    Returns the adapter so tests can inspect ``call_count``.
    """
    adapter = MockAdapter(responses)
    models2.register_stream("mock", adapter.stream)
    return adapter


# ── Legacy MockLLM (for tests/models/ that test the old LanguageModel ABC) ──


class MockLLM(ai.models.LanguageModel):
    """LLM that yields pre-configured response sequences, one per call.

    Converts pre-configured ``Message`` objects into ``StreamEvent`` sequences
    so the base-class ``stream()`` (which uses ``StreamHandler``) can
    reconstruct them.

    **Legacy** — kept for tests of the old ``models/`` module.
    New agent tests should use :func:`mock_llm` + ``MOCK_MODEL`` instead.
    """

    def __init__(self, responses: list[list[messages_.Message]]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.call_count = 0

    async def stream_events(
        self,
        messages: list[messages_.Message],
        tools: Sequence[ai.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
    ) -> AsyncGenerator[Any]:
        from vercel_ai_sdk.models.core import llm as llm_

        if self._call_index >= len(self._responses):
            raise RuntimeError("MockLLM: no more responses configured")
        self.call_count += 1
        seq = self._responses[self._call_index]
        self._call_index += 1

        for msg in seq:
            for i, part in enumerate(msg.parts):
                if isinstance(part, messages_.TextPart):
                    bid = f"text-{i}"
                    yield llm_.TextStart(block_id=bid)
                    if part.text:
                        yield llm_.TextDelta(block_id=bid, delta=part.text)
                    yield llm_.TextEnd(block_id=bid)

                elif isinstance(part, messages_.ReasoningPart):
                    bid = f"reasoning-{i}"
                    yield llm_.ReasoningStart(block_id=bid)
                    if part.text:
                        yield llm_.ReasoningDelta(block_id=bid, delta=part.text)
                    yield llm_.ReasoningEnd(block_id=bid, signature=part.signature)

                elif isinstance(part, messages_.ToolPart):
                    yield llm_.ToolStart(
                        tool_call_id=part.tool_call_id,
                        tool_name=part.tool_name,
                    )
                    if part.tool_args:
                        yield llm_.ToolArgsDelta(
                            tool_call_id=part.tool_call_id,
                            delta=part.tool_args,
                        )
                    yield llm_.ToolEnd(tool_call_id=part.tool_call_id)

        yield llm_.MessageDone()


# ── Helpers ──────────────────────────────────────────────────────


def text_msg(
    text: str, *, id: str = "msg-1", state: str = "done", delta: str | None = None
) -> messages_.Message:
    return messages_.Message(
        id=id,
        role="assistant",
        parts=[messages_.TextPart(text=text, state=state, delta=delta)],
    )


def tool_msg(
    *,
    id: str = "msg-1",
    tc_id: str = "tc-1",
    name: str = "test_tool",
    args: str = "{}",
    status: str = "pending",
    result: dict[str, object] | None = None,
) -> messages_.Message:
    return messages_.Message(
        id=id,
        role="assistant",
        parts=[
            messages_.ToolPart(
                tool_call_id=tc_id,
                tool_name=name,
                tool_args=args,
                status=status,
                result=result,
                state="done",
            )
        ],
    )
