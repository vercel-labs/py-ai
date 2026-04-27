from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable, Sequence
from typing import Any

import pydantic

import ai
from ai import models
from ai.types import builders
from ai.types import events as events_
from ai.types import messages as messages_


class MockProvider:
    """Minimal provider that satisfies the ``Provider`` protocol for tests.

    Carries just enough state so that ``Model`` objects can be constructed
    and ``auto_client`` can resolve a client.
    """

    def __init__(
        self,
        *,
        name: str = "mock",
        adapter: str = "mock",
        base_url: str = "http://mock.test",
        api_key_env: str | None = "MOCK_API_KEY",
    ) -> None:
        self._name = name
        self._adapter = adapter
        self._base_url = base_url
        self._api_key_env = api_key_env

    @property
    def name(self) -> str:
        return self._name

    @property
    def adapter(self) -> str:
        return self._adapter

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def api_key_env(self) -> str | None:
        return self._api_key_env

    def client(self) -> models.Client:
        import os

        api_key = os.environ.get(self._api_key_env) if self._api_key_env else None
        return models.Client(base_url=self._base_url, api_key=api_key)

    async def check(self, client: models.Client, model: models.Model) -> bool:
        return True

    async def list(self, *, client: models.Client | None = None) -> list[str]:
        return []

    def __call__(
        self,
        model_id: str,
        *,
        client: models.Client | None = None,
    ) -> models.Model:
        return models.Model(
            id=model_id,
            adapter=self._adapter,
            provider=self,
            client=client,
        )

    def __repr__(self) -> str:
        return self._name


MOCK_PROVIDER = MockProvider()

# A fixed Model used in tests — adapter="mock" dispatches to the mock adapter.
MOCK_MODEL = models.Model(
    id="mock-model",
    adapter="mock",
    provider=MOCK_PROVIDER,
)


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
        client: models.Client,
        model: models.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[events_.Event]:
        if self._call_index >= len(self._responses):
            raise RuntimeError("MockAdapter: no more responses configured")
        self.call_count += 1
        seq = self._responses[self._call_index]
        self._call_index += 1

        from ai.models.core.helpers import streaming as streaming_

        message_id = seq[0].id if seq else messages_.generate_id()
        handler = streaming_.StreamHandler(message_id=message_id)
        yield handler.message_start()

        for msg in seq:
            for i, part in enumerate(msg.parts):
                if isinstance(part, messages_.TextPart):
                    bid = f"text-{i}"
                    for event in handler.handle_event(
                        streaming_.TextStart(block_id=bid)
                    ):
                        yield event
                    if part.text:
                        for event in handler.handle_event(
                            streaming_.TextDelta(block_id=bid, delta=part.text)
                        ):
                            yield event
                    for event in handler.handle_event(streaming_.TextEnd(block_id=bid)):
                        yield event

                elif isinstance(part, messages_.ReasoningPart):
                    bid = f"reasoning-{i}"
                    for event in handler.handle_event(
                        streaming_.ReasoningStart(block_id=bid)
                    ):
                        yield event
                    if part.text:
                        for event in handler.handle_event(
                            streaming_.ReasoningDelta(block_id=bid, delta=part.text)
                        ):
                            yield event
                    for event in handler.handle_event(
                        streaming_.ReasoningEnd(block_id=bid, signature=part.signature)
                    ):
                        yield event

                elif isinstance(part, messages_.ToolCallPart):
                    for event in handler.handle_event(
                        streaming_.ToolStart(
                            tool_call_id=part.tool_call_id,
                            tool_name=part.tool_name,
                        )
                    ):
                        yield event
                    if part.tool_args:
                        for event in handler.handle_event(
                            streaming_.ToolArgsDelta(
                                tool_call_id=part.tool_call_id,
                                delta=part.tool_args,
                            )
                        ):
                            yield event
                    for event in handler.handle_event(
                        streaming_.ToolEnd(tool_call_id=part.tool_call_id)
                    ):
                        yield event

                elif isinstance(part, messages_.StructuredOutputPart):
                    handler._current_parts[part.id] = part

                elif isinstance(part, messages_.FilePart):
                    for event in handler.handle_event(
                        streaming_.FileEvent(
                            block_id=part.id,
                            media_type=part.media_type,
                            data=part.data if isinstance(part.data, str) else "",
                        )
                    ):
                        yield event

        for event in handler.handle_event(streaming_.MessageDone()):
            yield event


def mock_llm(responses: list[list[messages_.Message]]) -> MockAdapter:
    """Create a MockAdapter and register it in the models adapter registry.

    Returns the adapter so tests can inspect ``call_count``.
    """
    adapter = MockAdapter(responses)
    models.register_stream("mock", adapter.stream)
    return adapter


async def collect_messages(
    source: AsyncIterable[events_.Event],
) -> list[messages_.Message]:
    """Collect terminal messages from an event stream."""
    result: list[messages_.Message] = []
    async for event in source:
        if isinstance(event, events_.MessageEnd):
            result.append(event.message)
    return result


class MockGenerateAdapter:
    """Mock generate adapter that returns pre-configured responses.

    Each call pops the next response.  Tracks ``call_count`` for assertions.
    """

    def __init__(self, responses: list[messages_.Message]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.call_count = 0

    async def generate(
        self,
        client: models.Client,
        model: models.Model,
        messages: list[messages_.Message],
        params: Any = None,
    ) -> messages_.Message:
        if self._call_index >= len(self._responses):
            raise RuntimeError("MockGenerateAdapter: no more responses configured")
        self.call_count += 1
        msg = self._responses[self._call_index]
        self._call_index += 1
        return msg


def mock_generate(responses: list[messages_.Message]) -> MockGenerateAdapter:
    """Create a MockGenerateAdapter and register it.

    Returns the adapter so tests can inspect ``call_count``.
    """
    adapter = MockGenerateAdapter(responses)
    models.register_generate("mock", adapter.generate)
    return adapter


# ── Helpers ──────────────────────────────────────────────────────


def text_msg(
    text: str,
    *,
    id: str = "msg-1",
) -> messages_.Message:
    part: messages_.Part = messages_.TextPart(text=text)
    return messages_.Message(id=id, role="assistant", parts=[part])


def tool_call_msg(
    *,
    id: str = "msg-1",
    tc_id: str = "tc-1",
    name: str = "test_tool",
    args: str = "{}",
) -> messages_.Message:
    """Assistant message containing a tool call."""
    part: messages_.Part = messages_.ToolCallPart(
        tool_call_id=tc_id,
        tool_name=name,
        tool_args=args,
    )
    return messages_.Message(id=id, role="assistant", parts=[part])


def tool_result_msg(
    *,
    tc_id: str = "tc-1",
    name: str = "test_tool",
    result: Any = None,
    is_error: bool = False,
) -> messages_.Message:
    """Tool-result message."""
    return builders.tool_message(
        tool_call_id=tc_id,
        tool_name=name,
        result=result,
        is_error=is_error,
    )
