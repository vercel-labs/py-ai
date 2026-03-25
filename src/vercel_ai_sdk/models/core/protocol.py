"""Execution protocols and the Stream result type.

``StreamFn`` and ``GenerateFn`` define the execution contract that
provider adapters must satisfy.  ``Stream`` wraps an async generator
of :class:`Message` objects into an async-iterable *and* awaitable
result with convenience properties.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator, Sequence
from typing import Any, Protocol, runtime_checkable

import pydantic

from ...types import messages as messages_
from ...types import tools as tools_
from .model import Model

# ── Execution protocols ───────────────────────────────────────────


@runtime_checkable
class StreamFn(Protocol):
    """Protocol for streaming LLM calls.

    Implementations accept a :class:`Model`, messages, and optional tools /
    output type, and return an async generator that yields
    :class:`Message` snapshots as the response streams in.
    """

    def __call__(
        self,
        model: Model,
        messages: list[messages_.Message],
        tools: Sequence[tools_.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
    ) -> AsyncGenerator[messages_.Message]: ...


@runtime_checkable
class GenerateFn(Protocol):
    """Protocol for non-streaming generation (image, video, etc.).

    Implementations accept a :class:`Model`, messages, and arbitrary
    keyword arguments forwarded from the caller.
    """

    async def __call__(
        self,
        model: Model,
        messages: list[messages_.Message],
        **kwargs: Any,
    ) -> messages_.Message: ...


# ── Stream result ─────────────────────────────────────────────────


class Stream:
    """Async-iterable *and* awaitable wrapper around a message generator.

    Usage::

        # Streaming
        stream = Stream(gen)
        async for msg in stream:
            print(msg.text)

        # Or just await the final result
        stream = Stream(gen)
        await stream
        stream.result   # last Message
        stream.text     # concatenated text
    """

    def __init__(self, generator: AsyncGenerator[messages_.Message]) -> None:
        self._generator = generator
        self._messages: list[messages_.Message] = []
        self._done = False

    # ── Async iteration ───────────────────────────────────────────

    async def __aiter__(self) -> AsyncGenerator[messages_.Message]:
        if self._done:
            # Already consumed — replay from buffer
            for msg in self._messages:
                yield msg
            return

        async for msg in self._generator:
            self._messages.append(msg)
            yield msg
        self._done = True

    # ── Awaitable ─────────────────────────────────────────────────

    def __await__(self) -> Generator[Any, None, Stream]:
        return self._drain().__await__()

    async def _drain(self) -> Stream:
        """Consume the entire generator, populating result fields."""
        if not self._done:
            async for _ in self:
                pass
        return self

    # ── Result properties (available after iteration / await) ─────

    @property
    def messages(self) -> list[messages_.Message]:
        """All messages yielded during streaming."""
        return list(self._messages)

    @property
    def result(self) -> messages_.Message | None:
        """The last message (final snapshot), or ``None`` if empty."""
        return self._messages[-1] if self._messages else None

    @property
    def tool_calls(self) -> list[messages_.ToolPart]:
        """Tool-call parts from the final message."""
        if not self._messages:
            return []
        return [
            p for p in self._messages[-1].parts if isinstance(p, messages_.ToolPart)
        ]

    @property
    def text(self) -> str:
        """Concatenated text from the final message."""
        if not self._messages:
            return ""
        return "".join(
            p.text
            for p in self._messages[-1].parts
            if isinstance(p, messages_.TextPart)
        )

    @property
    def usage(self) -> messages_.Usage | None:
        """Usage from the final message, if available."""
        if not self._messages:
            return None
        return self._messages[-1].usage
