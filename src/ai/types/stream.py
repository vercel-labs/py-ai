"""StreamResultLike — structural protocol for stream results.

Middleware authors can type-check against this protocol without depending
on the concrete ``StreamResult`` class in ``ai.models``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Protocol, runtime_checkable

from . import messages as messages_


@runtime_checkable
class StreamResultLike(Protocol):
    """Structural protocol satisfied by :class:`ai.models.StreamResult`.

    Middleware that transforms or replaces the stream returned by
    ``wrap_model`` should return an object satisfying this protocol.
    The easiest way is ``StreamResult.from_generator(gen)``.
    """

    def __aiter__(self) -> AsyncGenerator[messages_.Message]: ...

    @property
    def text(self) -> str: ...

    @property
    def tool_calls(self) -> list[messages_.ToolCallPart]: ...

    @property
    def usage(self) -> messages_.Usage | None: ...

    @property
    def output(self) -> Any: ...

    @property
    def turn_id(self) -> str | None: ...
