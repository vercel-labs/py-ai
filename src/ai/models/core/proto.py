"""Adapter function protocols.

An *adapter function* translates between our ``Message`` types and a specific
provider API (e.g. ``"ai-gateway-v3"``, ``"anthropic-messages"``).

Adapter functions are plain async generators / coroutines — no base class
required.  The protocols below exist only for static type-checking.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Any, Protocol, runtime_checkable

import pydantic

from ...types import messages as messages_
from ...types import tools as tools_
from .client import Client
from .model import Model


@runtime_checkable
class StreamFn(Protocol):
    """Protocol for streaming adapter functions.

    Implementations yield ``Message`` snapshots as the response streams
    in.  Each snapshot is a complete, self-contained message reflecting
    the accumulated state up to that point.
    """

    def __call__(
        self,
        client: Client,
        model: Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[tools_.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[messages_.Message]: ...


@runtime_checkable
class GenerateFn(Protocol):
    """Protocol for non-streaming adapter functions (images, video, etc.).

    ``params`` is typed as ``Any`` at the protocol level because each adapter
    defines its own parameter types (e.g. ``ImageParams | VideoParams``).
    Type safety is enforced at the top-level ``generate()`` function.
    """

    async def __call__(
        self,
        client: Client,
        model: Model,
        messages: list[messages_.Message],
        params: Any = None,
    ) -> messages_.Message: ...


@runtime_checkable
class CheckConnFn(Protocol):
    """Protocol for connection-check functions.

    A check function verifies that *client* can reach the provider and that
    *model* is available there.  Returns ``True`` when the credentials are
    valid **and** the model exists on the remote side.

    The check must be **free** — it should only hit metadata / listing
    endpoints that don't consume tokens or credits.

    Non-auth transport errors (network failures, 5xx) should be raised
    rather than returning ``False`` so that callers can distinguish
    "bad credentials" from "provider unreachable".
    """

    async def __call__(
        self,
        client: Client,
        model: Model,
    ) -> bool: ...
