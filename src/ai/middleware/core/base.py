from __future__ import annotations

import contextvars
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING

import pydantic

from . import context
from ai import types


class Middleware:
    """Base middleware class.  Override the methods you need.

    Default implementations call ``next(call)`` — a transparent pass-through.
    """

    async def wrap_agent_run(
        self,
        args: context.AgentRunContext,
        wrapped: Callable[[context.AgentRunContext], AsyncGenerator[types.Message]],
    ) -> AsyncGenerator[types.Message]:
        """Wrap an agent run.

        ``next(call)`` returns an async generator of ``Message`` objects.
        Override to add tracing, durability checkpoints, or other
        run-scoped behavior::

            async def wrap_agent_run(self, call, next):
                span = start_span("agent.run")
                async for msg in next(call):
                    yield msg
                span.end()
        """
        async for msg in wrapped(args):
            yield msg

    async def wrap_model(
        self,
        args: context.ModelContext,
        wrapped: Callable[
            [context.ModelContext], Awaitable[types.stream.StreamResultLike]
        ],
    ) -> types.stream.StreamResultLike:
        """Wrap a model streaming call.

        ``next(call)`` returns a :class:`~ai.types.StreamResultLike` that
        is async-iterable over ``Message`` snapshots.  You can do work
        before, iterate / transform the stream, or do cleanup after.

        To transform the stream, use
        :meth:`~ai.models.StreamResult.from_generator`::

            async def wrap_model(self, call, next):
                stream = await next(call)
                async def _add_suffix():
                    async for msg in stream:
                        yield msg
                from ai.models import StreamResult
                return StreamResult.from_generator(_add_suffix())
        """
        return await wrapped(args)

    async def wrap_generate(
        self,
        args: context.GenerateContext,
        wrapped: Callable[[context.GenerateContext], Awaitable[types.Message]],
    ) -> types.Message:
        """Wrap a model generate call (images, video, etc.)."""
        return await wrapped(args)

    async def wrap_tool(
        self,
        args: context.ToolContext,
        wrapped: Callable[[context.ToolContext], Awaitable[types.Message]],
    ) -> types.Message:
        """Wrap a tool execution.

        ``next(call)`` returns the tool-result ``Message``.
        """
        return await wrapped(args)

    async def wrap_hook(
        self,
        args: context.HookContext,
        wrapped: Callable[[context.HookContext], Awaitable[pydantic.BaseModel]],
    ) -> pydantic.BaseModel:
        """Wrap a hook suspension point.

        ``next(call)`` blocks until the hook is resolved and returns the
        validated payload instance.
        """
        return await wrapped(args)


# ---------------------------------------------------------------------------
# Run-scoped middleware via ContextVar
# ---------------------------------------------------------------------------

_active: contextvars.ContextVar[list[Middleware]] = contextvars.ContextVar(
    "middleware",
)

_EMPTY: list[Middleware] = []


def get() -> list[Middleware]:
    """Return the middleware stack for the current run (empty if none)."""
    return _active.get(_EMPTY)


Token = contextvars.Token[list[Middleware]]


def activate(mw: list[Middleware]) -> Token:
    """Set the middleware stack for the current run.  Returns a token for reset."""
    return _active.set(mw)


def deactivate(token: Token) -> None:
    """Restore the previous middleware stack."""
    _active.reset(token)


# ---------------------------------------------------------------------------
# Chain builders — compose the middleware stack for each surface.
#
# Each builder takes the *real* implementation as a callable and returns
# a callable with the same signature that routes through middleware.
#
# When no middleware is active, the real implementation is returned
# directly — zero overhead.
# ---------------------------------------------------------------------------


def _build_model_chain(
    real: Callable[[ModelContext], Awaitable[StreamResultLike]],
) -> Callable[[ModelContext], Awaitable[StreamResultLike]]:
    mw = get()
    if not mw:
        return real

    chain = real
    for m in reversed(mw):

        def _make(
            m: Middleware,
            nxt: Callable[[ModelContext], Awaitable[StreamResultLike]],
        ) -> Callable[[ModelContext], Awaitable[StreamResultLike]]:
            async def _wrapped(call: ModelContext) -> StreamResultLike:
                return await m.wrap_model(call, nxt)

            return _wrapped

        chain = _make(m, chain)
    return chain


def _build_generate_chain(
    real: Callable[[GenerateContext], Awaitable[_Message]],
) -> Callable[[GenerateContext], Awaitable[_Message]]:
    mw = get()
    if not mw:
        return real

    chain = real
    for m in reversed(mw):

        def _make(
            m: Middleware, nxt: Callable[[GenerateContext], Awaitable[_Message]]
        ) -> Callable[[GenerateContext], Awaitable[_Message]]:
            async def _wrapped(call: GenerateContext) -> _Message:
                return await m.wrap_generate(call, nxt)

            return _wrapped

        chain = _make(m, chain)
    return chain


def _build_tool_chain(
    real: Callable[[ToolContext], Awaitable[_Message]],
) -> Callable[[ToolContext], Awaitable[_Message]]:
    mw = get()
    if not mw:
        return real

    chain = real
    for m in reversed(mw):

        def _make(
            m: Middleware, nxt: Callable[[ToolContext], Awaitable[_Message]]
        ) -> Callable[[ToolContext], Awaitable[_Message]]:
            async def _wrapped(call: ToolContext) -> _Message:
                return await m.wrap_tool(call, nxt)

            return _wrapped

        chain = _make(m, chain)
    return chain


def _build_hook_chain(
    real: Callable[[HookContext], Awaitable[pydantic.BaseModel]],
) -> Callable[[HookContext], Awaitable[pydantic.BaseModel]]:
    mw = get()
    if not mw:
        return real

    chain = real
    for m in reversed(mw):

        def _make(
            m: Middleware,
            nxt: Callable[[HookContext], Awaitable[pydantic.BaseModel]],
        ) -> Callable[[HookContext], Awaitable[pydantic.BaseModel]]:
            async def _wrapped(call: HookContext) -> pydantic.BaseModel:
                return await m.wrap_hook(call, nxt)

            return _wrapped

        chain = _make(m, chain)
    return chain


def _build_agent_run_chain(
    real: _AgentRunNext,
) -> _AgentRunNext:
    mw = get()
    if not mw:
        return real

    chain = real
    for m in reversed(mw):

        def _make(m: Middleware, nxt: _AgentRunNext) -> _AgentRunNext:
            async def _wrapped(call: AgentRunContext) -> AsyncGenerator[_Message]:
                async for msg in m.wrap_agent_run(call, nxt):
                    yield msg

            return _wrapped

        chain = _make(m, chain)
    return chain


__all__ = [
    "AgentRunContext",
    "GenerateContext",
    "HookContext",
    "Middleware",
    "ModelContext",
    "StreamResultLike",
    "ToolContext",
    "activate",
    "deactivate",
    "get",
]
