"""Utility functions"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

_EMPTY: Any = object()


class _Stop:
    pass


_STOP = _Stop()


@dataclasses.dataclass
class _Exception:
    exception: Exception


@contextlib.asynccontextmanager
async def unwrap_generator_exit() -> AsyncIterator[None]:
    """Convert a ``BaseExceptionGroup`` of only ``GeneratorExit``s into a single one.

    ``asyncio.TaskGroup``'s ``__aexit__`` wraps any body exception (including
    ``GeneratorExit``) into a ``BaseExceptionGroup``. Inside an async
    generator that means ``aclose()`` propagates an ``ExceptionGroup`` instead
    of the bare ``GeneratorExit`` the protocol expects, and the aclose-task
    ends up with an unretrieved exception. Wrapping the ``async with
    TaskGroup(...)`` block in this manager unwraps the group back to a plain
    ``GeneratorExit`` so the close path stays clean.
    """
    try:
        yield
    except BaseExceptionGroup as eg:
        matched, rest = eg.split(GeneratorExit)
        if matched is not None and rest is None:
            raise GeneratorExit from None
        raise


@contextlib.asynccontextmanager
async def maybe_aclosing[T: AsyncIterable[Any]](iter: T) -> AsyncIterator[T]:
    """Like ``contextlib.aclosing`` but a no-op if ``iter`` has no ``aclose``.

    Useful when consuming an arbitrary ``AsyncIterable[T]`` whose concrete
    type may or may not be an async generator.
    """
    try:
        yield iter
    finally:
        aclose = getattr(iter, "aclose", None)
        if aclose is not None:
            await aclose()


async def decouple[T](
    iter: AsyncIterable[T],
    *,
    task_group: asyncio.TaskGroup | None = None,
    size: int = 1,
) -> AsyncIterator[T]:
    """Drive ``iter`` from a single worker task and yield its items.

    Ensures every ``__anext__`` on ``iter`` runs in the same task context, so
    contextvars set or relied on by the iterable behave consistently across
    yields. Without this, callers that wrap each ``anext`` in a fresh task
    (e.g. ``merge``) would run each step in a different copy of the context.

    We try pretty hard to make sure that ``iter`` gets aclose()d in
    the same task that it was run it.

    On asyncio shutdown, tasks all get canceled before async
    generators are closed, so we should be OK.

    """
    queue: asyncio.Queue[_Stop | _Exception | T] = asyncio.Queue(size)

    async def worker() -> None:
        async with maybe_aclosing(iter):
            try:
                async for x in iter:
                    await queue.put(x)
            except Exception as e:
                await queue.put(_Exception(e))
                return
        await queue.put(_STOP)

    if task_group:
        task = task_group.create_task(worker())
    else:
        task = asyncio.create_task(worker())

    try:
        while True:
            el = await queue.get()
            if isinstance(el, _Stop):
                break
            elif isinstance(el, _Exception):
                raise el.exception
            yield el
    finally:
        task.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await task


async def merge[T](*aiterables: AsyncIterable[T]) -> AsyncIterator[T]:
    # We use unwrap_generator_exit() to keep a GeneratorExit that gets
    # packaged in an ExceptionGroup from causing grief. But maybe we
    # ought to not use a TaskGroup?
    async with unwrap_generator_exit(), asyncio.TaskGroup() as tg:
        aiters = [decouple(iter, task_group=tg) for iter in aiterables]

        # Launch a task doing anext on every iterator
        tasks: list[asyncio.Future[T] | None] = [
            tg.create_task(anext(iter, _EMPTY)) for iter in aiters
        ]

        while any(tasks):
            done, _ = await asyncio.wait(
                [t for t in tasks if t],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for t in done:
                idx = tasks.index(t)
                val = t.result()
                if val is _EMPTY:
                    tasks[idx] = None
                else:
                    # Fire off a new task for the relevant iterator
                    iter = aiters[idx]
                    tasks[idx] = tg.create_task(anext(iter, _EMPTY))
                    yield val
