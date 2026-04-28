"""Utility functions"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

_EMPTY: Any = object()


async def merge[T](*aiterables: AsyncIterable[T]) -> AsyncIterator[T]:
    aiters = [aiter(iter) for iter in aiterables]

    async with asyncio.TaskGroup() as tg:
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
