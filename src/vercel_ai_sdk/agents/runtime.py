"""Runtime: message sink that connects producer coroutines to the consumer."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable

from .. import types
from ..telemetry import events as telemetry


class Runtime:
    """Central message queue. Producers put messages, run() yields them."""

    class _Sentinel:
        pass

    _SENTINEL = _Sentinel()

    def __init__(self) -> None:
        self._message_queue: asyncio.Queue[types.Message | Runtime._Sentinel] = (
            asyncio.Queue()
        )
        self._hook_labels: set[str] = set()

    async def put_message(self, message: types.Message) -> None:
        await self._message_queue.put(message)

    async def signal_done(self) -> None:
        await self._message_queue.put(self._SENTINEL)

    def track_hook_label(self, label: str) -> None:
        """Register a hook label for cleanup when the run ends."""
        self._hook_labels.add(label)

    def cleanup_hooks(self) -> None:
        """Remove all hook registry entries for this run."""
        from . import hooks as hooks_

        hooks_.cleanup_run(self._hook_labels)
        self._hook_labels.clear()


_runtime: contextvars.ContextVar[Runtime] = contextvars.ContextVar("runtime")


def get_runtime() -> Runtime:
    """Return the active Runtime. Raises LookupError outside of run()."""
    return _runtime.get()


async def _stop_when_done(runtime: Runtime, task: Awaitable[None]) -> None:
    try:
        await task
    finally:
        await runtime.signal_done()


async def run(
    source: AsyncIterable[types.Message],
) -> AsyncGenerator[types.Message]:
    """Run *source* and yield every message that gets put into the Runtime queue."""
    from .mcp import client as mcp_client

    rt = Runtime()
    token = _runtime.set(rt)

    # MCP connection pool — scoped to this run.
    mcp_pool: dict[str, mcp_client._Connection] = {}
    mcp_token = mcp_client._pool.set(mcp_pool)

    token_run_id = telemetry.start_run()
    telemetry.handle(telemetry.RunStartEvent())

    run_error: BaseException | None = None
    total_usage: types.Usage | None = None

    async def _drain() -> None:
        async for message in source:
            await rt.put_message(message)

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(_stop_when_done(rt, _drain()))

            while True:
                item = await rt._message_queue.get()
                if isinstance(item, Runtime._Sentinel):
                    return
                # Track usage from done messages.
                if item.is_done and item.usage is not None:
                    total_usage = (
                        item.usage if total_usage is None else total_usage + item.usage
                    )
                yield item

    except BaseException as exc:
        run_error = exc
        raise

    finally:
        telemetry.handle(telemetry.RunFinishEvent(usage=total_usage, error=run_error))
        telemetry.end_run(token_run_id)

        rt.cleanup_hooks()

        await mcp_client.close_connections()
        mcp_client._pool.reset(mcp_token)

        _runtime.reset(token)
