from __future__ import annotations

import asyncio
import json
import contextvars
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from typing import TYPE_CHECKING, Any, get_type_hints

from .. import mcp
from . import messages as messages_
from . import tools as tools_
from . import llm as llm_
from . import streams as streams_
from . import hooks as hooks_

if TYPE_CHECKING:
    from .hooks import Hook


class Runtime:
    """
    Functions decorated with @stream submit step functions to the queue.
    run() pulls steps, runs them, yields messages, and resolves futures.
    """

    class _Sentinel:
        pass

    _SENTINEL = _Sentinel()

    def __init__(self) -> None:
        self._step_queue: asyncio.Queue[
            tuple[streams_.Stream, asyncio.Future[streams_.StreamResult]]
            | Runtime._Sentinel
        ] = asyncio.Queue()

        # Message queue for streaming tools and hooks (runtime.put_message)
        self._message_queue: asyncio.Queue[messages_.Message] = asyncio.Queue()

        # Hook support: pending hooks registry
        self._pending_hooks: dict[str, tuple[asyncio.Future[Any], Hook[Any]]] = {}

    async def put_step(
        self, step_fn: streams_.Stream, future: asyncio.Future[streams_.StreamResult]
    ) -> None:
        """Submit a step function to be executed by run()."""
        await self._step_queue.put((step_fn, future))

    async def get_step(
        self,
    ) -> tuple[streams_.Stream, asyncio.Future[streams_.StreamResult]] | _Sentinel:
        """Get next step from queue (called by run())."""
        return await self._step_queue.get()

    async def put_message(self, message: messages_.Message) -> None:
        await self._message_queue.put(message)

    def get_all_messages(self) -> list[messages_.Message]:
        """Drain all pending messages from the message queue."""
        msgs = []
        while not self._message_queue.empty():
            try:
                msgs.append(self._message_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return msgs

    async def put_hook(
        self, hook: hooks_.Hook[Any], future: asyncio.Future[Any]
    ) -> None:
        self._pending_hooks[hook.id] = (future, hook)
        await self._message_queue.put(hook.to_message(status="pending"))

    def get_all_hooks(self) -> dict[str, Hook[Any]]:
        """Get all pending hooks (for inspection/UI)."""
        return {k: v[1] for k, v in self._pending_hooks.items()}

    async def signal_done(self) -> None:
        """Signal that no more steps will be submitted."""
        await self._step_queue.put(self._SENTINEL)


_runtime: contextvars.ContextVar[Runtime] = contextvars.ContextVar("runtime")


def _find_runtime_param(fn: Callable[..., Any]) -> str | None:
    """Find a parameter typed as Runtime, return its name or None."""
    try:
        hints = get_type_hints(fn)
    except Exception:
        return None
    for name, hint in hints.items():
        if hint is Runtime:
            return name
    return None


# these are convenience functions assembled from
# the core primitives. users could use this for reference
# when implementing custom workflows.


@streams_.stream
async def stream_step(
    llm: llm_.LanguageModel,
    messages: list[messages_.Message],
    tools: list[tools_.Tool] | None = None,
    label: str | None = None,
) -> AsyncGenerator[messages_.Message, None]:
    """Single LLM call that streams to Runtime."""
    async for msg in llm.stream(messages=messages, tools=tools):
        msg.label = label
        yield msg


async def stream_loop(
    llm: llm_.LanguageModel,
    messages: list[messages_.Message],
    tools: list[tools_.Tool],
    label: str | None = None,
) -> streams_.StreamResult:
    """Agent loop: stream LLM, execute tools, repeat until done."""
    local_messages = list(messages)
    runtime = _runtime.get(None)

    while True:
        result = await stream_step(llm, local_messages, tools, label=label)

        if not result.tool_calls:
            return result

        local_messages.append(result.last_message)

        await asyncio.gather(*(tc.execute() for tc in result.tool_calls))

        # Emit a copy of the message with tool results so the UI sees the transition
        # from status="pending" to status="result". We copy to avoid mutation issues
        # since the original message object was already yielded.
        if runtime and result.last_message:
            await runtime.put_message(result.last_message.model_copy(deep=True))


async def execute_tool(
    tool_call: messages_.ToolPart,
    tools: list[tools_.Tool],
    message: messages_.Message | None = None,
) -> Any:
    """
    Execute a single tool call and optionally update the message.

    If message is provided, updates the tool part with the result.
    Can be wrapped with @workflow.step for durability.
    Use with asyncio.gather() for parallel execution.
    """
    tool_fn = next((t for t in tools if t.name == tool_call.tool_name), None)
    if tool_fn is None:
        raise ValueError(f"Tool not found: {tool_call.tool_name}")

    # Inject runtime if the tool has a Runtime-typed parameter
    kwargs: dict[str, Any] = (
        json.loads(tool_call.tool_args) if tool_call.tool_args else {}
    )
    rt = _runtime.get(None)
    if rt and (runtime_param := _find_runtime_param(tool_fn.fn)):
        kwargs[runtime_param] = rt

    result = await tool_fn.fn(**kwargs)

    if message is not None:
        tool_part = message.get_tool_part(tool_call.tool_call_id)
        if tool_part:
            tool_part.status = "result"
            tool_part.result = result

    return result


async def _stop_when_done(runtime: Runtime, task: Awaitable[None]) -> None:
    try:
        await task
    finally:
        await runtime.signal_done()


async def run(
    root: Callable[..., Coroutine[Any, Any, Any]],
    *args: Any,
) -> AsyncGenerator[messages_.Message, None]:
    """
    Main entry point.

    1. Starts the root function as a background task
    2. Pulls steps from the Runtime queue
    3. Executes each step, yielding messages
    4. Resolves futures to unblock user code
    """
    from . import hooks as hooks_

    runtime = Runtime()
    token_runtime = _runtime.set(runtime)

    mcp_pool: dict[str, mcp.client._Connection] = {}
    mcp_token = mcp.client._pool.set(mcp_pool)

    kwargs: dict[str, Any] = {}
    if runtime_param := _find_runtime_param(root):
        kwargs[runtime_param] = runtime

    try:
        async with asyncio.TaskGroup() as tg:
            _task: asyncio.Task[None] = tg.create_task(
                _stop_when_done(runtime, root(*args, **kwargs))
            )

            while True:
                # Drain any pending messages (including hook messages)
                for msg in runtime.get_all_messages():
                    yield msg

                # Use wait_for with a short timeout to allow checking queues periodically
                # This is needed because hook messages can arrive while we're waiting
                try:
                    step_item = await asyncio.wait_for(runtime.get_step(), timeout=0.1)
                except asyncio.TimeoutError:
                    # No step ready, loop back to drain queues
                    continue

                if isinstance(step_item, Runtime._Sentinel):
                    # Drain remaining messages before exiting
                    for msg in runtime.get_all_messages():
                        yield msg
                    break

                step_fn, future = step_item

                # Drain any messages that arrived before this step
                # (e.g., tool result messages from the previous step)
                for tool_msg in runtime.get_all_messages():
                    yield tool_msg

                result_messages: list[messages_.Message] = []

                async for msg in step_fn():
                    # Yield a copy to protect consumers from mutations
                    # (e.g., tool execution mutates ToolPart.status in place)
                    msg_copy = msg.model_copy(deep=True)
                    yield msg_copy
                    result_messages.append(msg)

                    # Also drain any messages during step
                    for tool_msg in runtime.get_all_messages():
                        yield tool_msg

                step_result = streams_.StreamResult(messages=result_messages)

                future.set_result(step_result)

                # Yield to allow the agent task to run (e.g., tool execution)
                # then drain any messages it produced
                await asyncio.sleep(0)
                for tool_msg in runtime.get_all_messages():
                    yield tool_msg

    except ExceptionGroup as eg:
        # Extract HookPending from ExceptionGroup and re-raise it directly
        # (TaskGroup wraps exceptions from tasks in ExceptionGroup)
        hook_pending = None
        other_exceptions = []
        for exc in eg.exceptions:
            if isinstance(exc, hooks_.HookPending):
                hook_pending = exc
            else:
                other_exceptions.append(exc)

        if hook_pending is not None and not other_exceptions:
            raise hook_pending from None
        else:
            raise

    finally:
        if mcp_token is not None:
            await mcp.client.close_connections()
            mcp.client._pool.reset(mcp_token)

        _runtime.reset(token_runtime)
