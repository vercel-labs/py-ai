"""
Core runtime for step-based agent execution.

Key components:
- Runtime: Step queue with Future coordination
- stream: Decorator to wire async generators into Runtime
- stream_step: Single LLM call, returns StepResult
- execute_tool: Single tool execution
- stream_loop: Convenience wrapper for full agent loop
- execute: Central executor that yields messages
"""

from __future__ import annotations

import abc
import asyncio
import contextvars
import functools
import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from .. import mcp
from . import messages as messages_
from . import tools as tools_


# --- Abstract base ---


class LanguageModel(abc.ABC):
    @abc.abstractmethod
    async def stream(
        self, messages: list[messages_.Message], tools: list[tools_.Tool] | None = None
    ) -> AsyncGenerator[messages_.Message, None]:
        raise NotImplementedError
        yield


# --- Step types ---


@dataclass
class ToolCall:
    """A tool call extracted from an LLM response."""

    tool_call_id: str
    tool_name: str
    tool_args: dict[str, Any]


@dataclass
class StepResult:
    """Result of executing a step - serializable for durability replay."""

    messages: list[messages_.Message] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def last_message(self) -> messages_.Message | None:
        return self.messages[-1] if self.messages else None

    @property
    def text(self) -> str:
        if self.last_message:
            return self.last_message.text
        return ""


StepFn = Callable[[], AsyncGenerator[messages_.Message, None]]


# --- Runtime ---


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
            tuple[StepFn, asyncio.Future[StepResult]] | Runtime._Sentinel
        ] = asyncio.Queue()

        # Message queue for streaming tools (runtime.put_message)
        self._message_queue: asyncio.Queue[messages_.Message] = asyncio.Queue()

    async def put_step(
        self, step_fn: StepFn, future: asyncio.Future[StepResult]
    ) -> None:
        """Submit a step function to be executed by run()."""
        await self._step_queue.put((step_fn, future))

    async def get_step(
        self,
    ) -> tuple[StepFn, asyncio.Future[StepResult]] | _Sentinel:
        """Get next step from queue (called by run())."""
        return await self._step_queue.get()

    async def signal_done(self) -> None:
        """Signal that no more steps will be submitted."""
        await self._step_queue.put(self._SENTINEL)

    # For streaming tools that want to emit messages directly
    async def put_message(self, message: messages_.Message) -> None:
        await self._message_queue.put(message)

    def consume_messages(self) -> list[messages_.Message]:
        """Drain all pending messages from the message queue."""
        msgs = []
        while not self._message_queue.empty():
            try:
                msgs.append(self._message_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return msgs


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


def _extract_tool_calls(message: messages_.Message) -> list[ToolCall]:
    """Extract tool calls from a completed message."""
    tool_calls = []
    for part in message.parts:
        if isinstance(part, messages_.ToolPart):
            tool_calls.append(
                ToolCall(
                    tool_call_id=part.tool_call_id,
                    tool_name=part.tool_name,
                    tool_args=json.loads(part.tool_args)
                    if isinstance(part.tool_args, str)
                    else part.tool_args,
                )
            )
    return tool_calls


# --- Decorators ---


def stream(
    fn: Callable[..., AsyncGenerator[messages_.Message, None]],
) -> Callable[..., Any]:
    """
    Decorator: wraps an async generator to submit as a step to Runtime.

    The decorated function submits its work to the Runtime queue and
    blocks until run() processes it, then returns the StepResult.
    """

    @functools.wraps(fn)
    async def wrapped(*args: Any, **kwargs: Any) -> StepResult:
        rt: Runtime | None = _runtime.get(None)
        if rt is None:
            raise ValueError("No Runtime context - must be called within ai.execute()")

        future: asyncio.Future[StepResult] = asyncio.Future()

        async def step_fn() -> AsyncGenerator[messages_.Message, None]:
            async for msg in fn(*args, **kwargs):
                yield msg

        await rt.put_step(step_fn, future)
        return await future

    return wrapped


# --- Primitives ---


@stream
async def stream_step(
    llm: LanguageModel,
    messages: list[messages_.Message],
    tools: list[tools_.Tool] | None = None,
    label: str | None = None,
) -> AsyncGenerator[messages_.Message, None]:
    """
    Single LLM call that streams to Runtime.

    Returns StepResult with .tool_calls, .text, .last_message when awaited.
    Can be wrapped with @workflow.step for durability.
    """
    async for msg in llm.stream(messages=messages, tools=tools):
        msg.label = label
        yield msg


async def execute_tool(
    tool_call: ToolCall,
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
    kwargs: dict[str, Any] = dict(tool_call.tool_args)
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


# --- Convenience ---


async def stream_loop(
    llm: LanguageModel,
    messages: list[messages_.Message],
    tools: list[tools_.Tool],
    label: str | None = None,
) -> StepResult:
    """
    Full agent loop: stream LLM, execute tools, repeat until done.

    Convenience wrapper that uses stream_step and execute_tool internally.
    Returns the final StepResult.
    """
    local_messages = list(messages)

    while True:
        result = await stream_step(llm, local_messages, tools, label=label)

        if not result.tool_calls:
            return result

        local_messages.append(result.last_message)

        await asyncio.gather(
            *(execute_tool(tc, tools, result.last_message) for tc in result.tool_calls)
        )


# --- Executor ---


async def _stop_when_done(runtime: Runtime, task: Awaitable[None]) -> None:
    try:
        await task
    finally:
        await runtime.signal_done()


async def run(
    root: Callable[..., Coroutine[Any, Any, Any]], *args: Any
) -> AsyncGenerator[messages_.Message, None]:
    """
    Main entry point.

    1. Starts the root function as a background task
    2. Pulls steps from the Runtime queue
    3. Executes each step, yielding messages
    4. Resolves futures to unblock user code
    """
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
                # Drain any messages from streaming tools
                for msg in runtime.consume_messages():
                    yield msg

                item = await runtime.get_step()

                if isinstance(item, Runtime._Sentinel):
                    # Drain remaining messages before exiting
                    for msg in runtime.consume_messages():
                        yield msg
                    break

                step_fn, future = item

                result_messages: list[messages_.Message] = []
                last_message: messages_.Message | None = None

                async for msg in step_fn():
                    yield msg
                    result_messages.append(msg)
                    if msg.is_done:
                        last_message = msg

                    # Also drain any messages from streaming tools during step
                    for tool_msg in runtime.consume_messages():
                        yield tool_msg

                tool_calls = _extract_tool_calls(last_message) if last_message else []
                step_result = StepResult(
                    messages=result_messages,
                    tool_calls=tool_calls,
                )

                future.set_result(step_result)

    finally:
        if mcp_token is not None:
            await mcp.client.close_connections()
            mcp.client._pool.reset(mcp_token)

        _runtime.reset(token_runtime)
