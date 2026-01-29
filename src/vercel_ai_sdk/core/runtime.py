from __future__ import annotations

import abc
import asyncio
import contextvars
import json
from collections.abc import AsyncGenerator, Awaitable, Coroutine, Generator
from typing import Any, Callable, get_type_hints

from .. import mcp
from . import messages as messages_
from . import step as step_
from . import tools as tools_


class LanguageModel(abc.ABC):
    @abc.abstractmethod
    async def stream(
        self, messages: list[messages_.Message], tools: list[tools_.Tool] | None = None
    ) -> AsyncGenerator[messages_.Message, None]:
        raise NotImplementedError
        yield


class Runtime:
    """
    Step-based execution runtime.
    
    Functions decorated with @stream submit step functions to the queue.
    execute() pulls steps, runs them, yields messages, and resolves futures.
    """

    class _Sentinel:
        pass

    _SENTINEL = _Sentinel()

    def __init__(self) -> None:
        # Step queue: (step_fn, future) pairs
        self._step_queue: asyncio.Queue[
            tuple[step_.StepFn, asyncio.Future[step_.StepResult]] | Runtime._Sentinel
        ] = asyncio.Queue()
        
        # Message queue for backward compatibility / direct puts
        self._message_queue: asyncio.Queue[messages_.Message] = asyncio.Queue()

    async def submit_step(
        self, step_fn: step_.StepFn, future: asyncio.Future[step_.StepResult]
    ) -> None:
        """Submit a step function to be executed by execute()."""
        await self._step_queue.put((step_fn, future))

    async def get_step(
        self,
    ) -> tuple[step_.StepFn, asyncio.Future[step_.StepResult]] | _Sentinel:
        """Get next step from queue (called by execute())."""
        return await self._step_queue.get()

    async def signal_done(self) -> None:
        """Signal that no more steps will be submitted."""
        await self._step_queue.put(self._SENTINEL)

    # Backward compatibility: direct message queue
    async def put(self, message: messages_.Message) -> None:
        await self._message_queue.put(message)

    async def get(self) -> messages_.Message:
        return await self._message_queue.get()

    def message_queue_empty(self) -> bool:
        return self._message_queue.empty()


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


async def _do_stream_step(
    llm: LanguageModel,
    messages: list[messages_.Message],
    tools: list[tools_.Tool] | None = None,
    label: str | None = None,
) -> AsyncGenerator[messages_.Message]:
    runtime = _runtime.get()
    if runtime is None:
        raise ValueError("Runtime not set")

    async for message in llm.stream(messages=messages, tools=tools):
        message.label = label
        await runtime.put(message)
        yield message


async def _do_stream_loop(
    llm: LanguageModel,
    messages: list[messages_.Message],
    tools: list[tools_.Tool],
    label: str | None = None,
) -> AsyncGenerator[messages_.Message]:
    runtime = _runtime.get()
    if runtime is None:
        raise ValueError("Runtime not set")

    while True:
        tool_calls = []
        assistant_msg = None

        async for message in llm.stream(messages=messages, tools=tools):
            message.label = label
            await runtime.put(message)
            yield message

            if message.is_done:
                assistant_msg = message
                for part in message.parts:
                    if isinstance(part, messages_.ToolPart):
                        tool_calls.append(part)

        if assistant_msg:
            messages.append(assistant_msg)

        if not tool_calls:
            break

        for tool_call in tool_calls:
            tool_fn = next(t for t in tools if t.name == tool_call.tool_name)
            args = json.loads(tool_call.tool_args)

            # Inject runtime if the tool has a Runtime-typed parameter
            if runtime_param := _find_runtime_param(tool_fn.fn):
                args[runtime_param] = runtime

            result = await tool_fn.fn(**args)

            assert assistant_msg is not None, "Assistant message not found"

            tool_part = assistant_msg.get_tool_part(tool_call.tool_call_id)

            assert tool_part is not None, (
                f"Tool part not found for tool call {tool_call.tool_call_id}"
            )

            tool_part.status = "result"
            tool_part.result = result

            await runtime.put(assistant_msg)
            yield assistant_msg


class Stream:
    def __init__(self, generator: AsyncGenerator[messages_.Message, None]) -> None:
        self._generator = generator
        self._messages: list[messages_.Message] = []
        self._is_consumed: bool = False

    async def __aiter__(self) -> AsyncGenerator[messages_.Message, None]:
        async for message in self._generator:
            if message.is_done:
                # upsert the message
                # tool results get added to the same message that contains the call
                existing_idx = next(
                    (i for i, m in enumerate(self._messages) if m.id == message.id),
                    None,
                )
                if existing_idx is not None:
                    self._messages[existing_idx] = message
            else:
                self._messages.append(message)
            yield message
        self._is_consumed = True

    def __await__(self) -> Generator[Any, None, list[messages_.Message]]:
        return self._collect().__await__()

    async def _collect(self) -> list[messages_.Message]:
        if self._is_consumed:
            return self._messages
        async for _ in self:
            pass
        return self._messages

    @property
    def result(self) -> list[messages_.Message]:
        if not self._is_consumed:
            raise ValueError("Stream is not consumed")
        return self._messages


def stream_step(
    llm: LanguageModel,
    messages: list[messages_.Message],
    tools: list[tools_.Tool] | None = None,
    label: str | None = None,
) -> Stream:
    return Stream(_do_stream_step(llm, messages, tools, label))


def stream_loop(
    llm: LanguageModel,
    messages: list[messages_.Message],
    tools: list[tools_.Tool],
    label: str | None = None,
) -> Stream:
    return Stream(_do_stream_loop(llm, messages, tools, label))


async def _stop_when_done(runtime: Runtime, task: Awaitable[None]) -> None:
    try:
        await task
    finally:
        await runtime.signal_done()


def _extract_tool_calls(message: messages_.Message) -> list[step_.ToolCall]:
    """Extract tool calls from a completed message."""
    tool_calls = []
    for part in message.parts:
        if isinstance(part, messages_.ToolPart):
            tool_calls.append(
                step_.ToolCall(
                    tool_call_id=part.tool_call_id,
                    tool_name=part.tool_name,
                    tool_args=json.loads(part.tool_args)
                    if isinstance(part.tool_args, str)
                    else part.tool_args,
                )
            )
    return tool_calls


async def execute(
    root: Callable[..., Coroutine[Any, Any, None]], *args: Any
) -> AsyncGenerator[messages_.Message, None]:
    """
    Execute an agent function, yielding messages as they stream.
    
    This is the central executor that:
    1. Starts the user's agent function as a background task
    2. Pulls steps from the Runtime queue
    3. Executes each step, yielding messages
    4. Resolves futures to unblock user code
    """
    runtime = Runtime()
    token_runtime = _runtime.set(runtime)

    mcp_pool: dict[str, mcp.client._Connection] = {}
    mcp_token = mcp.client._pool.set(mcp_pool)

    # Inject runtime as keyword arg if the function has a Runtime-typed parameter
    kwargs: dict[str, Any] = {}
    if runtime_param := _find_runtime_param(root):
        kwargs[runtime_param] = runtime

    try:
        async with asyncio.TaskGroup() as tg:
            _task: asyncio.Task[None] = tg.create_task(
                _stop_when_done(runtime, root(*args, **kwargs))
            )

            while True:
                item = await runtime.get_step()
                
                if isinstance(item, Runtime._Sentinel):
                    break
                
                step_fn, future = item
                
                # Execute the step and collect results
                result_messages: list[messages_.Message] = []
                last_message: messages_.Message | None = None
                
                async for msg in step_fn():
                    yield msg
                    result_messages.append(msg)
                    if msg.is_done:
                        last_message = msg
                
                # Build StepResult
                tool_calls = (
                    _extract_tool_calls(last_message) if last_message else []
                )
                step_result = step_.StepResult(
                    messages=result_messages,
                    tool_calls=tool_calls,
                )
                
                # Resolve future to unblock user code
                future.set_result(step_result)

    finally:
        if mcp_token is not None:
            await mcp.client.close_connections()
            mcp.client._pool.reset(mcp_token)

        _runtime.reset(token_runtime)
