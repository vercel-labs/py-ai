from __future__ import annotations

import abc
import asyncio
import contextvars
import json
from collections.abc import AsyncGenerator, Awaitable, Coroutine, Generator
from typing import Any, Callable

from .. import mcp
from . import messages as messages_
from . import tools as tools_


class LanguageModel(abc.ABC):
    @abc.abstractmethod
    async def stream(
        self, messages: list[messages_.Message], tools: list[tools_.Tool] | None = None
    ) -> AsyncGenerator[messages_.Message, None]:
        raise NotImplementedError
        yield


class Runtime:
    class _Sentinel:
        pass

    _SENTINEL = _Sentinel()

    def __init__(self) -> None:
        self._queue: asyncio.Queue[messages_.Message | Runtime._Sentinel] = (
            asyncio.Queue()
        )

    async def put(self, message: messages_.Message) -> None:
        await self._queue.put(message)

    async def get(self) -> messages_.Message | Runtime._Sentinel:
        return await self._queue.get()

    async def signal_done(self) -> None:
        await self._queue.put(self._SENTINEL)


_runtime: contextvars.ContextVar[Runtime] = contextvars.ContextVar("runtime")


async def _do_stream_text(
    llm: LanguageModel, messages: list[messages_.Message], label: str | None = None
) -> AsyncGenerator[messages_.Message]:
    runtime = _runtime.get()
    if runtime is None:
        raise ValueError("Runtime not set")

    async for message in llm.stream(messages=messages):
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


def stream_text(
    llm: LanguageModel, messages: list[messages_.Message], label: str | None = None
) -> Stream:
    return Stream(_do_stream_text(llm, messages, label))


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


async def execute(
    root: Callable[..., Coroutine[Any, Any, None]], *args: Any
) -> AsyncGenerator[messages_.Message]:
    runtime = Runtime()
    token_runtime = _runtime.set(runtime)

    mcp_pool: dict[str, mcp.client._Connection] = {}
    mcp_token = mcp.client._pool.set(mcp_pool)

    try:
        async with asyncio.TaskGroup() as tg:
            _task: asyncio.Task[None] = tg.create_task(
                _stop_when_done(runtime, root(*args))
            )

            while True:
                msg = await runtime.get()
                if isinstance(msg, Runtime._Sentinel):
                    break
                yield msg

    finally:
        if mcp_token is not None:
            await mcp.client.close_connections()
            mcp.client._pool.reset(mcp_token)

        _runtime.reset(token_runtime)
