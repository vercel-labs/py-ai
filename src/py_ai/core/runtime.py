from __future__ import annotations

import abc
import asyncio
import dataclasses
import inspect
import json
from collections.abc import AsyncGenerator, Awaitable, Coroutine, Generator
import contextvars
from typing import Any, Literal, Callable, get_origin, get_args, get_type_hints
import uuid

import pydantic

from .. import mcp

# tool introspection that uses pydantic to figure out the correct way
# to json-serialize arguments so that the tool description can be passed
# to the LLM
def _get_param_schema(param_type: type) -> dict[str, Any]:
    """Get JSON schema for a Python type using Pydantic's TypeAdapter."""
    schema = pydantic.TypeAdapter(param_type).json_schema()
    if "$defs" in schema and len(schema.get("$defs", {})) == 0:
        del schema["$defs"]
    return schema


def _is_optional(param_type: type) -> bool:
    """Check if a type is Optional (Union with None)."""
    origin = get_origin(param_type)
    if origin is type(None):
        return True
    if origin is not None:
        args = get_args(param_type)
        return type(None) in args
    return False


def tool(fn: Callable[..., Awaitable[Any]] | None = None):
    """Decorator to define a tool from an async function."""

    def make_tool(f: Callable[..., Awaitable[Any]]) -> Tool:
        sig = inspect.signature(f)
        hints = get_type_hints(f) if hasattr(f, "__annotations__") else {}

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            # Skip 'runtime' - it's injected, not from LLM
            if param_name == "runtime":
                continue

            param_type = hints.get(param_name, str)
            properties[param_name] = _get_param_schema(param_type)

            if param.default is inspect.Parameter.empty and not _is_optional(
                param_type
            ):
                required.append(param_name)

        parameters = {
            "type": "object",
            "properties": properties,
        }
        if required:
            parameters["required"] = required

        return Tool(
            name=f.__name__,
            description=inspect.getdoc(f) or "",
            parameters=parameters,
            fn=f,
        )

    if fn is not None:
        return make_tool(fn)
    return make_tool


@dataclasses.dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Awaitable[Any]]


# all data withing the framework gets normalized to this
# one message type that is made of of these parts
@dataclasses.dataclass
class TextPart:
    text: str
    type: Literal["text"] = "text"


@dataclasses.dataclass
class ToolCallPart:
    tool_call_id: str
    tool_name: str
    tool_args: str
    type: Literal["tool_call"] = "tool_call"


@dataclasses.dataclass
class ToolResultPart:
    tool_call_id: str
    result: dict[str, Any]
    type: Literal["tool_result"] = "tool_result"


@dataclasses.dataclass
class ReasoningPart:
    reasoning: str
    type: Literal["reasoning"] = "reasoning"
    # Anthropic's thinking blocks include a signature for cache/verification.
    # This must be preserved and sent back in multi-turn conversations.
    signature: str | None = None


Part = TextPart | ToolCallPart | ToolResultPart | ReasoningPart


def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclasses.dataclass
class ToolCallDelta:
    """Represents a streaming delta for tool call arguments."""
    tool_call_id: str
    tool_name: str
    args_delta: str


@dataclasses.dataclass
class Message:
    role: Literal["user", "assistant", "system", "tool"]
    parts: list[Part]
    id: str = dataclasses.field(default_factory=_gen_id)
    is_done: bool = False
    text_delta: str = ""
    reasoning_delta: str = ""
    tool_call_deltas: list[ToolCallDelta] = dataclasses.field(default_factory=list)
    label: str | None = None

    @property
    def text(self) -> str:
        for part in self.parts:
            if isinstance(part, TextPart):
                return part.text
        return ""

    @property
    def reasoning(self) -> str:
        for part in self.parts:
            if isinstance(part, ReasoningPart):
                return part.reasoning
        return ""


class LanguageModel(abc.ABC):
    @abc.abstractmethod
    async def stream(
        self, messages: list[Message], tools: list[Tool] | None = None
    ) -> AsyncGenerator[Message, None]:
        raise NotImplementedError
        yield


class Runtime:
    class _Sentinel:
        pass

    _SENTINEL = _Sentinel()

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Message | Runtime._Sentinel] = asyncio.Queue()

    async def put(self, message: Message) -> None:
        await self._queue.put(message)

    async def get(self) -> Message | Runtime._Sentinel:
        return await self._queue.get()

    async def signal_done(self) -> None:
        await self._queue.put(self._SENTINEL)


_runtime: contextvars.ContextVar[Runtime] = contextvars.ContextVar("runtime")


async def _do_stream_text(
    llm: LanguageModel, messages: list[Message], label: str | None = None
) -> AsyncGenerator[Message]:
    runtime = _runtime.get()
    if runtime is None:
        raise ValueError("Runtime not set")

    async for message in llm.stream(messages=messages):
        message.label = label
        await runtime.put(message)
        yield message


async def _do_stream_loop(
    llm: LanguageModel,
    messages: list[Message],
    tools: list[Tool],
    label: str | None = None,
) -> AsyncGenerator[Message]:
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
                    if isinstance(part, ToolCallPart):
                        tool_calls.append(part)

        if assistant_msg:
            messages.append(assistant_msg)

        if not tool_calls:
            break

        for tool_call in tool_calls:
            tool_fn = next(t for t in tools if t.name == tool_call.tool_name)
            args = json.loads(tool_call.tool_args)
            result = await tool_fn.fn(**args)

            tool_msg = Message(
                role="tool",
                parts=[
                    ToolResultPart(
                        tool_call_id=tool_call.tool_call_id,
                        result={"output": result},
                    )
                ],
                label=label,
            )
            messages.append(tool_msg)
            await runtime.put(tool_msg)
            yield tool_msg


class Stream:
    def __init__(self, generator: AsyncGenerator[Message, None]) -> None:
        self._generator = generator
        self._messages: list[Message] = []
        self._is_consumed: bool = False

    async def __aiter__(self) -> AsyncGenerator[Message, None]:
        async for message in self._generator:
            if message.is_done:
                self._messages.append(message)
            yield message
        self._is_consumed = True

    def __await__(self) -> Generator[Any, None, list[Message]]:
        return self._collect().__await__()

    async def _collect(self) -> list[Message]:
        if self._is_consumed:
            return self._messages
        async for _ in self:
            pass
        return self._messages

    @property
    def result(self) -> list[Message]:
        if not self._is_consumed:
            raise ValueError("Stream is not consumed")
        return self._messages


def stream_text(
    llm: LanguageModel, messages: list[Message], label: str | None = None
) -> Stream:
    return Stream(_do_stream_text(llm, messages, label))


def stream_loop(
    llm: LanguageModel,
    messages: list[Message],
    tools: list[Tool],
    label: str | None = None,
) -> Stream:
    return Stream(_do_stream_loop(llm, messages, tools, label))


async def _stop_when_done(runtime: Runtime, task: Awaitable[None]) -> None:
    try:
        await task
    finally:
        await runtime.signal_done()


async def execute(
    root: Callable[[Runtime], Coroutine[Any, Any, None]], *args: Any
) -> AsyncGenerator[Message]:
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

