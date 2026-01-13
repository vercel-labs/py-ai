from __future__ import annotations

import abc
import asyncio
import dataclasses
import inspect
from collections.abc import AsyncGenerator, Awaitable, Coroutine
from typing import Any, Literal, Callable, get_origin, get_args, get_type_hints
import uuid

import pydantic


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


Part = TextPart | ToolCallPart | ToolResultPart


def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclasses.dataclass
class Message:
    role: Literal["user", "assistant", "system", "tool"]
    parts: list[Part]
    id: str = dataclasses.field(default_factory=_gen_id)
    is_done: bool = False
    text_delta: str = ""
    label: str | None = None


class LanguageModel(abc.ABC):
    @abc.abstractmethod
    async def stream(
        self, messages: list[Message], tools: list[Tool] | None = None
    ) -> AsyncGenerator[Message, None]:
        raise NotImplementedError
        yield


class _Sentinel:
    pass


_SENTINEL = _Sentinel()


# owns message queue
# exposes .stream() that yields messages from queue
# is consumer and also transforms whatever is consumed into a generator
class Collector:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[Message | _Sentinel] = asyncio.Queue()
        self._done: bool = False

    async def stream(
        self, root: Callable[[Runtime], Awaitable[None]]
    ) -> AsyncGenerator[Message]:
        runtime = Runtime(self)

        async with asyncio.TaskGroup() as g:
            _ = g.create_task(self._stop_when_done(root(runtime)))

            while True:
                message = await self.queue.get()
                if isinstance(message, _Sentinel):
                    break
                yield message

    # checks if the task has quit and gracefully
    # shuts down the queue (why do we need this?)
    async def _stop_when_done(self, task: Awaitable[None]):
        try:
            await task
        finally:
            self.queue.put_nowait(_SENTINEL)  # FIXME: swap out for a proper sentinel

    def close(self) -> None:
        self._done = True
        self.queue.put_nowait(_SENTINEL)


# gets injected into tools so they can push to the Loop queue
# is produces
class Runtime:
    def __init__(self, collector: Collector) -> None:
        self._colletor: Collector = collector

    async def push(self, message: Message) -> None:
        await self._colletor.queue.put(message)


# defines the execution graph for the root
# calls the LLM, invokes tools, etc
def get_root(llm: LanguageModel, messages: list[Message], tools: list[Tool]):
    async def loop_fn(runtime: Runtime) -> None:
        import json

        while True:
            tool_calls = []
            assistant_msg = None

            async for message in llm.stream(messages=messages, tools=tools):
                await runtime.push(message)

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
                )
                messages.append(tool_msg)
                await runtime.push(tool_msg)

    return loop_fn
