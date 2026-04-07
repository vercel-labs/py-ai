"""Tool decorator, Tool class, and ToolCall callable wrapper."""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, get_type_hints, overload

import pydantic

from ..types import messages as messages_
from ..types.tools import ToolLike as ToolLike
from ..types.tools import ToolSchema as ToolSchema
from . import runtime


class Tool[**P, R]:
    def __init__(
        self,
        fn: Callable[P, Awaitable[R]],
        schema: ToolSchema,
        validator: type[pydantic.BaseModel] | None = None,
    ) -> None:
        self._fn = fn
        self._is_gen = inspect.isasyncgenfunction(fn)
        self._validator = validator
        self.schema = schema

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return await self._fn(*args, **kwargs)

    async def validate_and_call(self, json_str: str) -> Any:
        kwargs = json.loads(json_str) if json_str else {}
        if self._validator is not None:
            self._validator.model_validate(kwargs)

        if self._is_gen:
            return await self._drain_generator(kwargs)
        return await self._fn(**kwargs)  # type: ignore[call-arg]

    async def _drain_generator(self, kwargs: dict[str, Any]) -> Any:
        sink = runtime.get_sink()
        final: Any = None
        gen = self._fn(**kwargs)  # type: ignore[call-arg]
        async for msg in gen:  # type: ignore[attr-defined]
            final = msg
            if sink is not None:
                await sink.put(msg)
        return final

    @property
    def name(self) -> str:
        return self.schema.name

    @property
    def description(self) -> str:
        return self.schema.description

    @property
    def param_schema(self) -> dict[str, Any]:
        return self.schema.param_schema


@overload
def tool[**P, R](fn: Callable[P, Awaitable[R]]) -> Tool[P, R]: ...
@overload
def tool[**P, R](fn: Callable[P, AsyncGenerator[R]]) -> Tool[P, R]: ...


def tool[**P, R](fn: Callable[P, Any]) -> Tool[P, R]:
    """Decorator to define a tool from an async function or async generator."""
    sig = inspect.signature(fn)
    hints = get_type_hints(fn) if hasattr(fn, "__annotations__") else {}

    fields: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        param_type = hints.get(param_name, str)
        if param.default is inspect.Parameter.empty:
            fields[param_name] = (param_type, ...)
        else:
            fields[param_name] = (param_type, param.default)

    validator = pydantic.create_model(f"{fn.__name__}_Args", **fields)

    schema = ToolSchema(
        name=fn.__name__,
        description=inspect.getdoc(fn) or "",
        param_schema=validator.model_json_schema(),
        return_type=hints.get("return", None),
    )

    return Tool(fn=fn, schema=schema, validator=validator)


class ToolCall:
    """Callable bridge between a ToolPart (data from model) and a Tool (executable)."""

    def __init__(self, part: messages_.ToolPart, tool: Tool[..., Any]) -> None:
        self._part = part
        self._tool = tool

    @property
    def id(self) -> str:
        return self._part.tool_call_id

    @property
    def name(self) -> str:
        return self._part.tool_name

    @property
    def args(self) -> str:
        return self._part.tool_args

    async def __call__(self) -> messages_.Message:
        result = await self._tool.validate_and_call(self._part.tool_args)
        updated_part = self._part.with_result(result)
        return messages_.Message(role="assistant", parts=[updated_part])
