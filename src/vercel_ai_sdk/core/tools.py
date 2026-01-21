from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Awaitable
from typing import Any, Callable, get_args, get_origin, get_type_hints, overload

import pydantic


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


@overload
def tool(fn: Callable[..., Awaitable[Any]]) -> Tool: ...


@overload
def tool(fn: None = None) -> Callable[[Callable[..., Awaitable[Any]]], Tool]: ...


def tool(fn: Callable[..., Awaitable[Any]] | None = None) -> Tool | Callable[[Callable[..., Awaitable[Any]]], Tool]:
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
