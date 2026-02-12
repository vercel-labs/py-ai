from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Awaitable
from typing import Any, Callable, get_args, get_origin, get_type_hints, overload

import pydantic

# Module-level tool registry - populated at decoration time
_tool_registry: dict[str, "Tool"] = {}


def get_tool(name: str) -> "Tool | None":
    """Look up a tool by name from the global registry."""
    return _tool_registry.get(name)


def _is_runtime_type(hint: Any) -> bool:
    """Check if a type hint is the Runtime class."""
    # Import here to avoid circular import at runtime
    from .runtime import Runtime

    return hint is Runtime


def _get_param_schema(param_type: type) -> dict[str, Any]:
    """Get JSON schema for a Python type using Pydantic's TypeAdapter."""
    schema = pydantic.TypeAdapter(param_type).json_schema()
    if "$defs" in schema and len(schema.get("$defs", {})) == 0:
        del schema["$defs"]
    return schema


def _is_optional(param_type: type) -> bool:
    """Check if a type is Optional (Union with None)."""
    origin = get_origin(param_type)
    if origin is not None:
        args = get_args(param_type)
        return type(None) in args
    return False


@overload
def tool(fn: Callable[..., Awaitable[Any]]) -> Tool: ...


@overload
def tool(fn: None = None) -> Callable[[Callable[..., Awaitable[Any]]], Tool]: ...


def tool(
    fn: Callable[..., Awaitable[Any]] | None = None,
) -> Tool | Callable[[Callable[..., Awaitable[Any]]], Tool]:
    """Decorator to define a tool from an async function."""

    def make_tool(f: Callable[..., Awaitable[Any]]) -> Tool:
        sig = inspect.signature(f)
        hints = get_type_hints(f) if hasattr(f, "__annotations__") else {}

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            param_type = hints.get(param_name, str)

            # Skip Runtime-typed parameters - they're injected, not from LLM
            if _is_runtime_type(param_type):
                continue
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

        t = Tool(
            name=f.__name__,
            description=inspect.getdoc(f) or "",
            schema=parameters,
            fn=f,
        )
        # Register in global registry
        _tool_registry[t.name] = t
        return t

    if fn is not None:
        return make_tool(fn)
    return make_tool


@dataclasses.dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    fn: Callable[..., Awaitable[Any]]
