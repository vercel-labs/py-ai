from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import (
    Any,
    get_args,
    get_origin,
    get_type_hints,
)

import pydantic

# Module-level tool registry - populated at decoration time
_tool_registry: dict[str, Tool] = {}


def get_tool(name: str) -> Tool | None:
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


class ToolSchema(pydantic.BaseModel):
    """What the LLM sees: name, description, and JSON Schema for parameters."""

    name: str
    description: str
    param_schema: dict[str, Any]
    return_type: Any


class Tool[**P, R]:
    def __init__(self, fn: Callable[P, Awaitable[R]], schema: ToolSchema) -> None:
        self._fn = fn
        self.schema = schema

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return await self._fn(*args, **kwargs)

    @property
    def name(self) -> str:
        return self.schema.name

    @property
    def description(self) -> str:
        return self.schema.description

    @property
    def param_schema(self) -> dict[str, Any]:
        return self.schema.param_schema


def tool[**P, R](fn: Callable[P, Awaitable[R]]) -> Tool[P, R]:
    """Decorator to define a tool from an async function."""

    # 1. build tool schema by parsing the function
    sig = inspect.signature(fn)
    hints = get_type_hints(fn) if hasattr(fn, "__annotations__") else {}

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        param_type = hints.get(param_name, str)

        # Skip Runtime-typed parameters - they're injected, not from LLM
        if _is_runtime_type(param_type):
            continue

        properties[param_name] = _get_param_schema(param_type)

        if param.default is inspect.Parameter.empty and not _is_optional(param_type):
            required.append(param_name)

    parameters = {
        "type": "object",
        "properties": properties,
    }

    if required:
        parameters["required"] = required

    # 2. instantiate the tool

    schema = ToolSchema(
        name=fn.__name__,
        description=inspect.getdoc(fn) or "",
        param_schema=parameters,
        return_type=hints.get("return", None),
    )

    t = Tool(fn=fn, schema=schema)

    # Register in global registry
    _tool_registry[t.name] = t
    return t
