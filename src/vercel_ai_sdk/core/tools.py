from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, get_type_hints, runtime_checkable

import pydantic

if TYPE_CHECKING:
    from . import runtime as runtime_

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


@runtime_checkable
class ToolLike(Protocol):
    """Anything the LLM layer can use as a tool definition."""

    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def param_schema(self) -> dict[str, Any]: ...


class ToolSchema(pydantic.BaseModel):
    """What the LLM sees: name, description, and JSON Schema for parameters."""

    name: str
    description: str
    param_schema: dict[str, Any]
    return_type: Any


class Tool[**P, R]:
    def __init__(
        self,
        fn: Callable[P, Awaitable[R]],
        schema: ToolSchema,
        validator: type[pydantic.BaseModel] | None = None,
    ) -> None:
        self._fn = fn
        self._validator = validator
        self.schema = schema

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return await self._fn(*args, **kwargs)

    async def validate_and_call(
        self, json_str: str, runtime: runtime_.Runtime | None
    ) -> R:
        from .runtime import _find_runtime_param

        kwargs = json.loads(json_str) if json_str else {}

        if runtime and (rt_param := _find_runtime_param(self._fn)):
            kwargs[rt_param] = runtime

        # validate llm-generated inputs (skipped for MCP tools)
        if self._validator is not None:
            self._validator.model_validate(kwargs)
        return await self._fn(**kwargs)  # type: ignore[arg-type]

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

    fields = {}

    for param_name, param in sig.parameters.items():
        param_type = hints.get(param_name, str)

        if _is_runtime_type(param_type):
            continue
        if param.default is inspect.Parameter.empty:
            fields[param_name] = (param_type, ...)
        else:
            fields[param_name] = (param_type, param.default)

    validator = pydantic.create_model(f"{fn.__name__}_Args", **fields)

    # 2. instantiate the tool

    schema = ToolSchema(
        name=fn.__name__,
        description=inspect.getdoc(fn) or "",
        param_schema=validator.model_json_schema(),
        return_type=hints.get("return", None),
    )

    t = Tool(fn=fn, schema=schema, validator=validator)

    # 3. register in global registry
    _tool_registry[t.name] = t
    return t
