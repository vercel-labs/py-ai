"""Tool schema types — what the LLM layer sees.

These are schema-only definitions used by LanguageModel.stream(tools=...).
The executable Tool class and @tool decorator live in agents.tools.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pydantic


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
