"""Context — everything the LLM sees during a run.

Consolidates tool registry, system prompt, message history, and model
reference into a single, serializable object.  Independent of execution
machinery (Runtime) — can be constructed, inspected, and serialized
without starting a run.

The context is stashed in a contextvar during ``run()`` so that
framework internals (``execute_tool``, MCP client, etc.) can access it.
"""

from __future__ import annotations

import contextvars
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pydantic

from ..types import messages as messages_

if TYPE_CHECKING:
    from . import tools as tools_


# ── ToolSource ────────────────────────────────────────────────────


class ToolSource(pydantic.BaseModel):
    """Provenance info for a tool — how to find or reconstruct it.

    Carries enough information to locate the code behind a tool,
    whether it's a decorated Python function or an MCP server.

    Attributes:
        kind: ``"python"``, ``"mcp_stdio"``, or ``"mcp_http"``.
        module: Python module path, e.g. ``"myapp.tools"``.
        qualname: Qualified name, e.g. ``"get_weather"``.
        uri: Remote URL for HTTP-based MCP servers.
        server_command: Launch command for stdio MCP servers.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    kind: str
    module: str | None = None
    qualname: str | None = None
    uri: str | None = None
    server_command: str | None = None


# ── Context ───────────────────────────────────────────────────────


class Context(pydantic.BaseModel):
    """Everything the LLM sees: tools, system prompt, messages, model.

    Independent of execution machinery (Runtime).  Constructable by the
    user or auto-constructed by ``run()``.

    Usage::

        ctx = Context(
            system_prompt="You are a helpful assistant.",
            tools=[get_weather, get_population],
        )
        ctx.get_tool("get_weather")   # look up by name
        data = ctx.model_dump()       # serializable snapshot
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    model: Any = None
    system_prompt: str = ""
    messages: list[messages_.Message] = pydantic.Field(default_factory=list)

    _tools: dict[str, tools_.Tool[..., Any]] = pydantic.PrivateAttr(
        default_factory=dict
    )

    def __init__(
        self,
        *,
        tools: Sequence[tools_.Tool[..., Any]] | None = None,
        **data: Any,
    ) -> None:
        super().__init__(**data)
        if tools:
            for t in tools:
                self.register_tool(t)

    # ── Tool registry (scoped to this context) ────────────────

    def register_tool(self, tool: tools_.Tool[..., Any]) -> None:
        """Register a tool in this context's scoped registry."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> tools_.Tool[..., Any] | None:
        """Look up a tool by name.  Returns ``None`` if not found."""
        return self._tools.get(name)

    @property
    def tools(self) -> list[tools_.Tool[..., Any]]:
        """All tools registered in this context."""
        return list(self._tools.values())

    @property
    def tool_schemas(self) -> list[tools_.ToolSchema]:
        """Tool schemas — what gets sent to the LLM."""
        return [t.schema for t in self._tools.values()]

    # ── Serialization ─────────────────────────────────────────

    @pydantic.model_serializer
    def _serialize(self) -> dict[str, Any]:
        """Serialize including tool schemas and sources.

        Tool code is not serialized — only schemas and source
        references.
        """
        return {
            "system_prompt": self.system_prompt,
            "messages": [m.model_dump() for m in self.messages],
            "tools": [
                {
                    "schema": t.schema.model_dump(),
                    "source": (t.source.model_dump() if t.source is not None else None),
                }
                for t in self._tools.values()
            ],
        }

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _validate(
        cls,
        data: Any,
        handler: pydantic.ValidatorFunctionWrapHandler,
    ) -> Context:
        """Reconstruct from serialized form or pass through normal init.

        When deserializing, tools are schema-only (not executable)
        unless their sources can be resolved from the global registry.
        """
        # Normal construction (already a Context, or keyword args without
        # a ``tools`` key that looks like serialized tool dicts).
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict) or "tools" not in data:
            result: Context = handler(data)
            return result

        # Check whether tools contains serialized dicts (from model_dump)
        # vs. live Tool objects (from normal __init__).
        tools_value = data["tools"]
        if tools_value and isinstance(tools_value[0], dict):
            return cls._from_serialized(data)

        # Live Tool objects — let the normal init path handle it.
        result = handler(data)
        return result

    @classmethod
    def _from_serialized(cls, data: dict[str, Any]) -> Context:
        """Reconstruct from ``model_dump()`` output."""
        from . import tools as tools_

        ctx = cls(
            system_prompt=data.get("system_prompt", ""),
            messages=[
                messages_.Message.model_validate(m) for m in data.get("messages", [])
            ],
        )

        for tool_data in data.get("tools", []):
            schema = tools_.ToolSchema.model_validate(tool_data["schema"])
            source_data = tool_data.get("source")
            source = ToolSource(**source_data) if source_data else None

            # Try to resolve the tool from the global registry
            live_tool = tools_.get_tool(schema.name)
            if live_tool is not None:
                ctx.register_tool(live_tool)
            else:
                # Schema-only placeholder — inspectable but not executable
                placeholder = tools_.Tool(
                    fn=tools_._unresolvable_tool_fn(schema.name),
                    schema=schema,
                    source=source,
                )
                ctx.register_tool(placeholder)

        return ctx


# ── Contextvar ────────────────────────────────────────────────────

_context: contextvars.ContextVar[Context] = contextvars.ContextVar("context")


def get_context() -> Context:
    """Get the active Context from the current run.

    Raises ``LookupError`` if called outside of ``ai.run()``.
    """
    return _context.get()
