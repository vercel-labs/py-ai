"""Tool schema types — what the LLM layer sees."""

from typing import Any, Protocol, runtime_checkable

import pydantic
from pydantic.alias_generators import to_camel


@runtime_checkable
class ToolSchemaLike(Protocol):
    """Anything that exposes a tool schema to the LLM layer.

    Structural type: ``name``, ``description``, ``param_schema``.
    Satisfied by both :class:`ToolSchema` and the agents' ``Tool`` class.
    """

    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def param_schema(self) -> dict[str, Any]: ...


class ToolApproval(pydantic.BaseModel):
    """Payload schema for tool-approval hooks.

    Usage inside a loop::

        approval = await hook(
            f"approve_{tc.tool_call_id}",
            payload=ToolApproval,
            metadata={"tool_name": tc.tool_name},
            interrupt_loop=True,
        )
        if approval.granted:
            ...
    """

    granted: bool
    reason: str | None = None


class ToolSchema(pydantic.BaseModel):
    """What the LLM sees: name, description, and JSON Schema for parameters."""

    name: str
    description: str
    param_schema: dict[str, Any]
    return_type: Any


class BuiltinTool(pydantic.BaseModel):
    """Base for provider-executed built-in tools.

    All concrete subclasses (and any nested config models they expose)
    automatically get camelCase aliases for snake_case Python fields, so:

    * users can construct with snake_case kwargs (Pythonic),
    * ``model_dump()`` (default) emits snake_case (e.g. for the native
      Anthropic API which uses snake_case),
    * ``model_dump(by_alias=True)`` emits camelCase (e.g. for the AI
      Gateway v3 wire which uses camelCase).

    Nested ``BaseModel`` config types are expected to inherit
    ``BuiltinToolConfig`` (or set the same ``alias_generator`` /
    ``populate_by_name``) so aliasing works recursively.
    """

    model_config = pydantic.ConfigDict(
        frozen=True,
        populate_by_name=True,
        alias_generator=to_camel,
    )

    @property
    def name(self) -> str:
        """Tool name.

        Provider-specific subclasses must override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override the `name` property"
        )


# Anything the LLM layer can use as a tool: either a host-executed
# function tool (described by a schema) or a provider-executed built-in
# tool subclass.
type ToolLike = ToolSchemaLike | BuiltinTool


class BuiltinToolConfig(pydantic.BaseModel):
    """Base for nested config models used inside :class:`BuiltinTool` fields.

    Carries the same alias-generator config so ``model_dump(by_alias=True)``
    on the parent recursively emits camelCase.
    """

    model_config = pydantic.ConfigDict(
        frozen=True,
        populate_by_name=True,
        alias_generator=to_camel,
    )
