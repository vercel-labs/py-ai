"""AI Gateway v4 adapter-owned request parameter types."""

from __future__ import annotations

from typing import Any, Literal

import pydantic
from pydantic.alias_generators import to_camel

from ... import types

_CONFIG_MODEL = pydantic.ConfigDict(
    extra="allow",
    frozen=True,
    populate_by_name=True,
    alias_generator=to_camel,
)


class AutoToolChoice(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    type: Literal["auto"] = "auto"


class NoneToolChoice(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    type: Literal["none"] = "none"


class RequiredToolChoice(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    type: Literal["required"] = "required"


class NamedToolChoice(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    type: Literal["tool"] = "tool"
    tool_name: str


type ToolChoice = AutoToolChoice | NoneToolChoice | RequiredToolChoice | NamedToolChoice


class LanguageParams(pydantic.BaseModel):
    """Typed v4 ``LanguageModelV4CallOptions`` knobs.

    ``prompt`` and ``tools`` come from the framework. ``responseFormat`` is
    set via the ``output_type=`` kwarg. HTTP headers belong on the
    :class:`~ai.Client`. Anything else can be passed through verbatim.
    """

    model_config = _CONFIG_MODEL

    max_output_tokens: int | None = None
    temperature: float | None = None
    stop_sequences: list[str] | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    reasoning: (
        Literal[
            "provider-default",
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
        ]
        | None
    ) = None
    tool_choice: ToolChoice | None = None
    include_raw_chunks: bool | None = None
    provider_options: dict[str, Any] | None = None


class GatewayFunctionToolArgs(types.tools.FunctionToolArgs):
    """v4-specific extensions to function tool declarations."""

    model_config = _CONFIG_MODEL

    input_examples: list[dict[str, Any]] | None = None
    strict: bool | None = None
    provider_options: dict[str, Any] | None = None


__all__ = [
    "AutoToolChoice",
    "GatewayFunctionToolArgs",
    "LanguageParams",
    "NamedToolChoice",
    "NoneToolChoice",
    "RequiredToolChoice",
    "ToolChoice",
]
