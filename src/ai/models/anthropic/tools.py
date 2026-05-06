"""Anthropic provider-executed tools."""

from __future__ import annotations

from typing import Literal

import pydantic
from pydantic.alias_generators import to_camel

from ... import types

_CONFIG_MODEL = pydantic.ConfigDict(
    frozen=True,
    populate_by_name=True,
    alias_generator=to_camel,
)


class UserLocation(pydantic.BaseModel):
    """Approximate user location for geographically relevant search results."""

    model_config = _CONFIG_MODEL

    type: Literal["approximate"] = "approximate"
    city: str | None = None
    region: str | None = None
    country: str | None = None
    timezone: str | None = None


class Citations(pydantic.BaseModel):
    """Citation configuration for web fetch."""

    model_config = _CONFIG_MODEL

    enabled: bool


class WebSearchArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    user_location: UserLocation | None = None


class WebFetchArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    citations: Citations | None = None
    max_content_tokens: int | None = None


class CodeExecutionArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL


class ComputerUseArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    display_width_px: int
    display_height_px: int
    display_number: int | None = None
    enable_zoom: bool | None = None


class TextEditorArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    max_characters: int | None = None


class BashArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL


class MemoryArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL


def _check_domains(
    tool_name: str,
    allowed_domains: list[str] | None,
    blocked_domains: list[str] | None,
) -> None:
    if allowed_domains and blocked_domains:
        raise ValueError(
            f"anthropic.{tool_name}: pass only one of "
            "`allowed_domains` or `blocked_domains`"
        )


def web_search(args: WebSearchArgs) -> types.tools.Tool:
    _check_domains("web_search", args.allowed_domains, args.blocked_domains)
    return types.tools.Tool(
        kind="provider",
        name="web_search",
        args=args,
    )


def web_fetch(args: WebFetchArgs) -> types.tools.Tool:
    _check_domains("web_fetch", args.allowed_domains, args.blocked_domains)
    return types.tools.Tool(
        kind="provider",
        name="web_fetch",
        args=args,
    )


def code_execution(args: CodeExecutionArgs) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="code_execution",
        args=args,
    )


def computer_use(args: ComputerUseArgs) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="computer",
        args=args,
    )


def text_editor(args: TextEditorArgs) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="str_replace_based_edit_tool",
        args=args,
    )


def bash(args: BashArgs) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="bash",
        args=args,
    )


def memory(args: MemoryArgs) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="memory",
        args=args,
    )


__all__ = [
    "BashArgs",
    "Citations",
    "CodeExecutionArgs",
    "ComputerUseArgs",
    "MemoryArgs",
    "TextEditorArgs",
    "UserLocation",
    "WebFetchArgs",
    "WebSearchArgs",
    "bash",
    "code_execution",
    "computer_use",
    "memory",
    "text_editor",
    "web_fetch",
    "web_search",
]
