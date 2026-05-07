"""Anthropic provider-executed tools."""

from __future__ import annotations

from typing import ClassVar, Literal

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


class AnthropicProviderArgs(pydantic.BaseModel):
    """Base for Anthropic provider-executed tool args."""

    model_config = _CONFIG_MODEL

    anthropic_type: ClassVar[str]
    anthropic_beta: ClassVar[str | None] = None


class WebSearchArgs(AnthropicProviderArgs):
    anthropic_type: ClassVar[str] = "web_search_20260209"
    anthropic_beta: ClassVar[str | None] = "code-execution-web-tools-2026-02-09"

    model_config = _CONFIG_MODEL

    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    user_location: UserLocation | None = None


class WebFetchArgs(AnthropicProviderArgs):
    anthropic_type: ClassVar[str] = "web_fetch_20260209"
    anthropic_beta: ClassVar[str | None] = "code-execution-web-tools-2026-02-09"

    model_config = _CONFIG_MODEL

    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    citations: Citations | None = None
    max_content_tokens: int | None = None


class CodeExecutionArgs(AnthropicProviderArgs):
    anthropic_type: ClassVar[str] = "code_execution_20260120"

    model_config = _CONFIG_MODEL


class ComputerUseArgs(AnthropicProviderArgs):
    anthropic_type: ClassVar[str] = "computer_20251124"
    anthropic_beta: ClassVar[str | None] = "computer-use-2025-11-24"

    model_config = _CONFIG_MODEL

    display_width_px: int
    display_height_px: int
    display_number: int | None = None
    enable_zoom: bool | None = None


class TextEditorArgs(AnthropicProviderArgs):
    anthropic_type: ClassVar[str] = "text_editor_20250728"

    model_config = _CONFIG_MODEL

    max_characters: int | None = None


class BashArgs(AnthropicProviderArgs):
    anthropic_type: ClassVar[str] = "bash_20250124"
    anthropic_beta: ClassVar[str | None] = "computer-use-2025-01-24"

    model_config = _CONFIG_MODEL


class MemoryArgs(AnthropicProviderArgs):
    anthropic_type: ClassVar[str] = "memory_20250818"
    anthropic_beta: ClassVar[str | None] = "context-management-2025-06-27"

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


def web_search(
    *,
    max_uses: int | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    user_location: UserLocation | None = None,
) -> types.tools.Tool:
    args = WebSearchArgs(
        max_uses=max_uses,
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
        user_location=user_location,
    )
    _check_domains("web_search", args.allowed_domains, args.blocked_domains)
    return types.tools.Tool(
        kind="provider",
        name="web_search",
        args=args,
    )


def web_fetch(
    *,
    max_uses: int | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    citations: Citations | bool | None = None,
    max_content_tokens: int | None = None,
) -> types.tools.Tool:
    args = WebFetchArgs(
        max_uses=max_uses,
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
        citations=Citations(enabled=citations)
        if isinstance(citations, bool)
        else citations,
        max_content_tokens=max_content_tokens,
    )
    _check_domains("web_fetch", args.allowed_domains, args.blocked_domains)
    return types.tools.Tool(
        kind="provider",
        name="web_fetch",
        args=args,
    )


def code_execution() -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="code_execution",
        args=CodeExecutionArgs(),
    )


def computer_use(
    *,
    display_width_px: int,
    display_height_px: int,
    display_number: int | None = None,
    enable_zoom: bool | None = None,
) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="computer",
        args=ComputerUseArgs(
            display_width_px=display_width_px,
            display_height_px=display_height_px,
            display_number=display_number,
            enable_zoom=enable_zoom,
        ),
    )


def text_editor(*, max_characters: int | None = None) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="str_replace_based_edit_tool",
        args=TextEditorArgs(max_characters=max_characters),
    )


def bash() -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="bash",
        args=BashArgs(),
    )


def memory() -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="memory",
        args=MemoryArgs(),
    )


__all__ = [
    "AnthropicProviderArgs",
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
