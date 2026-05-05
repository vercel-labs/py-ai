"""Anthropic provider-executed (built-in) tools.

Each class subclasses :class:`ai.types.tools.BuiltinTool` with typed
configuration fields. The adapter dispatches on subclass identity
(``isinstance``) to convert each instance to Anthropic wire format.

Usage::

    from ai.models import anthropic

    tools = [
        anthropic.tools.web_search(max_uses=5),
        anthropic.tools.code_execution(),
    ]
    s = ai.stream(model, msgs, tools=tools)

We ship the latest stable variants of each tool. Older versions can be
added on demand. The version is captured in the ``type_`` ClassVar
on each subclass; adapters read these — users don't.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import pydantic

from ...types import tools as tools_

# ---------------------------------------------------------------------------
# Shared sub-types
# ---------------------------------------------------------------------------


class UserLocation(tools_.BuiltinToolConfig):
    """Approximate user location for geographically relevant search results.

    The ``type`` field defaults to ``"approximate"`` which is the only value
    the Anthropic API currently accepts.  Users can omit it.
    """

    type: Literal["approximate"] = "approximate"
    city: str | None = None
    region: str | None = None
    country: str | None = None
    timezone: str | None = None


class Citations(tools_.BuiltinToolConfig):
    """Citation configuration for web fetch."""

    enabled: bool


# ---------------------------------------------------------------------------
# Tool classes
# ---------------------------------------------------------------------------


class _AnthropicBuiltin(tools_.BuiltinTool):
    """Internal base for Anthropic built-ins.

    Each subclass declares the wire metadata used by ``anthropic/adapter.py``.
    """

    # Anthropic beta feature flag required to enable this tool, if any.
    beta: ClassVar[str | None] = None
    # The ``"type"`` field sent to the API (e.g. ``"web_search_20260209"``).
    # Trailing underscore avoids shadowing the ``type`` builtin.
    type_: ClassVar[str] = ""
    # The ``"name"`` field sent to the API (e.g. ``"web_search"``).
    # Trailing underscore avoids shadowing the ``name`` property on the base.
    name_: ClassVar[str] = ""

    @property
    def name(self) -> str:
        return self.name_


class WebSearch(_AnthropicBuiltin):
    """Web search.

    Domain filters are mutually exclusive — pass only one of
    ``allowed_domains`` / ``blocked_domains``.
    """

    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    user_location: UserLocation | None = None

    beta: ClassVar[str | None] = "code-execution-web-tools-2026-02-09"
    type_: ClassVar[str] = "web_search_20260209"
    name_: ClassVar[str] = "web_search"

    @pydantic.model_validator(mode="after")
    def _check_domains(self) -> WebSearch:
        if self.allowed_domains and self.blocked_domains:
            raise ValueError(
                "anthropic.web_search: pass only one of "
                "`allowed_domains` or `blocked_domains`"
            )
        return self


class WebFetch(_AnthropicBuiltin):
    """Web fetch."""

    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    citations: Citations | None = None
    max_content_tokens: int | None = None

    beta: ClassVar[str | None] = "code-execution-web-tools-2026-02-09"
    type_: ClassVar[str] = "web_fetch_20260209"
    name_: ClassVar[str] = "web_fetch"

    @pydantic.model_validator(mode="after")
    def _check_domains(self) -> WebFetch:
        if self.allowed_domains and self.blocked_domains:
            raise ValueError(
                "anthropic.web_fetch: pass only one of "
                "`allowed_domains` or `blocked_domains`"
            )
        return self


class CodeExecution(_AnthropicBuiltin):
    """Code execution sandbox."""

    beta: ClassVar[str | None] = None
    type_: ClassVar[str] = "code_execution_20260120"
    name_: ClassVar[str] = "code_execution"


class ComputerUse(_AnthropicBuiltin):
    """Computer-use control."""

    display_width_px: int
    display_height_px: int
    display_number: int | None = None
    enable_zoom: bool | None = None

    beta: ClassVar[str | None] = "computer-use-2025-11-24"
    type_: ClassVar[str] = "computer_20251124"
    name_: ClassVar[str] = "computer"


class TextEditor(_AnthropicBuiltin):
    """Text editor."""

    max_characters: int | None = None

    beta: ClassVar[str | None] = None
    type_: ClassVar[str] = "text_editor_20250728"
    name_: ClassVar[str] = "str_replace_based_edit_tool"


class Bash(_AnthropicBuiltin):
    """Bash shell."""

    beta: ClassVar[str | None] = "computer-use-2025-01-24"
    type_: ClassVar[str] = "bash_20250124"
    name_: ClassVar[str] = "bash"


class Memory(_AnthropicBuiltin):
    """Persistent memory tool."""

    beta: ClassVar[str | None] = "context-management-2025-06-27"
    type_: ClassVar[str] = "memory_20250818"
    name_: ClassVar[str] = "memory"


# ---------------------------------------------------------------------------
# Factory functions — convenient call-site syntax
# ---------------------------------------------------------------------------


def web_search(
    *,
    max_uses: int | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    user_location: UserLocation | None = None,
) -> WebSearch:
    return WebSearch(
        max_uses=max_uses,
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
        user_location=user_location,
    )


def web_fetch(
    *,
    max_uses: int | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    citations: Citations | bool | None = None,
    max_content_tokens: int | None = None,
) -> WebFetch:
    cit: Citations | None = (
        Citations(enabled=citations) if isinstance(citations, bool) else citations
    )
    return WebFetch(
        max_uses=max_uses,
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
        citations=cit,
        max_content_tokens=max_content_tokens,
    )


def code_execution() -> CodeExecution:
    return CodeExecution()


def computer_use(
    *,
    display_width_px: int,
    display_height_px: int,
    display_number: int | None = None,
    enable_zoom: bool | None = None,
) -> ComputerUse:
    return ComputerUse(
        display_width_px=display_width_px,
        display_height_px=display_height_px,
        display_number=display_number,
        enable_zoom=enable_zoom,
    )


def text_editor(*, max_characters: int | None = None) -> TextEditor:
    return TextEditor(max_characters=max_characters)


def bash() -> Bash:
    return Bash()


def memory() -> Memory:
    return Memory()


__all__ = [
    "Bash",
    "Citations",
    "CodeExecution",
    "ComputerUse",
    "Memory",
    "TextEditor",
    "UserLocation",
    "WebFetch",
    "WebSearch",
    "bash",
    "code_execution",
    "computer_use",
    "memory",
    "text_editor",
    "web_fetch",
    "web_search",
]
