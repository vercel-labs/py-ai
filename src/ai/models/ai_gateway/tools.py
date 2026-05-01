"""AI Gateway provider-executed (built-in) tools.

These types describe the built-in tool surface routable through the AI
Gateway. Each class carries a ``gateway_id`` ClassVar of the form
``<provider>.<wire_type>`` (the AI-Gateway/Vercel convention), which the
adapter forwards verbatim in the v3 wire ``provider`` tool block.

The types here are **independent** of the native provider packages
(``ai.models.anthropic`` and ``ai.models.openai``); their fields are
duplicated on purpose so the gateway package stays self-contained.

Usage::

    from ai.models import ai_gateway

    model = ai_gateway("anthropic/claude-sonnet-4")
    s = ai.stream(
        model,
        msgs,
        tools=[
            ai_gateway.tools.anthropic_web_search(max_uses=5),
            ai_gateway.tools.openai_code_interpreter(),
        ],
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal, TypedDict

import pydantic

from ...types import tools as tools_

# ---------------------------------------------------------------------------
# Internal base
# ---------------------------------------------------------------------------


_PARAMS_CONFIG = pydantic.ConfigDict(frozen=True, populate_by_name=True)


class _GatewayBuiltin(tools_.BuiltinTool):
    """Internal base for built-ins routable through the AI Gateway.

    Each subclass declares the ``gateway_id`` used by ``ai_gateway/adapter.py``
    when emitting v3 ``provider`` tool blocks.
    """

    # Wire id of the form ``<provider>.<wire_type>`` (Vercel/AI-Gateway
    # convention), e.g. ``"anthropic.web_search_20260209"``.
    gateway_id: ClassVar[str] = ""

    model_config = _PARAMS_CONFIG


# ---------------------------------------------------------------------------
# Anthropic-flavored built-ins
# ---------------------------------------------------------------------------


class UserLocation(TypedDict, total=False):
    """Approximate user location for geographically relevant search results."""

    city: str
    region: str
    country: str
    timezone: str


class Citations(pydantic.BaseModel):
    """Citation configuration for web fetch."""

    enabled: bool

    model_config = pydantic.ConfigDict(frozen=True)


class AnthropicWebSearch(_GatewayBuiltin):
    """Anthropic web search.

    Domain filters are mutually exclusive — pass only one of
    ``allowed_domains`` / ``blocked_domains``.
    """

    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    user_location: UserLocation | None = None

    gateway_id: ClassVar[str] = "anthropic.web_search_20260209"

    @pydantic.model_validator(mode="after")
    def _check_domains(self) -> AnthropicWebSearch:
        if self.allowed_domains and self.blocked_domains:
            raise ValueError(
                "ai_gateway.tools.anthropic_web_search: pass only one of "
                "`allowed_domains` or `blocked_domains`"
            )
        return self


class AnthropicWebFetch(_GatewayBuiltin):
    """Anthropic web fetch."""

    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    citations: Citations | None = None
    max_content_tokens: int | None = None

    gateway_id: ClassVar[str] = "anthropic.web_fetch_20260209"

    @pydantic.model_validator(mode="after")
    def _check_domains(self) -> AnthropicWebFetch:
        if self.allowed_domains and self.blocked_domains:
            raise ValueError(
                "ai_gateway.tools.anthropic_web_fetch: pass only one of "
                "`allowed_domains` or `blocked_domains`"
            )
        return self


class AnthropicCodeExecution(_GatewayBuiltin):
    """Anthropic code execution sandbox."""

    gateway_id: ClassVar[str] = "anthropic.code_execution_20260120"


class AnthropicComputerUse(_GatewayBuiltin):
    """Anthropic computer-use control."""

    display_width_px: int
    display_height_px: int
    display_number: int | None = None
    enable_zoom: bool | None = None

    gateway_id: ClassVar[str] = "anthropic.computer_20251124"


class AnthropicTextEditor(_GatewayBuiltin):
    """Anthropic text editor."""

    max_characters: int | None = None

    gateway_id: ClassVar[str] = "anthropic.text_editor_20250728"


class AnthropicBash(_GatewayBuiltin):
    """Anthropic bash shell."""

    gateway_id: ClassVar[str] = "anthropic.bash_20250124"


class AnthropicMemory(_GatewayBuiltin):
    """Anthropic persistent memory tool."""

    gateway_id: ClassVar[str] = "anthropic.memory_20250818"


# ---------------------------------------------------------------------------
# OpenAI-flavored built-ins
# ---------------------------------------------------------------------------


class WebSearchUserLocation(TypedDict, total=False):
    type: Literal["approximate"]
    city: str
    region: str
    country: str
    timezone: str


class WebSearchFilters(TypedDict, total=False):
    allowed_domains: list[str]


class FileSearchRanking(pydantic.BaseModel):
    ranker: str | None = None
    score_threshold: float | None = None

    model_config = pydantic.ConfigDict(frozen=True)


class CodeInterpreterContainer(pydantic.BaseModel):
    type: Literal["auto"] = "auto"
    file_ids: list[str] | None = None

    model_config = pydantic.ConfigDict(frozen=True)


class OpenAIWebSearch(_GatewayBuiltin):
    """OpenAI web search (Responses API)."""

    external_web_access: bool | None = None
    filters: WebSearchFilters | None = None
    search_context_size: Literal["low", "medium", "high"] | None = None
    user_location: WebSearchUserLocation | None = None

    gateway_id: ClassVar[str] = "openai.web_search"


class OpenAIWebSearchPreview(_GatewayBuiltin):
    """OpenAI web search preview (Responses API)."""

    search_context_size: Literal["low", "medium", "high"] | None = None
    user_location: WebSearchUserLocation | None = None

    gateway_id: ClassVar[str] = "openai.web_search_preview"


class OpenAIFileSearch(_GatewayBuiltin):
    """OpenAI file search (vector store retrieval)."""

    vector_store_ids: Sequence[str]
    max_num_results: int | None = None
    ranking: FileSearchRanking | None = None
    filters: dict[str, Any] | None = None

    gateway_id: ClassVar[str] = "openai.file_search"


class OpenAICodeInterpreter(_GatewayBuiltin):
    """OpenAI Python code interpreter sandbox."""

    container: CodeInterpreterContainer | str | None = None

    gateway_id: ClassVar[str] = "openai.code_interpreter"


class OpenAIImageGeneration(_GatewayBuiltin):
    """OpenAI image generation tool."""

    background: Literal["transparent", "opaque", "auto"] | None = None
    input_fidelity: Literal["high", "low"] | None = None
    model: str | None = None
    moderation: Literal["auto", "low"] | None = None
    output_compression: int | None = None
    output_format: Literal["png", "webp", "jpeg"] | None = None
    partial_images: int | None = None
    quality: Literal["low", "medium", "high", "auto"] | None = None
    size: str | None = None

    gateway_id: ClassVar[str] = "openai.image_generation"


class OpenAILocalShell(_GatewayBuiltin):
    gateway_id: ClassVar[str] = "openai.local_shell"


class OpenAIShell(_GatewayBuiltin):
    environment: str | None = None
    gateway_id: ClassVar[str] = "openai.shell"


class OpenAIApplyPatch(_GatewayBuiltin):
    gateway_id: ClassVar[str] = "openai.apply_patch"


class OpenAIMCP(_GatewayBuiltin):
    """OpenAI MCP (Model Context Protocol) server."""

    server_label: str
    server_url: str | None = None
    connector_id: str | None = None
    authorization: str | None = None
    headers: dict[str, str] | None = None
    allowed_tools: list[str] | dict[str, Any] | None = None
    server_description: str | None = None

    gateway_id: ClassVar[str] = "openai.mcp"


class OpenAIToolSearch(_GatewayBuiltin):
    """OpenAI dynamic tool search (defer-load gated)."""

    description: str | None = None
    parameters: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None

    gateway_id: ClassVar[str] = "openai.tool_search"


# ---------------------------------------------------------------------------
# Factory functions — Anthropic
# ---------------------------------------------------------------------------


def anthropic_web_search(
    *,
    max_uses: int | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    user_location: UserLocation | None = None,
) -> AnthropicWebSearch:
    return AnthropicWebSearch(
        max_uses=max_uses,
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
        user_location=user_location,
    )


def anthropic_web_fetch(
    *,
    max_uses: int | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    citations: Citations | bool | None = None,
    max_content_tokens: int | None = None,
) -> AnthropicWebFetch:
    cit: Citations | None = (
        Citations(enabled=citations) if isinstance(citations, bool) else citations
    )
    return AnthropicWebFetch(
        max_uses=max_uses,
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
        citations=cit,
        max_content_tokens=max_content_tokens,
    )


def anthropic_code_execution() -> AnthropicCodeExecution:
    return AnthropicCodeExecution()


def anthropic_computer_use(
    *,
    display_width_px: int,
    display_height_px: int,
    display_number: int | None = None,
    enable_zoom: bool | None = None,
) -> AnthropicComputerUse:
    return AnthropicComputerUse(
        display_width_px=display_width_px,
        display_height_px=display_height_px,
        display_number=display_number,
        enable_zoom=enable_zoom,
    )


def anthropic_text_editor(*, max_characters: int | None = None) -> AnthropicTextEditor:
    return AnthropicTextEditor(max_characters=max_characters)


def anthropic_bash() -> AnthropicBash:
    return AnthropicBash()


def anthropic_memory() -> AnthropicMemory:
    return AnthropicMemory()


# ---------------------------------------------------------------------------
# Factory functions — OpenAI
# ---------------------------------------------------------------------------


def openai_web_search(
    *,
    external_web_access: bool | None = None,
    filters: WebSearchFilters | None = None,
    search_context_size: Literal["low", "medium", "high"] | None = None,
    user_location: WebSearchUserLocation | None = None,
) -> OpenAIWebSearch:
    return OpenAIWebSearch(
        external_web_access=external_web_access,
        filters=filters,
        search_context_size=search_context_size,
        user_location=user_location,
    )


def openai_web_search_preview(
    *,
    search_context_size: Literal["low", "medium", "high"] | None = None,
    user_location: WebSearchUserLocation | None = None,
) -> OpenAIWebSearchPreview:
    return OpenAIWebSearchPreview(
        search_context_size=search_context_size,
        user_location=user_location,
    )


def openai_file_search(
    *,
    vector_store_ids: Sequence[str],
    max_num_results: int | None = None,
    ranking: FileSearchRanking | None = None,
    filters: dict[str, Any] | None = None,
) -> OpenAIFileSearch:
    return OpenAIFileSearch(
        vector_store_ids=vector_store_ids,
        max_num_results=max_num_results,
        ranking=ranking,
        filters=filters,
    )


def openai_code_interpreter(
    *,
    container: CodeInterpreterContainer | str | None = None,
) -> OpenAICodeInterpreter:
    return OpenAICodeInterpreter(container=container)


def openai_image_generation(
    *,
    background: Literal["transparent", "opaque", "auto"] | None = None,
    input_fidelity: Literal["high", "low"] | None = None,
    model: str | None = None,
    moderation: Literal["auto", "low"] | None = None,
    output_compression: int | None = None,
    output_format: Literal["png", "webp", "jpeg"] | None = None,
    partial_images: int | None = None,
    quality: Literal["low", "medium", "high", "auto"] | None = None,
    size: str | None = None,
) -> OpenAIImageGeneration:
    return OpenAIImageGeneration(
        background=background,
        input_fidelity=input_fidelity,
        model=model,
        moderation=moderation,
        output_compression=output_compression,
        output_format=output_format,
        partial_images=partial_images,
        quality=quality,
        size=size,
    )


def openai_local_shell() -> OpenAILocalShell:
    return OpenAILocalShell()


def openai_shell(*, environment: str | None = None) -> OpenAIShell:
    return OpenAIShell(environment=environment)


def openai_apply_patch() -> OpenAIApplyPatch:
    return OpenAIApplyPatch()


def openai_mcp(
    *,
    server_label: str,
    server_url: str | None = None,
    connector_id: str | None = None,
    authorization: str | None = None,
    headers: dict[str, str] | None = None,
    allowed_tools: list[str] | dict[str, Any] | None = None,
    server_description: str | None = None,
) -> OpenAIMCP:
    return OpenAIMCP(
        server_label=server_label,
        server_url=server_url,
        connector_id=connector_id,
        authorization=authorization,
        headers=headers,
        allowed_tools=allowed_tools,
        server_description=server_description,
    )


def openai_tool_search(
    *,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
) -> OpenAIToolSearch:
    return OpenAIToolSearch(
        description=description,
        parameters=parameters,
        execution=execution,
    )


__all__ = [
    "AnthropicBash",
    "AnthropicCodeExecution",
    "AnthropicComputerUse",
    "AnthropicMemory",
    "AnthropicTextEditor",
    "AnthropicWebFetch",
    "AnthropicWebSearch",
    "Citations",
    "CodeInterpreterContainer",
    "FileSearchRanking",
    "OpenAIApplyPatch",
    "OpenAICodeInterpreter",
    "OpenAIFileSearch",
    "OpenAIImageGeneration",
    "OpenAILocalShell",
    "OpenAIMCP",
    "OpenAIShell",
    "OpenAIToolSearch",
    "OpenAIWebSearch",
    "OpenAIWebSearchPreview",
    "UserLocation",
    "WebSearchFilters",
    "WebSearchUserLocation",
    "anthropic_bash",
    "anthropic_code_execution",
    "anthropic_computer_use",
    "anthropic_memory",
    "anthropic_text_editor",
    "anthropic_web_fetch",
    "anthropic_web_search",
    "openai_apply_patch",
    "openai_code_interpreter",
    "openai_file_search",
    "openai_image_generation",
    "openai_local_shell",
    "openai_mcp",
    "openai_shell",
    "openai_tool_search",
    "openai_web_search",
    "openai_web_search_preview",
]
