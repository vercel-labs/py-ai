"""OpenAI provider-executed (built-in) tools.

These types describe OpenAI's built-in tool surface (web search, code
interpreter, file search, image generation, MCP, ...). They are valid
tools to pass through ``ai.stream(model, msgs, tools=[...])`` when the
model is reached via the AI Gateway provider, which forwards
provider-executed tools transparently.

The native OpenAI chat-completions adapter does **not** support
provider-executed tools (those require the Responses API). Passing one
to a model returned by ``openai("...")`` raises ``NotImplementedError``;
use ``ai_gateway("openai/...")`` until a native Responses adapter ships.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal, TypedDict

import pydantic

from ...types import tools as tools_

# ---------------------------------------------------------------------------
# Shared sub-types
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


# ---------------------------------------------------------------------------
# Tool classes
# ---------------------------------------------------------------------------


_PARAMS_CONFIG = pydantic.ConfigDict(frozen=True, populate_by_name=True)


class _OpenAIBuiltin(tools_.BuiltinTool):
    """Internal base for OpenAI built-ins.

    Each subclass declares the wire metadata used by the Responses-API
    adapter (and by the AI-Gateway adapter for routing).
    """

    # The wire-level ``type`` field passed to the Responses API
    # (e.g. ``"web_search"``, ``"code_interpreter"``).
    wire_type: ClassVar[str] = ""

    model_config = _PARAMS_CONFIG


class WebSearch(_OpenAIBuiltin):
    """Web search (Responses API)."""

    external_web_access: bool | None = None
    filters: WebSearchFilters | None = None
    search_context_size: Literal["low", "medium", "high"] | None = None
    user_location: WebSearchUserLocation | None = None

    wire_type: ClassVar[str] = "web_search"


class WebSearchPreview(_OpenAIBuiltin):
    """Web search preview (Responses API)."""

    search_context_size: Literal["low", "medium", "high"] | None = None
    user_location: WebSearchUserLocation | None = None

    wire_type: ClassVar[str] = "web_search_preview"


class FileSearch(_OpenAIBuiltin):
    """File search (vector store retrieval)."""

    vector_store_ids: Sequence[str]
    max_num_results: int | None = None
    ranking: FileSearchRanking | None = None
    filters: dict[str, Any] | None = None

    wire_type: ClassVar[str] = "file_search"


class CodeInterpreter(_OpenAIBuiltin):
    """Python code interpreter sandbox."""

    container: CodeInterpreterContainer | str | None = None

    wire_type: ClassVar[str] = "code_interpreter"


class ImageGeneration(_OpenAIBuiltin):
    """Image generation tool."""

    background: Literal["transparent", "opaque", "auto"] | None = None
    input_fidelity: Literal["high", "low"] | None = None
    model: str | None = None
    moderation: Literal["auto", "low"] | None = None
    output_compression: int | None = None
    output_format: Literal["png", "webp", "jpeg"] | None = None
    partial_images: int | None = None
    quality: Literal["low", "medium", "high", "auto"] | None = None
    size: str | None = None

    wire_type: ClassVar[str] = "image_generation"


class LocalShell(_OpenAIBuiltin):
    wire_type: ClassVar[str] = "local_shell"


class Shell(_OpenAIBuiltin):
    environment: str | None = None
    wire_type: ClassVar[str] = "shell"


class ApplyPatch(_OpenAIBuiltin):
    wire_type: ClassVar[str] = "apply_patch"


class MCP(_OpenAIBuiltin):
    """MCP (Model Context Protocol) server."""

    server_label: str
    server_url: str | None = None
    connector_id: str | None = None
    authorization: str | None = None
    headers: dict[str, str] | None = None
    allowed_tools: list[str] | dict[str, Any] | None = None
    server_description: str | None = None

    wire_type: ClassVar[str] = "mcp"


class ToolSearch(_OpenAIBuiltin):
    """Dynamic tool search (defer-load gated)."""

    description: str | None = None
    parameters: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None

    wire_type: ClassVar[str] = "tool_search"


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def web_search(
    *,
    external_web_access: bool | None = None,
    filters: WebSearchFilters | None = None,
    search_context_size: Literal["low", "medium", "high"] | None = None,
    user_location: WebSearchUserLocation | None = None,
) -> WebSearch:
    return WebSearch(
        external_web_access=external_web_access,
        filters=filters,
        search_context_size=search_context_size,
        user_location=user_location,
    )


def web_search_preview(
    *,
    search_context_size: Literal["low", "medium", "high"] | None = None,
    user_location: WebSearchUserLocation | None = None,
) -> WebSearchPreview:
    return WebSearchPreview(
        search_context_size=search_context_size,
        user_location=user_location,
    )


def file_search(
    *,
    vector_store_ids: Sequence[str],
    max_num_results: int | None = None,
    ranking: FileSearchRanking | None = None,
    filters: dict[str, Any] | None = None,
) -> FileSearch:
    return FileSearch(
        vector_store_ids=vector_store_ids,
        max_num_results=max_num_results,
        ranking=ranking,
        filters=filters,
    )


def code_interpreter(
    *,
    container: CodeInterpreterContainer | str | None = None,
) -> CodeInterpreter:
    return CodeInterpreter(container=container)


def image_generation(
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
) -> ImageGeneration:
    return ImageGeneration(
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


def local_shell() -> LocalShell:
    return LocalShell()


def shell(*, environment: str | None = None) -> Shell:
    return Shell(environment=environment)


def apply_patch() -> ApplyPatch:
    return ApplyPatch()


def mcp(
    *,
    server_label: str,
    server_url: str | None = None,
    connector_id: str | None = None,
    authorization: str | None = None,
    headers: dict[str, str] | None = None,
    allowed_tools: list[str] | dict[str, Any] | None = None,
    server_description: str | None = None,
) -> MCP:
    return MCP(
        server_label=server_label,
        server_url=server_url,
        connector_id=connector_id,
        authorization=authorization,
        headers=headers,
        allowed_tools=allowed_tools,
        server_description=server_description,
    )


def tool_search(
    *,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
) -> ToolSearch:
    return ToolSearch(
        description=description,
        parameters=parameters,
        execution=execution,
    )


__all__ = [
    "MCP",
    "ApplyPatch",
    "CodeInterpreter",
    "CodeInterpreterContainer",
    "FileSearch",
    "FileSearchRanking",
    "ImageGeneration",
    "LocalShell",
    "Shell",
    "ToolSearch",
    "WebSearch",
    "WebSearchFilters",
    "WebSearchPreview",
    "WebSearchUserLocation",
    "apply_patch",
    "code_interpreter",
    "file_search",
    "image_generation",
    "local_shell",
    "mcp",
    "shell",
    "tool_search",
    "web_search",
    "web_search_preview",
]
