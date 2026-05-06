"""OpenAI provider-executed tools."""

from __future__ import annotations

from typing import Any, Literal

import pydantic
from pydantic.alias_generators import to_camel

from ... import types

_CONFIG_MODEL = pydantic.ConfigDict(
    frozen=True,
    populate_by_name=True,
    alias_generator=to_camel,
)


class WebSearchUserLocation(pydantic.BaseModel):
    """User-location hint for OpenAI web search."""

    model_config = _CONFIG_MODEL

    type: Literal["approximate"] = "approximate"
    city: str | None = None
    region: str | None = None
    country: str | None = None
    timezone: str | None = None


class WebSearchFilters(pydantic.BaseModel):
    """Filters for OpenAI web search."""

    model_config = _CONFIG_MODEL

    allowed_domains: list[str] | None = None


class FileSearchRanking(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    ranker: str | None = None
    score_threshold: float | None = None


class CodeInterpreterContainer(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    type: Literal["auto"] = "auto"
    file_ids: list[str] | None = None


class WebSearchArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    external_web_access: bool | None = None
    filters: WebSearchFilters | None = None
    search_context_size: Literal["low", "medium", "high"] | None = None
    user_location: WebSearchUserLocation | None = None


class WebSearchPreviewArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    search_context_size: Literal["low", "medium", "high"] | None = None
    user_location: WebSearchUserLocation | None = None


class FileSearchArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    vector_store_ids: list[str]
    max_num_results: int | None = None
    ranking: FileSearchRanking | None = None
    filters: dict[str, Any] | None = None


class CodeInterpreterArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    container: CodeInterpreterContainer | str | None = None


class ImageGenerationArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    background: Literal["transparent", "opaque", "auto"] | None = None
    input_fidelity: Literal["high", "low"] | None = None
    model: str | None = None
    moderation: Literal["auto", "low"] | None = None
    output_compression: int | None = None
    output_format: Literal["png", "webp", "jpeg"] | None = None
    partial_images: int | None = None
    quality: Literal["low", "medium", "high", "auto"] | None = None
    size: str | None = None


class LocalShellArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL


class ShellArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    environment: str | None = None


class ApplyPatchArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL


class McpArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    server_label: str
    server_url: str | None = None
    connector_id: str | None = None
    authorization: str | None = None
    headers: dict[str, str] | None = None
    allowed_tools: list[str] | dict[str, Any] | None = None
    server_description: str | None = None


class ToolSearchArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    description: str | None = None
    parameters: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None


def web_search(args: WebSearchArgs) -> types.tools.Tool:
    return types.tools.Tool(kind="provider", name="web_search", args=args)


def web_search_preview(args: WebSearchPreviewArgs) -> types.tools.Tool:
    return types.tools.Tool(kind="provider", name="web_search_preview", args=args)


def file_search(args: FileSearchArgs) -> types.tools.Tool:
    return types.tools.Tool(kind="provider", name="file_search", args=args)


def code_interpreter(args: CodeInterpreterArgs) -> types.tools.Tool:
    return types.tools.Tool(kind="provider", name="code_interpreter", args=args)


def image_generation(args: ImageGenerationArgs) -> types.tools.Tool:
    return types.tools.Tool(kind="provider", name="image_generation", args=args)


def local_shell(args: LocalShellArgs) -> types.tools.Tool:
    return types.tools.Tool(kind="provider", name="local_shell", args=args)


def shell(args: ShellArgs) -> types.tools.Tool:
    return types.tools.Tool(kind="provider", name="shell", args=args)


def apply_patch(args: ApplyPatchArgs) -> types.tools.Tool:
    return types.tools.Tool(kind="provider", name="apply_patch", args=args)


def mcp(args: McpArgs) -> types.tools.Tool:
    return types.tools.Tool(kind="provider", name="mcp", args=args)


def tool_search(args: ToolSearchArgs) -> types.tools.Tool:
    return types.tools.Tool(kind="provider", name="tool_search", args=args)


__all__ = [
    "ApplyPatchArgs",
    "CodeInterpreterArgs",
    "CodeInterpreterContainer",
    "FileSearchArgs",
    "FileSearchRanking",
    "ImageGenerationArgs",
    "LocalShellArgs",
    "McpArgs",
    "ShellArgs",
    "ToolSearchArgs",
    "WebSearchArgs",
    "WebSearchFilters",
    "WebSearchPreviewArgs",
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
