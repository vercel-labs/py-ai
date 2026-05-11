"""Google provider-executed tools."""

from __future__ import annotations

from typing import Any, ClassVar

import pydantic

from ... import types

_CONFIG_MODEL = pydantic.ConfigDict(
    frozen=True,
    populate_by_name=True,
    extra="forbid",
)


class SearchTypes(pydantic.BaseModel):
    """Google Search result types."""

    model_config = _CONFIG_MODEL

    web_search: dict[str, Any] | None = None
    image_search: dict[str, Any] | None = None


class TimeRangeFilter(pydantic.BaseModel):
    """Timestamp range filter for Google Search."""

    model_config = _CONFIG_MODEL

    start_time: str
    end_time: str


class GoogleProviderArgs(pydantic.BaseModel):
    """Base for Google provider-executed tool args."""

    model_config = _CONFIG_MODEL

    google_tool_name: ClassVar[str]


class GoogleSearchArgs(GoogleProviderArgs):
    google_tool_name: ClassVar[str] = "google_search"

    model_config = _CONFIG_MODEL

    search_types: SearchTypes | dict[str, Any] | None = None
    time_range_filter: TimeRangeFilter | dict[str, Any] | None = None


class UrlContextArgs(GoogleProviderArgs):
    google_tool_name: ClassVar[str] = "url_context"

    model_config = _CONFIG_MODEL


class CodeExecutionArgs(GoogleProviderArgs):
    google_tool_name: ClassVar[str] = "code_execution"

    model_config = _CONFIG_MODEL


class FileSearchArgs(GoogleProviderArgs):
    google_tool_name: ClassVar[str] = "file_search"

    model_config = _CONFIG_MODEL

    file_search_store_names: list[str]
    top_k: int | None = None
    metadata_filter: str | None = None


class GoogleMapsArgs(GoogleProviderArgs):
    google_tool_name: ClassVar[str] = "google_maps"

    model_config = _CONFIG_MODEL


def google_search(
    *,
    search_types: SearchTypes | dict[str, Any] | None = None,
    time_range_filter: TimeRangeFilter | dict[str, Any] | None = None,
) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="google_search",
        args=GoogleSearchArgs(
            search_types=search_types,
            time_range_filter=time_range_filter,
        ),
    )


def url_context() -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="url_context",
        args=UrlContextArgs(),
    )


def code_execution() -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="code_execution",
        args=CodeExecutionArgs(),
    )


def file_search(
    *,
    file_search_store_names: list[str],
    top_k: int | None = None,
    metadata_filter: str | None = None,
) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="file_search",
        args=FileSearchArgs(
            file_search_store_names=file_search_store_names,
            top_k=top_k,
            metadata_filter=metadata_filter,
        ),
    )


def google_maps() -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="google_maps",
        args=GoogleMapsArgs(),
    )


__all__ = [
    "CodeExecutionArgs",
    "FileSearchArgs",
    "GoogleMapsArgs",
    "GoogleProviderArgs",
    "GoogleSearchArgs",
    "SearchTypes",
    "TimeRangeFilter",
    "UrlContextArgs",
    "code_execution",
    "file_search",
    "google_maps",
    "google_search",
    "url_context",
]
