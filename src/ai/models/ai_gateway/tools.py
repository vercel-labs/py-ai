"""Gateway-native provider-executed tools.

These tools are executed server-side by the AI Gateway itself (not by the
underlying language-model provider) and therefore work with **any**
gateway-routed model — Anthropic, OpenAI, Google, etc.

Usage::

    from ai.models import ai_gateway

    tools = [
        ai_gateway.tools.perplexity_search(max_results=5),
        ai_gateway.tools.parallel_search(mode="agentic"),
    ]
    s = ai.stream(model, msgs, tools=tools)

The configuration fields here mirror the JS gateway tool configs and are
sent over the wire as ``args`` of a ``provider``-typed tool block.
"""

from __future__ import annotations

from typing import ClassVar, Literal, TypedDict

import pydantic

from ...types import tools as tools_

_PARAMS_CONFIG = pydantic.ConfigDict(frozen=True, populate_by_name=True)


class _GatewayBuiltin(tools_.BuiltinTool):
    """Internal base for gateway-native built-ins.

    Each subclass declares ``wire_id`` (the ``provider.tool`` id sent on
    the wire, e.g. ``"gateway.perplexity_search"``) and ``wire_name`` (the
    unique name within the model call).
    """

    wire_id: ClassVar[str] = ""
    wire_name: ClassVar[str] = ""

    model_config = _PARAMS_CONFIG


# ---------------------------------------------------------------------------
# perplexity_search
# ---------------------------------------------------------------------------


class PerplexitySearch(_GatewayBuiltin):
    """Search the web using Perplexity's Search API.

    Provides ranked search results with advanced filtering options including
    domain, language, date range, and recency filters.
    """

    max_results: int | None = None
    max_tokens_per_page: int | None = None
    max_tokens: int | None = None
    country: str | None = None
    search_domain_filter: list[str] | None = None
    search_language_filter: list[str] | None = None
    search_recency_filter: Literal["day", "week", "month", "year"] | None = None

    wire_id: ClassVar[str] = "gateway.perplexity_search"
    wire_name: ClassVar[str] = "perplexity_search"


def perplexity_search(
    *,
    max_results: int | None = None,
    max_tokens_per_page: int | None = None,
    max_tokens: int | None = None,
    country: str | None = None,
    search_domain_filter: list[str] | None = None,
    search_language_filter: list[str] | None = None,
    search_recency_filter: Literal["day", "week", "month", "year"] | None = None,
) -> PerplexitySearch:
    return PerplexitySearch(
        max_results=max_results,
        max_tokens_per_page=max_tokens_per_page,
        max_tokens=max_tokens,
        country=country,
        search_domain_filter=search_domain_filter,
        search_language_filter=search_language_filter,
        search_recency_filter=search_recency_filter,
    )


# ---------------------------------------------------------------------------
# parallel_search
# ---------------------------------------------------------------------------


class SourcePolicy(TypedDict, total=False):
    """Source policy for controlling which domains to include/exclude."""

    include_domains: list[str]
    exclude_domains: list[str]
    after_date: str


class Excerpts(TypedDict, total=False):
    """Excerpt configuration for controlling result length."""

    max_chars_per_result: int
    max_chars_total: int


class FetchPolicy(TypedDict, total=False):
    """Fetch policy for controlling content freshness."""

    max_age_seconds: int


class ParallelSearch(_GatewayBuiltin):
    """Search the web using Parallel AI's Search API.

    Takes a natural-language objective and returns relevant excerpts,
    replacing multiple keyword searches with a single call for broad
    or complex queries.
    """

    mode: Literal["one-shot", "agentic"] | None = None
    max_results: int | None = None
    source_policy: SourcePolicy | None = None
    excerpts: Excerpts | None = None
    fetch_policy: FetchPolicy | None = None

    wire_id: ClassVar[str] = "gateway.parallel_search"
    wire_name: ClassVar[str] = "parallel_search"


def parallel_search(
    *,
    mode: Literal["one-shot", "agentic"] | None = None,
    max_results: int | None = None,
    source_policy: SourcePolicy | None = None,
    excerpts: Excerpts | None = None,
    fetch_policy: FetchPolicy | None = None,
) -> ParallelSearch:
    return ParallelSearch(
        mode=mode,
        max_results=max_results,
        source_policy=source_policy,
        excerpts=excerpts,
        fetch_policy=fetch_policy,
    )


__all__ = [
    "Excerpts",
    "FetchPolicy",
    "ParallelSearch",
    "PerplexitySearch",
    "SourcePolicy",
    "parallel_search",
    "perplexity_search",
]
