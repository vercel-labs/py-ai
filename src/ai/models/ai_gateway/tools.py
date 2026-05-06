"""AI Gateway-native provider-executed tools."""

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


class SourcePolicy(pydantic.BaseModel):
    """Source policy for controlling which domains to include/exclude."""

    model_config = _CONFIG_MODEL

    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None
    after_date: str | None = None


class Excerpts(pydantic.BaseModel):
    """Excerpt configuration for controlling result length."""

    model_config = _CONFIG_MODEL

    max_chars_per_result: int | None = None
    max_chars_total: int | None = None


class FetchPolicy(pydantic.BaseModel):
    """Fetch policy for controlling content freshness."""

    model_config = _CONFIG_MODEL

    max_age_seconds: int | None = None


class PerplexitySearchArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    max_results: int | None = None
    max_tokens_per_page: int | None = None
    max_tokens: int | None = None
    country: str | None = None
    search_domain_filter: list[str] | None = None
    search_language_filter: list[str] | None = None
    search_recency_filter: Literal["day", "week", "month", "year"] | None = None


class ParallelSearchArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    mode: Literal["one-shot", "agentic"] | None = None
    max_results: int | None = None
    source_policy: SourcePolicy | None = None
    excerpts: Excerpts | None = None
    fetch_policy: FetchPolicy | None = None


def perplexity_search(args: PerplexitySearchArgs) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="perplexity_search",
        args=args,
    )


def parallel_search(args: ParallelSearchArgs) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="parallel_search",
        args=args,
    )


__all__ = [
    "Excerpts",
    "FetchPolicy",
    "ParallelSearchArgs",
    "PerplexitySearchArgs",
    "SourcePolicy",
    "parallel_search",
    "perplexity_search",
]
