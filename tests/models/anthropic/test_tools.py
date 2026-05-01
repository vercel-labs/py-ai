"""Tests for the Anthropic built-in tool surface.

The adapter must:

1. Translate each :class:`_AnthropicBuiltin` subclass to its registered
   wire ``type`` / ``name`` and field set.
2. Merge the ``beta`` ClassVar from each tool into the
   ``anthropic-beta`` request header alongside user-supplied betas,
   deduplicated.
3. Reject mutually-exclusive domain filters at construction time
   (``allowed_domains`` + ``blocked_domains``).
"""

from __future__ import annotations

from typing import Any

import pytest

import ai
from ai import models
from ai.models.anthropic import adapter, anthropic
from ai.models.anthropic import params as anthropic_params
from ai.models.anthropic import tools as anthropic_tools

from .conftest import FakeAnthropicClient

_CLIENT = models.Client(base_url="https://anthropic.test", api_key="sk-test")
_MODEL = anthropic("claude-sonnet-4-6")


async def _capture_tools(
    monkeypatch: pytest.MonkeyPatch,
    tools: list[Any],
    *,
    params: anthropic_params.AnthropicParams | None = None,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        adapter, "_make_client", lambda client: FakeAnthropicClient(captured)
    )
    stream = adapter.stream(
        _CLIENT, _MODEL, [ai.user_message("Hi")], tools=tools, params=params
    )
    async for _ in stream:
        pass
    return captured


def _beta_header(captured: dict[str, Any]) -> str | None:
    return (captured.get("extra_headers") or {}).get("anthropic-beta")


async def test_web_search_full_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = await _capture_tools(
        monkeypatch,
        [
            anthropic_tools.web_search(
                max_uses=3,
                allowed_domains=["example.com"],
                user_location={"city": "SF", "country": "US"},
            )
        ],
    )

    assert captured["tools"] == [
        {
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": 3,
            "allowed_domains": ["example.com"],
            "user_location": {
                "type": "approximate",
                "city": "SF",
                "country": "US",
            },
        }
    ]
    assert _beta_header(captured) == "code-execution-web-tools-2026-02-09"


async def test_web_fetch_citations_bool_coerced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = await _capture_tools(
        monkeypatch, [anthropic_tools.web_fetch(citations=True)]
    )

    assert captured["tools"] == [
        {
            "type": "web_fetch_20260209",
            "name": "web_fetch",
            "citations": {"enabled": True},
        }
    ]


async def test_computer_use_required_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = await _capture_tools(
        monkeypatch,
        [
            anthropic_tools.computer_use(
                display_width_px=1024,
                display_height_px=768,
            )
        ],
    )

    assert captured["tools"] == [
        {
            "type": "computer_20251124",
            "name": "computer",
            "display_width_px": 1024,
            "display_height_px": 768,
        }
    ]
    assert _beta_header(captured) == "computer-use-2025-11-24"


async def test_text_editor_wire_name_differs_from_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``TextEditor`` ships under the ``str_replace_based_edit_tool`` wire name."""
    captured = await _capture_tools(monkeypatch, [anthropic_tools.text_editor()])

    assert captured["tools"] == [
        {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}
    ]


async def test_bare_tools_emit_only_type_and_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = await _capture_tools(
        monkeypatch,
        [
            anthropic_tools.code_execution(),
            anthropic_tools.bash(),
            anthropic_tools.memory(),
        ],
    )

    assert captured["tools"] == [
        {"type": "code_execution_20260120", "name": "code_execution"},
        {"type": "bash_20250124", "name": "bash"},
        {"type": "memory_20250818", "name": "memory"},
    ]


async def test_betas_merge_dedup_with_user_betas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User betas + tool betas merge into a single comma-joined header."""
    captured = await _capture_tools(
        monkeypatch,
        [anthropic_tools.web_search(), anthropic_tools.web_fetch()],
        params=anthropic_params.AnthropicParams(betas=["custom-beta-2026-01-01"]),
    )

    header = _beta_header(captured)
    assert header is not None
    parts = header.split(",")
    # User betas come first, tool betas next; both shared tool betas dedup.
    assert parts == [
        "custom-beta-2026-01-01",
        "code-execution-web-tools-2026-02-09",
    ]


def test_web_search_rejects_mutually_exclusive_domains() -> None:
    with pytest.raises(ValueError, match="only one of"):
        anthropic_tools.web_search(
            allowed_domains=["a.example"],
            blocked_domains=["b.example"],
        )


def test_web_fetch_rejects_mutually_exclusive_domains() -> None:
    with pytest.raises(ValueError, match="only one of"):
        anthropic_tools.web_fetch(
            allowed_domains=["a.example"],
            blocked_domains=["b.example"],
        )
