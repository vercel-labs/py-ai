"""@tool decorator: schema extraction, registry, Runtime parameter handling."""

from typing import Optional

import pytest

import vercel_ai_sdk as ai
from vercel_ai_sdk.core.runtime import Runtime
from vercel_ai_sdk.core.tools import _tool_registry, get_tool


# -- Schema extraction from type hints ------------------------------------


def test_simple_types_produce_correct_schema() -> None:
    @ai.tool
    async def greet(name: str, count: int) -> str:
        """Say hello."""
        return f"Hello {name}" * count

    assert greet.name == "greet"
    assert greet.description == "Say hello."
    props = greet.param_schema["properties"]
    assert props["name"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert set(greet.param_schema["required"]) == {"name", "count"}


def test_optional_param_not_required() -> None:
    @ai.tool
    async def search(query: str, limit: Optional[int] = None) -> str:
        """Search."""
        return query

    assert "query" in search.param_schema.get("required", [])
    assert "limit" not in search.param_schema.get("required", [])
    # limit should still appear in properties
    assert "limit" in search.param_schema["properties"]


def test_default_value_not_required() -> None:
    @ai.tool
    async def fetch(url: str, timeout: int = 30) -> str:
        """Fetch URL."""
        return url

    assert "url" in search_required(fetch)
    assert "timeout" not in search_required(fetch)


def test_complex_type_schema() -> None:
    @ai.tool
    async def send(recipients: list[str], urgent: bool = False) -> str:
        """Send message."""
        return "sent"

    props = send.param_schema["properties"]
    assert props["recipients"]["type"] == "array"
    assert props["recipients"]["items"]["type"] == "string"


# -- Runtime parameter skipping -------------------------------------------


def test_runtime_param_excluded_from_schema() -> None:
    @ai.tool
    async def needs_runtime(query: str, rt: Runtime) -> str:
        """Tool that needs runtime."""
        return query

    props = needs_runtime.param_schema["properties"]
    assert "rt" not in props
    assert "query" in props
    assert set(needs_runtime.param_schema.get("required", [])) == {"query"}


# -- Registry -------------------------------------------------------------


def test_tool_registered_on_decoration() -> None:
    @ai.tool
    async def unique_tool_abc() -> str:
        """Unique."""
        return "ok"

    assert get_tool("unique_tool_abc") is unique_tool_abc


def test_get_tool_returns_none_for_missing() -> None:
    assert get_tool("nonexistent_tool_xyz") is None


# -- Execution ------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_fn_is_callable() -> None:
    @ai.tool
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    result = await add(a=1, b=2)
    assert result == 3


# -- Helpers ---------------------------------------------------------------


def search_required(tool: ai.Tool) -> list[str]:
    return tool.param_schema.get("required", [])
