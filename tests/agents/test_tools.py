"""@tool decorator: schema extraction, execution, ToolCall."""

from __future__ import annotations

from typing import Any

import pydantic
import pytest

import ai

# -- Schema extraction from type hints ------------------------------------


def test_simple_types_produce_correct_schema() -> None:
    @ai.tool
    async def greet(name: str, count: int) -> str:
        """Say hello."""
        return f"Hello {name}" * count

    assert greet.name == "greet"
    assert _function_args(greet).description == "Say hello."
    props = _schema(greet)["properties"]
    assert props["name"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert set(_schema(greet)["required"]) == {"name", "count"}


def test_optional_param_not_required() -> None:
    @ai.tool
    async def search(query: str, limit: int | None = None) -> str:
        """Search."""
        return query

    schema = _schema(search)
    assert "query" in schema.get("required", [])
    assert "limit" not in schema.get("required", [])
    assert "limit" in schema["properties"]


def test_default_value_not_required() -> None:
    @ai.tool
    async def fetch(url: str, timeout: int = 30) -> str:
        """Fetch URL."""
        return url

    assert "url" in _required(fetch)
    assert "timeout" not in _required(fetch)


def test_complex_type_schema() -> None:
    @ai.tool
    async def send(recipients: list[str], urgent: bool = False) -> str:
        """Send message."""
        return "sent"

    props = _schema(send)["properties"]
    assert props["recipients"]["type"] == "array"
    assert props["recipients"]["items"]["type"] == "string"


# -- Execution (ToolCall) --------------------------------------------------


async def test_tool_call_with_json_args() -> None:
    @ai.tool
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    part = ai.messages.ToolCallPart(
        tool_call_id="tc-add",
        tool_name="add",
        tool_args='{"a": 1, "b": 2}',
    )
    result = await ai.ToolCall(part=part, tool=add)()
    assert result.results[0].result == 3


# -- ToolCall binds a ToolCallPart to a Tool and returns tool messages ----


async def test_tool_call_returns_tool_message() -> None:
    @ai.tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    part = ai.messages.ToolCallPart(
        tool_call_id="tc-1",
        tool_name="double",
        tool_args='{"x": 5}',
    )
    tc = ai.ToolCall(part=part, tool=double)
    result = await tc()

    assert tc.fn.__name__ == "double"
    assert tc.kwargs == {"x": 5}
    assert result.message.role == "tool"
    assert len(result.results) == 1
    assert result.results[0].tool_call_id == "tc-1"
    assert result.results[0].tool_name == "double"
    assert result.results[0].result == 10
    assert not result.results[0].is_error


async def test_tool_call_catches_errors() -> None:
    @ai.tool
    async def fail(x: int) -> int:
        """Always fails."""
        raise ValueError("boom")

    part = ai.messages.ToolCallPart(
        tool_call_id="tc-err",
        tool_name="fail",
        tool_args='{"x": 1}',
    )
    tc = ai.ToolCall(part=part, tool=fail)
    result = await tc()

    assert result.results[0].is_error
    assert "boom" in str(result.results[0].result)


async def test_tool_call_allows_kwarg_overrides() -> None:
    @ai.tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    part = ai.messages.ToolCallPart(
        tool_call_id="tc-1",
        tool_name="double",
        tool_args='{"x": 5}',
    )
    tc = ai.ToolCall(part=part, tool=double)

    result = await tc(x=7)

    assert result.results[0].result == 14


async def test_tool_call_override_validation_failure() -> None:
    @ai.tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    part = ai.messages.ToolCallPart(
        tool_call_id="tc-1",
        tool_name="double",
        tool_args='{"x": 5}',
    )
    tc = ai.ToolCall(part=part, tool=double)

    with pytest.raises(pydantic.ValidationError):
        await tc(x="bad")


async def test_tool_call_malformed_args_become_error_message() -> None:
    @ai.tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    part = ai.messages.ToolCallPart(
        tool_call_id="tc-1",
        tool_name="double",
        tool_args='{"x": ',
    )
    tc = ai.ToolCall(part=part, tool=double)

    result = await tc()

    assert result.results[0].is_error


# -- Helpers ---------------------------------------------------------------


def _required(tool: ai.AgentTool) -> list[str]:
    result = _schema(tool).get("required", [])
    assert isinstance(result, list)
    return result


def _schema(tool: ai.AgentTool) -> dict[str, Any]:
    return _function_args(tool).params


def _function_args(tool: ai.AgentTool) -> ai.tools.FunctionToolArgs:
    args = tool.tool.args
    assert isinstance(args, ai.tools.FunctionToolArgs)
    return args
