"""MCP client: tool registration in global registry, end-to-end execution."""

import contextlib
from typing import Any

import mcp.types
import pytest

import vercel_ai_sdk as ai
from vercel_ai_sdk.core.tools import _tool_registry, get_tool
from vercel_ai_sdk.mcp.client import _mcp_tool_to_native

from ..conftest import MockLLM, text_msg, tool_msg


def _fake_mcp_tool(
    name: str = "mcp_echo", description: str = "Echo input"
) -> mcp.types.Tool:
    """Build a minimal mcp.types.Tool for testing."""
    return mcp.types.Tool(
        name=name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )


def _noop_transport_factory() -> contextlib.AbstractAsyncContextManager[Any]:
    """Dummy transport factory — never actually called in these tests."""
    raise NotImplementedError("should not be called")


# -- _mcp_tool_to_native registers in global registry ----------------------


def test_mcp_tool_to_native_registers_in_global_registry() -> None:
    """Converting an MCP tool to native registers it in _tool_registry."""
    mcp_tool = _fake_mcp_tool(name="mcp_reg_test")
    native = _mcp_tool_to_native(mcp_tool, "test:key", _noop_transport_factory, None)

    assert native.name == "mcp_reg_test"
    assert get_tool("mcp_reg_test") is native
    assert _tool_registry["mcp_reg_test"] is native


def test_mcp_tool_to_native_with_prefix() -> None:
    """Tool prefix is prepended to the name and both name forms are correct."""
    mcp_tool = _fake_mcp_tool(name="echo")
    native = _mcp_tool_to_native(mcp_tool, "test:key", _noop_transport_factory, "ctx7")

    assert native.name == "ctx7_echo"
    assert get_tool("ctx7_echo") is native


def test_mcp_tool_to_native_schema_preserved() -> None:
    """The inputSchema from the MCP tool is passed through as param_schema."""
    mcp_tool = _fake_mcp_tool()
    native = _mcp_tool_to_native(mcp_tool, "test:key", _noop_transport_factory, None)

    assert native.param_schema == mcp_tool.inputSchema
    assert native.description == "Echo input"


# -- End-to-end: MCP tool executes through stream_loop --------------------


@pytest.mark.asyncio
async def test_mcp_tool_executes_through_stream_loop() -> None:
    """MCP-style tool via _mcp_tool_to_native can be called by the agent loop."""
    call_log: list[dict[str, str]] = []

    async def fake_fn(**kwargs: str) -> str:
        call_log.append(kwargs)
        return f"echoed: {kwargs.get('text', '')}"

    # Build and register a tool the same way the MCP client does,
    # but with a fake fn so we don't need a real MCP server.
    mcp_tool = _fake_mcp_tool(name="mcp_e2e_echo")
    native = _mcp_tool_to_native(mcp_tool, "test:key", _noop_transport_factory, None)
    # Replace the real fn (which would try to connect) with our fake
    native._fn = fake_fn
    _tool_registry[native.name] = native

    async def graph(llm: ai.LanguageModel) -> ai.StreamResult:
        return await ai.stream_loop(
            llm,
            messages=ai.make_messages(user="echo hello"),
            tools=[native],
        )

    call1 = [tool_msg(tc_id="tc-mcp-1", name="mcp_e2e_echo", args='{"text": "hello"}')]
    call2 = [text_msg("Done.", id="msg-2")]
    llm = MockLLM([call1, call2])

    result = ai.run(graph, llm)
    msgs = [m async for m in result]

    # Tool was called with the right args
    assert len(call_log) == 1
    assert call_log[0] == {"text": "hello"}

    # Tool result is visible in messages
    tool_results = [
        m for m in msgs if m.tool_calls and m.tool_calls[0].status == "result"
    ]
    assert len(tool_results) >= 1
    assert tool_results[0].tool_calls[0].result == "echoed: hello"

    # LLM was called twice (tool call + final text)
    assert llm.call_count == 2
