from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import json
from typing import Any, Callable

import mcp.client.session
import mcp.client.stdio
import mcp.client.streamable_http
import mcp.types

from .. import core

__all__ = [
    "get_stdio_tools",
    "get_http_tools",
    "close_connections",
]


@dataclasses.dataclass
class _Connection:
    """Internal connection state - never exposed to users."""

    client: mcp.client.session.ClientSession
    exit_stack: contextlib.AsyncExitStack


# Connection pool stored in contextvar, scoped to execute()
# The pool is set by execute() and cleaned up when execute() finishes
_pool: contextvars.ContextVar[dict[str, _Connection] | None] = contextvars.ContextVar(
    "mcp_connections", default=None
)

_pool_lock = asyncio.Lock()


async def _get_or_create_connection(
    key: str,
    transport_factory: Callable[[], contextlib.AbstractAsyncContextManager[Any]],
) -> mcp.client.session.ClientSession:
    """Get an existing connection or create a new one."""
    pool = _pool.get()

    if pool is None:
        raise RuntimeError(
            "MCP tools must be used inside ai.execute(). "
            "The connection pool is not initialized."
        )

    async with _pool_lock:
        if key in pool:
            return pool[key].client

        # Use AsyncExitStack for clean resource management
        exit_stack = contextlib.AsyncExitStack()

        try:
            # Enter the transport context
            streams = await exit_stack.enter_async_context(transport_factory())

            # Handle both (read, write) and (read, write, callback) returns
            read_stream, write_stream = streams[0], streams[1]

            # Create and initialize the client session
            client = mcp.client.session.ClientSession(
                read_stream=read_stream,
                write_stream=write_stream,
            )
            await exit_stack.enter_async_context(client)
            await client.initialize()

            pool[key] = _Connection(client=client, exit_stack=exit_stack)
            return client

        except BaseException:
            # Clean up on any error during setup
            await exit_stack.aclose()
            raise


def _make_tool_fn(
    connection_key: str,
    tool_name: str,
    transport_factory: Callable[[], contextlib.AbstractAsyncContextManager[Any]],
) -> Callable[..., Any]:
    """Create a tool function that manages its own connection."""

    async def call_tool(**kwargs: Any) -> Any:
        client = await _get_or_create_connection(connection_key, transport_factory)
        try:
            result = await asyncio.wait_for(
                client.call_tool(tool_name, kwargs),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"MCP tool call timed out after 30 seconds: {tool_name}")

        # Handle error responses
        if result.isError:
            error_text = " ".join(
                part.text
                for part in result.content
                if isinstance(part, mcp.types.TextContent)
            )
            raise RuntimeError(f"MCP tool error: {error_text or 'Unknown error'}")

        # Prefer structured content if available
        if result.structuredContent is not None:
            return result.structuredContent

        # Fall back to parsing content
        for part in result.content:
            if isinstance(part, mcp.types.TextContent):
                text = part.text
                # Try to parse JSON, otherwise return raw text
                if text.startswith(("{", "[")):
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        pass
                return text

        return result.content

    return call_tool


async def get_stdio_tools(
    command: str,
    *args: str,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    tool_prefix: str | None = None,
) -> list[core.tools.Tool]:
    """
    Get tools from an MCP server running as a subprocess.

    Connection is managed automatically - created on first use, cleaned up
    when execute() finishes.

    Args:
        command: The command to run (e.g., "npx", "python").
        *args: Arguments to pass to the command.
        env: Environment variables for the subprocess.
        cwd: Working directory for the subprocess.
        tool_prefix: Optional prefix to add to all tool names.

    Returns:
        List of Tool objects that can be passed to stream_loop.

    Example:
        tools = await ai.mcp.get_stdio_tools(
            "npx", "-y", "@anthropic/mcp-server-filesystem", "/tmp"
        )
    """
    connection_key = f"stdio:{command}:{':'.join(args)}"

    def transport_factory():
        return mcp.client.stdio.stdio_client(
            mcp.client.stdio.StdioServerParameters(
                command=command,
                args=list(args),
                env=env,
                cwd=cwd,
            )
        )

    client = await _get_or_create_connection(connection_key, transport_factory)
    result = await client.list_tools()

    return [
        _mcp_tool_to_native(mcp_tool, connection_key, transport_factory, tool_prefix)
        for mcp_tool in result.tools
    ]


async def get_http_tools(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    tool_prefix: str | None = None,
) -> list[core.tools.Tool]:
    """
    Get tools from an MCP server over HTTP (Streamable HTTP transport).

    Connection is managed automatically - created on first use, cleaned up
    when execute() finishes.

    Args:
        url: The URL of the MCP server endpoint.
        headers: Optional HTTP headers (e.g., for authentication).
        tool_prefix: Optional prefix to add to all tool names.

    Returns:
        List of Tool objects that can be passed to stream_loop.

    Example:
        tools = await ai.mcp.get_http_tools(
            "http://localhost:3000/mcp",
            headers={"Authorization": "Bearer xxx"}
        )
    """
    connection_key = f"http:{url}"

    def transport_factory():
        return mcp.client.streamable_http.streamablehttp_client(
            url=url, headers=headers
        )

    client = await _get_or_create_connection(connection_key, transport_factory)
    result = await client.list_tools()

    return [
        _mcp_tool_to_native(mcp_tool, connection_key, transport_factory, tool_prefix)
        for mcp_tool in result.tools
    ]


def _mcp_tool_to_native(
    mcp_tool: mcp.types.Tool,
    connection_key: str,
    transport_factory: Callable[[], contextlib.AbstractAsyncContextManager[Any]],
    tool_prefix: str | None,
) -> core.tools.Tool:
    """Convert an MCP tool to a native Tool."""
    name = mcp_tool.name
    if tool_prefix:
        name = f"{tool_prefix}_{name}"

    return core.tools.Tool(
        name=name,
        description=mcp_tool.description or "",
        parameters=mcp_tool.inputSchema,
        fn=_make_tool_fn(connection_key, mcp_tool.name, transport_factory),
    )


async def close_connections() -> None:
    """
    Close all MCP connections in the current context.

    This is called automatically by execute(), but can be called
    manually for explicit cleanup.
    """
    pool = _pool.get()
    if pool is None:
        return

    async with _pool_lock:
        if not pool:
            return

        # Use TaskGroup for concurrent cleanup
        async with asyncio.TaskGroup() as tg:
            for conn in pool.values():
                tg.create_task(_close_connection_safely(conn))

        pool.clear()


async def _close_connection_safely(conn: _Connection) -> None:
    """Close a connection, suppressing any errors."""
    with contextlib.suppress(Exception):
        await conn.exit_stack.aclose()
