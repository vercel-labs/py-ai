from __future__ import annotations

import contextvars

import vercel_ai_sdk as ai
from . import proto

_filesystem: contextvars.ContextVar[proto.Filesystem] = contextvars.ContextVar(
    "agent_filesystem"
)


def _fs() -> proto.Filesystem:
    fs = _filesystem.get(None)
    if fs is None:
        raise RuntimeError("No filesystem bound — tools must run inside Agent.run()")
    return fs


@ai.tool
async def read(path: str, offset: int | None = None, limit: int | None = None) -> str:
    """Read a file and return its contents with line numbers. For large files, use offset/limit to paginate."""
    return await _fs().read(path, offset=offset, limit=limit)


@ai.tool
async def write(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories automatically. Overwrites existing files."""
    await _fs().write(path, content)
    return f"Wrote {len(content)} bytes to {path}"


@ai.tool
async def edit(path: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing an exact string match. Fails if old_string is not found or appears multiple times."""
    await _fs().edit(path, old_string, new_string)
    return f"Edited {path}"


@ai.tool
async def ls(
    path: str = ".",
    depth: int | None = None,
    pattern: str | None = None,
    include_hidden: bool = False,
) -> str:
    """List directory contents recursively. Control depth to balance detail vs overview."""
    return await _fs().ls(
        path, depth=depth, pattern=pattern, include_hidden=include_hidden
    )


@ai.tool
async def glob(pattern: str, path: str | None = None) -> str:
    """Find files matching a glob pattern (e.g. '**/*.py', 'src/**/*.ts')."""
    matches = await _fs().glob(pattern, path=path)
    if not matches:
        return "(no matches)"
    return "\n".join(matches)


@ai.tool
async def grep(
    pattern: str,
    path: str | None = None,
    include: str | None = None,
    context_lines: int | None = None,
    max_count: int | None = None,
    case_sensitive: bool = True,
) -> str:
    """Search file contents using regex (ripgrep syntax). Use include to filter by file pattern (e.g. '*.py')."""
    return await _fs().grep(
        pattern,
        path=path,
        include=include,
        context_lines=context_lines,
        max_count=max_count,
        case_sensitive=case_sensitive,
    )


@ai.tool
async def bash(command: str, timeout: int | None = None) -> str:
    """Execute a bash command in the workspace. Use timeout (seconds) to limit long-running commands."""
    return await _fs().bash(command, timeout=timeout)


BUILTIN_TOOLS: list[ai.Tool] = [read, write, edit, ls, glob, grep, bash]
