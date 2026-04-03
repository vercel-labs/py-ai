"""Agent loop orchestration — tools, hooks, runtime, and streaming.

Depends on types/ and models2/. Provides the loop machinery that
plugs a model into a tool-calling loop with hooks and checkpoints.
"""

from . import mcp
from .checkpoint import Checkpoint, PendingHookInfo
from .hooks import Hook, ToolApproval, hook
from .runtime import (
    EventLog,
    HookInfo,
    LoopExecutor,
    RunResult,
    Runtime,
    execute_tool,
    get_checkpoint,
    run,
    stream_loop,
    stream_step,
)
from .streams import StreamResult, stream
from .tools import Tool, ToolLike, ToolSchema, get_tool, tool

__all__ = [
    # Core loop
    "run",
    "stream_step",
    "stream_loop",
    "execute_tool",
    "get_checkpoint",
    # Runtime (composition)
    "Runtime",
    "EventLog",
    "LoopExecutor",
    "RunResult",
    "HookInfo",
    # Stream
    "stream",
    "StreamResult",
    # Tools
    "Tool",
    "ToolLike",
    "ToolSchema",
    "tool",
    "get_tool",
    # Hooks
    "Hook",
    "hook",
    "ToolApproval",
    # Checkpoint
    "Checkpoint",
    "PendingHookInfo",
    # Submodules
    "mcp",
]
