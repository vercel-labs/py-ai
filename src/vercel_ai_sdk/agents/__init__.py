"""Agent loop orchestration — tools, hooks, runtime, and streaming.

Depends on types/ and models/. Provides the loop machinery that
plugs a model into a tool-calling loop with hooks and checkpoints.
"""

from . import mcp
from .agent import Agent, AgentRun, LoopFn, agent, stream_step
from .checkpoint import Checkpoint, PendingHookInfo
from .context import Context, ToolSource, get_context
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
)
from .streams import StreamResult, stream
from .tools import Tool, ToolLike, ToolSchema, get_tool, tool

__all__ = [
    # Agent (primary user API)
    "Agent",
    "AgentRun",
    "agent",
    "LoopFn",
    # Composition primitives
    "stream_step",
    "execute_tool",
    "get_checkpoint",
    # Context
    "Context",
    "ToolSource",
    "get_context",
    # Runtime (developer API)
    "Runtime",
    "EventLog",
    "LoopExecutor",
    "RunResult",
    "HookInfo",
    "run",
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
