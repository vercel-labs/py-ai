from . import adapters, models, models2, telemetry
from .adapters import ai_sdk_ui
from .agents2 import (
    Agent,
    AgentRun,
    Checkpoint,
    Context,
    Hook,
    HookInfo,
    LoopFn,
    PendingHookInfo,
    RunResult,
    Runtime,
    StreamResult,
    Tool,
    ToolApproval,
    ToolSource,
    agent,
    execute_tool,
    get_checkpoint,
    get_context,
    hook,
    mcp,
    stream,
    stream_step,
    tool,
)
from .models2 import Client, Model, ModelCost

# Re-export core types
from .types import (
    FilePart,
    HookPart,
    Message,
    Part,
    PartState,
    ReasoningPart,
    StructuredOutputPart,
    TextPart,
    ToolDelta,
    ToolLike,
    ToolPart,
    ToolSchema,
    Usage,
    make_messages,
)

__all__ = [
    # Types (from types/)
    "Message",
    "Part",
    "PartState",
    "TextPart",
    "ToolPart",
    "ToolDelta",
    "ReasoningPart",
    "FilePart",
    "HookPart",
    "StructuredOutputPart",
    "ToolLike",
    "ToolSchema",
    "Usage",
    "make_messages",
    # Models (from models2/)
    "Model",
    "ModelCost",
    "Client",
    "models2",
    # Legacy (from models/) — kept during transition
    "models",
    # Agents — primary API
    "Agent",
    "AgentRun",
    "agent",
    "LoopFn",
    # Agents — composition primitives
    "stream_step",
    "execute_tool",
    "get_checkpoint",
    "stream",
    "StreamResult",
    # Agents — tools
    "Tool",
    "tool",
    # Agents — hooks
    "Hook",
    "hook",
    "ToolApproval",
    # Agents — context
    "Context",
    "ToolSource",
    "get_context",
    # Agents — runtime (developer API)
    "Runtime",
    "RunResult",
    "HookInfo",
    # Agents — checkpoint
    "Checkpoint",
    "PendingHookInfo",
    # Submodules
    "telemetry",
    "mcp",
    "ai_sdk_ui",
    "adapters",
]
