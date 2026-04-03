from . import adapters, models, models2, telemetry
from .adapters import ai_sdk_ui
from .agents import (
    Checkpoint,
    Hook,
    HookInfo,
    PendingHookInfo,
    RunResult,
    Runtime,
    StreamResult,
    Tool,
    ToolApproval,
    execute_tool,
    get_checkpoint,
    hook,
    mcp,
    run,
    stream,
    stream_loop,
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
    # Agents (from agents/)
    "Tool",
    "Runtime",
    "RunResult",
    "HookInfo",
    "StreamResult",
    "Hook",
    "ToolApproval",
    "Checkpoint",
    "PendingHookInfo",
    # Functions (from agents/)
    "tool",
    "stream",
    "stream_step",
    "stream_loop",
    "execute_tool",
    "get_checkpoint",
    "run",
    "hook",
    # Submodules
    "telemetry",
    "mcp",
    "ai_sdk_ui",
    "adapters",
]
