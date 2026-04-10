from . import adapters, middleware, models
from .adapters import ai_sdk_ui
from .agents import (
    TOOL_APPROVAL_HOOK_TYPE,
    Agent,
    Context,
    Tool,
    ToolApproval,
    ToolCall,
    agent,
    cancel_hook,
    hook,
    mcp,
    resolve_hook,
    tool,
    yield_from,
)
from .middleware import AgentRunContext, Middleware
from .models import (
    Client,
    ImageParams,
    Model,
    ModelCost,
    VideoParams,
    check_connection,
    generate,
    model,
    stream,
)

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
    ToolCallPart,
    ToolDelta,
    ToolLike,
    ToolResultPart,
    ToolSchema,
    Usage,
)
from .types.builders import (
    assistant_message,
    file_part,
    system_message,
    thinking,
    tool_message,
    tool_result,
    user_message,
)

__all__ = [
    # Types (from types/)
    "Message",
    "Part",
    "PartState",
    "TextPart",
    "ToolCallPart",
    "ToolResultPart",
    "ToolDelta",
    "ReasoningPart",
    "FilePart",
    "HookPart",
    "StructuredOutputPart",
    "ToolLike",
    "ToolSchema",
    "Usage",
    # Builders (from types/builders)
    "user_message",
    "assistant_message",
    "system_message",
    "tool_message",
    "tool_result",
    "file_part",
    "thinking",
    # Models (from models/)
    "Model",
    "ModelCost",
    "ImageParams",
    "VideoParams",
    "Client",
    "check_connection",
    "model",
    "models",
    "stream",
    "generate",
    # Agents — primary API
    "Agent",
    "agent",
    "Context",
    # Agents — tools
    "Tool",
    "ToolCall",
    "tool",
    # Agents — composition
    "yield_from",
    # Agents — hooks
    "hook",
    "resolve_hook",
    "cancel_hook",
    "ToolApproval",
    "TOOL_APPROVAL_HOOK_TYPE",
    # Middleware
    "AgentRunContext",
    "Middleware",
    "middleware",
    # Submodules
    "mcp",
    "ai_sdk_ui",
    "adapters",
]
