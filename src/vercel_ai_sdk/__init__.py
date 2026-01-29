from . import anthropic, mcp, openai, ai_sdk_ui

# Re-export core types for convenient access
from .core.messages import (
    Message,
    Part,
    TextPart,
    ToolPart,
    ToolDelta,
    ReasoningPart,
    make_messages,
)
from .core.tools import Tool, tool
from .core.runtime import (
    LanguageModel,
    Runtime,
    Stream,
    execute,
    stream_loop,
    stream_step,
)
from .core.hooks import (
    hook,
    HookPending,
    set_hook_resolutions,
    reset_hook_resolutions,
    get_hook_resolutions,
)
# New step-based API
from .core.step import StepResult, ToolCall
from .core.decorators import stream
from .core.primitives import stream_llm, execute_tool

__all__ = [
    # Types
    "Message",
    "Part",
    "TextPart",
    "ToolPart",
    "ToolDelta",
    "ReasoningPart",
    "Tool",
    "Stream",
    "LanguageModel",
    "Runtime",
    # Step types
    "StepResult",
    "ToolCall",
    # Hook types
    "hook",
    "HookPending",
    # Functions
    "tool",
    "execute",
    "stream_loop",
    "stream_step",
    "make_messages",
    # New primitives
    "stream",
    "stream_llm",
    "execute_tool",
    # Hook functions
    "set_hook_resolutions",
    "reset_hook_resolutions",
    "get_hook_resolutions",
    # Submodules
    "anthropic",
    "mcp",
    "openai",
    "ai_sdk_ui",
]
