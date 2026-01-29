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
    # Legacy API (still works, backward compatible)
    stream_loop as _legacy_stream_loop,
    stream_step as _legacy_stream_step,
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
from .core.primitives import (
    raw_stream_llm,
    execute_tool,
    stream_step,
)

# Convenience: keep stream_loop pointing to legacy for now
stream_loop = _legacy_stream_loop

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
    "make_messages",
    # New step-based API
    "stream",
    "stream_step",
    "execute_tool",
    "raw_stream_llm",
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
