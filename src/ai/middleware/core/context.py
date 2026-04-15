from __future__ import annotations

import contextvars
import dataclasses
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any

import pydantic

from .types import messages as messages_
from .types import tools as tools_
from .types.stream import StreamResultLike

# ---------------------------------------------------------------------------
# Call context objects — frozen dataclasses with isolated mutable fields.
#
# Mutable container fields (``list``, ``dict``) are shallow-copied at
# construction via ``__post_init__`` so that middleware sees its own copy
# and cannot accidentally mutate the caller's data.  To modify fields,
# use ``dataclasses.replace(call, messages=new_msgs)`` before passing
# to ``next``.
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from .agents.agent import Tool
    from .models.core.model import Model


@dataclasses.dataclass(frozen=True)
class ModelContext:
    """Context for a model streaming call."""

    model: Model
    messages: list[messages_.Message]
    tools: Sequence[tools_.ToolLike] | None
    output_type: type[pydantic.BaseModel] | None
    kwargs: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", list(self.messages))
        if self.tools is not None:
            object.__setattr__(self, "tools", list(self.tools))
        object.__setattr__(self, "kwargs", dict(self.kwargs))


@dataclasses.dataclass(frozen=True)
class GenerateContext:
    """Context for a model generate call (images, video, etc.)."""

    model: Model
    messages: list[messages_.Message]
    params: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", list(self.messages))


@dataclasses.dataclass(frozen=True)
class ToolContext:
    """Context for a tool execution."""

    tool_call_id: str
    tool_name: str
    kwargs: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", dict(self.kwargs))


@dataclasses.dataclass(frozen=True)
class HookContext:
    """Context for a hook suspension point."""

    label: str
    payload: type[pydantic.BaseModel]
    metadata: dict[str, Any]
    interrupt_loop: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclasses.dataclass(frozen=True)
class AgentRunContext:
    """Context for an agent run."""

    model: Model
    messages: list[messages_.Message]
    tools: list[Tool[..., Any]]
    label: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", list(self.messages))
        object.__setattr__(self, "tools", list(self.tools))
