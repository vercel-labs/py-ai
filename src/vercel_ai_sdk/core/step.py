"""Step protocol and result types for the execution model."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import Any

from . import messages as messages_


@dataclass
class ToolCall:
    """A tool call extracted from an LLM response."""

    tool_call_id: str
    tool_name: str
    tool_args: dict[str, Any]


@dataclass
class StepResult:
    """Result of executing a step - serializable for durability replay."""

    messages: list[messages_.Message] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def last_message(self) -> messages_.Message | None:
        return self.messages[-1] if self.messages else None

    @property
    def text(self) -> str:
        if self.last_message:
            return self.last_message.text
        return ""


# Type alias for step functions
StepFn = Callable[[], AsyncGenerator[messages_.Message, None]]
