from __future__ import annotations

import json
import uuid
from typing import Annotated, Any, Literal, TYPE_CHECKING

import pydantic

if TYPE_CHECKING:
    from . import tools as tools_


# Streaming state for parts
PartState = Literal["streaming", "done"]


class TextPart(pydantic.BaseModel):
    text: str
    type: Literal["text"] = "text"
    # Streaming state
    state: PartState | None = None  # None = finalized/restored from storage
    delta: str | None = None  # Current delta, None when not actively streaming


class ToolPart(pydantic.BaseModel):
    tool_call_id: str
    tool_name: str
    tool_args: str
    status: Literal["pending", "result"] = "pending"  # Execution status
    result: dict[str, Any] | None = None
    type: Literal["tool"] = "tool"
    # Streaming state (for args streaming)
    state: PartState | None = None
    args_delta: str | None = None  # Delta for tool_args

    def set_result(self, result: Any) -> None:
        """Set the tool result and mark as completed."""
        self.status = "result"
        self.result = result

    async def execute(self) -> Any:
        """Execute this tool call using the global tool registry.

        Looks up the tool by name, parses args, injects Runtime if needed,
        and updates this part with the result.
        """
        from . import tools as tools_
        from . import runtime as runtime_

        tool = tools_.get_tool(self.tool_name)
        if tool is None:
            raise ValueError(f"Tool not found in registry: {self.tool_name}")

        kwargs: dict[str, Any] = json.loads(self.tool_args) if self.tool_args else {}

        # Inject runtime if the tool has a Runtime-typed parameter
        rt = runtime_._runtime.get(None)
        if rt and (runtime_param := runtime_._find_runtime_param(tool.fn)):
            kwargs[runtime_param] = rt

        result = await tool.fn(**kwargs)
        self.set_result(result)
        return result


class ReasoningPart(pydantic.BaseModel):
    text: str
    type: Literal["reasoning"] = "reasoning"
    # Anthropic's thinking blocks include a signature for cache/verification.
    # This must be preserved and sent back in multi-turn conversations.
    signature: str | None = None
    # Streaming state
    state: PartState | None = None
    delta: str | None = None


class HookPart(pydantic.BaseModel):
    """Part representing a hook suspension point in the agent's turn."""

    hook_id: str
    hook_type: str
    status: Literal[
        "pending", "resolved", "cancelled"
    ]  # TODO should be shared with hook type
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
    resolution: dict[str, Any] | None = None  # TODO should have payload type
    type: Literal["hook"] = "hook"


Part = Annotated[
    TextPart | ToolPart | ReasoningPart | HookPart,
    pydantic.Field(discriminator="type"),
]


def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


class ToolDelta(pydantic.BaseModel):
    tool_call_id: str
    tool_name: str
    args_delta: str


class Message(pydantic.BaseModel):
    role: Literal["user", "assistant", "system"]
    parts: list[Part]
    id: str = pydantic.Field(default_factory=_gen_id)
    label: str | None = None

    @property
    def is_done(self) -> bool:
        """Message is done when all parts are done (or have no streaming state)."""
        for part in self.parts:
            if isinstance(part, (TextPart, ReasoningPart, ToolPart)):
                if part.state == "streaming":
                    return False
        return True

    @property
    def text_delta(self) -> str:
        """Get current text delta from parts."""
        for part in self.parts:
            if isinstance(part, TextPart) and part.delta:
                return part.delta
        return ""

    @property
    def reasoning_delta(self) -> str:
        """Get current reasoning delta from parts."""
        for part in self.parts:
            if isinstance(part, ReasoningPart) and part.delta:
                return part.delta
        return ""

    @property
    def tool_deltas(self) -> list[ToolDelta]:
        """Get current tool deltas from parts."""
        deltas = []
        for part in self.parts:
            if isinstance(part, ToolPart) and part.args_delta:
                deltas.append(
                    ToolDelta(
                        tool_call_id=part.tool_call_id,
                        tool_name=part.tool_name,
                        args_delta=part.args_delta,
                    )
                )
        return deltas

    @property
    def text(self) -> str:
        for part in self.parts:
            if isinstance(part, TextPart):
                return part.text
        return ""

    @property
    def reasoning(self) -> str:
        for part in self.parts:
            if isinstance(part, ReasoningPart):
                return part.text
        return ""

    @property
    def tool_calls(self) -> list[ToolPart]:
        # TODO properly validate args?
        return [part for part in self.parts if isinstance(part, ToolPart)]

    def get_tool_part(self, tool_call_id: str) -> ToolPart | None:
        for part in self.parts:
            if isinstance(part, ToolPart) and part.tool_call_id == tool_call_id:
                return part
        return None


def make_messages(*, system: str | None = None, user: str) -> list[Message]:
    """Convenience builder for common system + user message pattern."""
    result: list[Message] = []
    if system is not None:
        result.append(Message(role="system", parts=[TextPart(text=system)]))
    result.append(Message(role="user", parts=[TextPart(text=user)]))
    return result
