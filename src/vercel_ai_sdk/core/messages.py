from __future__ import annotations

import importlib
import uuid
from typing import Annotated, Any, Literal

import pydantic

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
    status: Literal["pending", "result", "error"] = "pending"  # Execution status
    result: Any = None
    type: Literal["tool"] = "tool"
    # Streaming state (for args streaming)
    state: PartState | None = None
    args_delta: str | None = None  # Delta for tool_args

    def set_result(self, result: Any) -> None:
        """Set the tool result and mark as completed."""
        self.status = "result"
        self.result = result

    def set_error(self, message: str) -> None:
        """Set a tool error and mark as failed."""
        self.status = "error"
        self.result = message


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


def _resolve_class(fully_qualified_name: str) -> type[pydantic.BaseModel]:
    """Import and return a class from its fully qualified name.

    E.g. ``"myapp.models.WeatherForecast"`` → the ``WeatherForecast`` class.
    """
    module_path, _, class_name = fully_qualified_name.rpartition(".")
    if not module_path:
        raise ImportError(
            f"Cannot resolve '{fully_qualified_name}': "
            "expected a fully qualified name like 'mypackage.module.ClassName'"
        )
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Module '{module_path}' has no attribute '{class_name}'")
    if not (isinstance(cls, type) and issubclass(cls, pydantic.BaseModel)):
        raise TypeError(
            f"'{fully_qualified_name}' is not a pydantic.BaseModel subclass"
        )
    return cls


class StructuredOutputPart(pydantic.BaseModel):
    """Part containing a validated structured output from the LLM.

    ``data`` stores the parsed JSON dict (always serializable).
    ``output_type_name`` stores the fully qualified class name so the typed
    Pydantic model can be lazily rehydrated via the ``value`` property.
    """

    data: dict[str, Any]
    output_type_name: str
    type: Literal["structured_output"] = "structured_output"

    _hydrated: Any = pydantic.PrivateAttr(default=None)

    @property
    def value(self) -> Any:
        """Lazily resolve the output type class and validate ``data`` into it."""
        if self._hydrated is None:
            cls = _resolve_class(self.output_type_name)
            self._hydrated = cls.model_validate(self.data)
        return self._hydrated


Part = Annotated[
    TextPart | ToolPart | ReasoningPart | HookPart | StructuredOutputPart,
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
    def output(self) -> Any:
        """Return the validated structured output, or None."""
        for part in self.parts:
            if isinstance(part, StructuredOutputPart):
                return part.value
        return None

    @property
    def is_done(self) -> bool:
        """Message is done when all parts are done (or have no streaming state)."""
        for part in self.parts:
            if (
                isinstance(part, (TextPart, ReasoningPart, ToolPart))
                and part.state == "streaming"
            ):
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

    def get_hook_part(self, hook_id: str | None = None) -> HookPart | None:
        """Find a HookPart by hook_id, or return the first HookPart if no id given."""
        for part in self.parts:
            if isinstance(part, HookPart) and (
                hook_id is None or part.hook_id == hook_id
            ):
                return part
        return None


def make_messages(*, system: str | None = None, user: str) -> list[Message]:
    """Convenience builder for common system + user message pattern."""
    result: list[Message] = []
    if system is not None:
        result.append(Message(role="system", parts=[TextPart(text=system)]))
    result.append(Message(role="user", parts=[TextPart(text=user)]))
    return result
