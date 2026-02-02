from __future__ import annotations

import abc
import dataclasses
from collections.abc import AsyncGenerator

from . import messages as messages_
from . import tools as tools_


@dataclasses.dataclass
class TextStart:
    block_id: str


@dataclasses.dataclass
class TextDelta:
    block_id: str
    delta: str


@dataclasses.dataclass
class TextEnd:
    block_id: str


@dataclasses.dataclass
class ReasoningStart:
    block_id: str


@dataclasses.dataclass
class ReasoningDelta:
    block_id: str
    delta: str


@dataclasses.dataclass
class ReasoningEnd:
    block_id: str
    signature: str | None = None


@dataclasses.dataclass
class ToolStart:
    tool_call_id: str
    tool_name: str


@dataclasses.dataclass
class ToolArgsDelta:
    tool_call_id: str
    delta: str


@dataclasses.dataclass
class ToolEnd:
    tool_call_id: str


@dataclasses.dataclass
class MessageDone:
    finish_reason: str | None = None


StreamEvent = (
    TextStart
    | TextDelta
    | TextEnd
    | ReasoningStart
    | ReasoningDelta
    | ReasoningEnd
    | ToolStart
    | ToolArgsDelta
    | ToolEnd
    | MessageDone
)


@dataclasses.dataclass
class StreamHandler:
    """
    Accumulates LLM adapter events and produces Messages with stateful parts.

    This is the normalization layer between LLM adapters and the rest of the system.
    """

    message_id: str = dataclasses.field(default_factory=messages_._gen_id)

    # Accumulators
    _text_blocks: dict[str, str] = dataclasses.field(default_factory=dict)
    _reasoning_blocks: dict[str, tuple[str, str | None]] = dataclasses.field(
        default_factory=dict
    )  # (text, signature)
    _tool_calls: dict[str, tuple[str, str]] = dataclasses.field(
        default_factory=dict
    )  # (name, args)

    # Active tracking
    _active_text_id: str | None = None
    _active_reasoning_id: str | None = None
    _active_tool_ids: set[str] = dataclasses.field(default_factory=set)

    _is_done: bool = False

    def handle_event(self, event: StreamEvent) -> messages_.Message:
        """Process event and return current Message state."""

        # Current deltas (reset each call)
        text_delta: str | None = None
        reasoning_delta: str | None = None
        tool_deltas: dict[str, str] = {}  # tool_call_id -> delta

        match event:
            case TextStart(block_id=bid):
                self._text_blocks[bid] = ""
                self._active_text_id = bid

            case TextDelta(block_id=bid, delta=d):
                self._text_blocks[bid] += d
                text_delta = d

            case TextEnd(block_id=bid):
                if self._active_text_id == bid:
                    self._active_text_id = None

            case ReasoningStart(block_id=bid):
                self._reasoning_blocks[bid] = ("", None)
                self._active_reasoning_id = bid

            case ReasoningDelta(block_id=bid, delta=d):
                text, sig = self._reasoning_blocks[bid]
                self._reasoning_blocks[bid] = (text + d, sig)
                reasoning_delta = d

            case ReasoningEnd(block_id=bid, signature=sig):
                text, _ = self._reasoning_blocks[bid]
                self._reasoning_blocks[bid] = (text, sig)
                if self._active_reasoning_id == bid:
                    self._active_reasoning_id = None

            case ToolStart(tool_call_id=tcid, tool_name=name):
                self._tool_calls[tcid] = (name, "")
                self._active_tool_ids.add(tcid)

            case ToolArgsDelta(tool_call_id=tcid, delta=d):
                name, args = self._tool_calls[tcid]
                self._tool_calls[tcid] = (name, args + d)
                tool_deltas[tcid] = d

            case ToolEnd(tool_call_id=tcid):
                self._active_tool_ids.discard(tcid)

            case MessageDone():
                self._is_done = True
                self._active_text_id = None
                self._active_reasoning_id = None
                self._active_tool_ids.clear()

        return self._build_message(text_delta, reasoning_delta, tool_deltas)

    def _build_message(
        self,
        text_delta: str | None,
        reasoning_delta: str | None,
        tool_deltas: dict[str, str],
    ) -> messages_.Message:
        parts: list[messages_.Part] = []

        # Reasoning parts first (like thinking blocks)
        for bid, (text, sig) in self._reasoning_blocks.items():
            is_active = bid == self._active_reasoning_id
            parts.append(
                messages_.ReasoningPart(
                    text=text,
                    signature=sig,
                    state="streaming" if is_active else "done",
                    delta=reasoning_delta if is_active else None,
                )
            )

        # Text parts
        for bid, text in self._text_blocks.items():
            is_active = bid == self._active_text_id
            parts.append(
                messages_.TextPart(
                    text=text,
                    state="streaming" if is_active else "done",
                    delta=text_delta if is_active else None,
                )
            )

        # Tool parts
        for tcid, (name, args) in self._tool_calls.items():
            is_active = tcid in self._active_tool_ids
            parts.append(
                messages_.ToolPart(
                    tool_call_id=tcid,
                    tool_name=name,
                    tool_args=args,
                    state="streaming" if is_active else "done",
                    args_delta=tool_deltas.get(tcid),
                )
            )

        return messages_.Message(
            id=self.message_id,
            role="assistant",
            parts=parts,
        )


class LanguageModel(abc.ABC):
    @abc.abstractmethod
    async def stream(
        self, messages: list[messages_.Message], tools: list[tools_.Tool] | None = None
    ) -> AsyncGenerator[messages_.Message, None]:
        raise NotImplementedError
        yield
