from __future__ import annotations

import dataclasses

from ....types import events as events_
from ....types import messages as messages_
from ....types import usage as usage_


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
class FileEvent:
    """A complete generated file from the LLM (e.g. inline image from Gemini/GPT)."""

    block_id: str
    media_type: str
    data: str  # base64 string or data-URL from the gateway


@dataclasses.dataclass
class MessageDone:
    finish_reason: str | None = None
    usage: usage_.Usage | None = None


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
    | FileEvent
    | MessageDone
)


@dataclasses.dataclass
class StreamHandler:
    """
    Accumulates LLM adapter events and produces public Event objects.

    This is the normalization layer between LLM adapters and the rest of the system.
    Parts are tracked in a single ``_current_parts`` dict keyed by block/tool id,
    updated in place as events stream in.
    """

    message_id: str = dataclasses.field(default_factory=messages_.generate_id)

    # Single source of truth for part state, keyed by id. Insertion order
    # preserves provider emission order.
    _current_parts: dict[str, messages_.Part] = dataclasses.field(default_factory=dict)

    # Active tracking
    _active_text_id: str | None = None
    _active_reasoning_id: str | None = None
    _active_tool_ids: set[str] = dataclasses.field(default_factory=set)

    _is_done: bool = False
    _usage: usage_.Usage | None = None

    def message_start(self) -> events_.MessageStart:
        """Emit a MessageStart event at the beginning of a stream."""
        return events_.MessageStart(message=self._build_message())

    def handle_event(self, event: StreamEvent) -> list[events_.Event]:
        """Process an adapter event and return public Event objects."""

        out: list[events_.Event] = []

        match event:
            case TextStart(block_id=bid):
                part: messages_.Part = messages_.TextPart(id=bid, text="")
                self._current_parts[bid] = part
                self._active_text_id = bid
                out.append(events_.TextStart(block_id=bid))

            case TextDelta(block_id=bid, delta=d):
                existing = self._current_parts[bid]
                assert isinstance(existing, messages_.TextPart)
                part = messages_.TextPart(id=bid, text=existing.text + d)
                self._current_parts[bid] = part
                out.append(events_.TextDelta(chunk=d, block_id=bid))

            case TextEnd(block_id=bid):
                if self._active_text_id == bid:
                    self._active_text_id = None
                out.append(events_.TextEnd(block_id=bid))

            case ReasoningStart(block_id=bid):
                part = messages_.ReasoningPart(id=bid, text="")
                self._current_parts[bid] = part
                self._active_reasoning_id = bid
                out.append(events_.ReasoningStart(block_id=bid))

            case ReasoningDelta(block_id=bid, delta=d):
                existing = self._current_parts[bid]
                assert isinstance(existing, messages_.ReasoningPart)
                part = messages_.ReasoningPart(
                    id=bid,
                    text=existing.text + d,
                    signature=existing.signature,
                )
                self._current_parts[bid] = part
                out.append(events_.ReasoningDelta(chunk=d, block_id=bid))

            case ReasoningEnd(block_id=bid, signature=sig):
                existing = self._current_parts[bid]
                assert isinstance(existing, messages_.ReasoningPart)
                part = messages_.ReasoningPart(
                    id=bid, text=existing.text, signature=sig
                )
                self._current_parts[bid] = part
                if self._active_reasoning_id == bid:
                    self._active_reasoning_id = None
                out.append(events_.ReasoningEnd(block_id=bid, signature=sig))

            case ToolStart(tool_call_id=tcid, tool_name=name):
                part = messages_.ToolCallPart(
                    id=tcid,
                    tool_call_id=tcid,
                    tool_name=name,
                    tool_args="",
                )
                self._current_parts[tcid] = part
                self._active_tool_ids.add(tcid)
                out.append(events_.ToolStart(tool_call_id=tcid, tool_name=name))

            case ToolArgsDelta(tool_call_id=tcid, delta=d):
                existing = self._current_parts[tcid]
                assert isinstance(existing, messages_.ToolCallPart)
                part = messages_.ToolCallPart(
                    id=tcid,
                    tool_call_id=existing.tool_call_id,
                    tool_name=existing.tool_name,
                    tool_args=existing.tool_args + d,
                )
                self._current_parts[tcid] = part
                out.append(events_.ToolDelta(chunk=d, tool_call_id=tcid))

            case ToolEnd(tool_call_id=tcid):
                self._active_tool_ids.discard(tcid)
                out.append(events_.ToolEnd(tool_call_id=tcid))

            case FileEvent(block_id=bid, media_type=mt, data=d):
                self._current_parts[bid] = messages_.FilePart(
                    id=bid, data=d, media_type=mt
                )

            case MessageDone(usage=u):
                self._is_done = True
                self._usage = u
                self._active_text_id = None
                self._active_reasoning_id = None
                self._active_tool_ids.clear()
                msg = self._build_message()
                out.append(events_.MessageEnd(message=msg, usage=u))

        return out

    def _build_message(self) -> messages_.Message:
        return messages_.Message(
            id=self.message_id,
            role="assistant",
            parts=list(self._current_parts.values()),
            usage=self._usage if self._is_done else None,
        )
