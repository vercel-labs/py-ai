from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncGenerator

import pydantic

from ....types import messages as messages_


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
    usage: messages_.Usage | None = None


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
    Accumulates LLM adapter events and produces Messages with stateful parts.

    This is the normalization layer between LLM adapters and the rest of the system.
    """

    message_id: str = dataclasses.field(default_factory=messages_.generate_id)

    # Accumulators
    _text_blocks: dict[str, str] = dataclasses.field(default_factory=dict)
    _reasoning_blocks: dict[str, tuple[str, str | None]] = dataclasses.field(
        default_factory=dict
    )  # (text, signature)
    _tool_calls: dict[str, tuple[str, str]] = dataclasses.field(
        default_factory=dict
    )  # (name, args)
    _files: dict[str, tuple[str, str]] = dataclasses.field(
        default_factory=dict
    )  # block_id -> (media_type, data)

    # Active tracking
    _active_text_id: str | None = None
    _active_reasoning_id: str | None = None
    _active_tool_ids: set[str] = dataclasses.field(default_factory=set)

    _is_done: bool = False
    _usage: messages_.Usage | None = None

    def handle_event(self, event: StreamEvent) -> messages_.Message:
        """Process event and return current Message state."""

        # Sidecar events for this yield (reset each call).
        stream_events: list[messages_.StreamEvent] = []

        match event:
            case TextStart(block_id=bid):
                self._text_blocks[bid] = ""
                self._active_text_id = bid
                stream_events.append(messages_.PartOpened(part_id=bid))

            case TextDelta(block_id=bid, delta=d):
                self._text_blocks[bid] += d
                stream_events.append(messages_.PartDelta(part_id=bid, chunk=d))

            case TextEnd(block_id=bid):
                if self._active_text_id == bid:
                    self._active_text_id = None
                stream_events.append(messages_.PartClosed(part_id=bid))

            case ReasoningStart(block_id=bid):
                self._reasoning_blocks[bid] = ("", None)
                self._active_reasoning_id = bid
                stream_events.append(messages_.PartOpened(part_id=bid))

            case ReasoningDelta(block_id=bid, delta=d):
                text, sig = self._reasoning_blocks[bid]
                self._reasoning_blocks[bid] = (text + d, sig)
                stream_events.append(messages_.PartDelta(part_id=bid, chunk=d))

            case ReasoningEnd(block_id=bid, signature=sig):
                text, _ = self._reasoning_blocks[bid]
                self._reasoning_blocks[bid] = (text, sig)
                if self._active_reasoning_id == bid:
                    self._active_reasoning_id = None
                stream_events.append(messages_.PartClosed(part_id=bid))

            case ToolStart(tool_call_id=tcid, tool_name=name):
                self._tool_calls[tcid] = (name, "")
                self._active_tool_ids.add(tcid)
                stream_events.append(messages_.PartOpened(part_id=tcid))

            case ToolArgsDelta(tool_call_id=tcid, delta=d):
                name, args = self._tool_calls[tcid]
                self._tool_calls[tcid] = (name, args + d)
                stream_events.append(messages_.PartDelta(part_id=tcid, chunk=d))

            case ToolEnd(tool_call_id=tcid):
                self._active_tool_ids.discard(tcid)
                stream_events.append(messages_.PartClosed(part_id=tcid))

            case FileEvent(block_id=bid, media_type=mt, data=d):
                self._files[bid] = (mt, d)

            case MessageDone(usage=usage):
                self._is_done = True
                self._usage = usage
                self._active_text_id = None
                self._active_reasoning_id = None
                self._active_tool_ids.clear()

        return self._build_message(stream_events)

    def _build_message(
        self,
        stream_events: list[messages_.StreamEvent],
    ) -> messages_.Message:
        parts: list[messages_.Part] = []

        # Reasoning parts first (like thinking blocks)
        for bid, (text, sig) in self._reasoning_blocks.items():
            parts.append(messages_.ReasoningPart(id=bid, text=text, signature=sig))

        # Text parts
        for bid, text in self._text_blocks.items():
            parts.append(messages_.TextPart(id=bid, text=text))

        # Tool call parts
        for tcid, (name, args) in self._tool_calls.items():
            parts.append(
                messages_.ToolCallPart(
                    id=tcid,
                    tool_call_id=tcid,
                    tool_name=name,
                    tool_args=args,
                )
            )

        # File parts (inline images/videos from LLMs like Gemini, GPT-5)
        for bid, (media_type, data) in self._files.items():
            parts.append(messages_.FilePart(id=bid, data=data, media_type=media_type))

        return messages_.Message(
            id=self.message_id,
            role="assistant",
            parts=parts,
            usage=self._usage if self._is_done else None,
            stream=messages_.StreamState(
                new_events=stream_events,
                is_done=self._is_done,
            ),
        )


async def events_to_messages(
    events: AsyncGenerator[StreamEvent],
    output_type: type[pydantic.BaseModel] | None = None,
) -> AsyncGenerator[messages_.Message]:
    """Convert a stream of events into Message snapshots.

    This is the standalone version of the logic that ``LanguageModel.stream()``
    uses.  Wire functions call this to turn their ``StreamEvent`` generators
    into ``Message`` generators suitable for ``Stream``.
    """
    handler = StreamHandler()
    msg: messages_.Message | None = None
    async for event in events:
        msg = handler.handle_event(event)
        yield msg

    # After stream completes, validate and attach structured output part
    if output_type is not None and msg is not None and msg.text:
        data = json.loads(msg.text)
        output_type.model_validate(data)  # fail fast on bad data
        part = messages_.StructuredOutputPart(
            data=data,
            output_type_name=f"{output_type.__module__}.{output_type.__qualname__}",
        )
        msg = msg.model_copy(update={"parts": [*msg.parts, part]})
        yield msg
