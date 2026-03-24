from __future__ import annotations

import abc
import dataclasses
import json
from collections.abc import AsyncGenerator, Sequence

import pydantic

from ...types import messages as messages_
from ...types import tools as tools_


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

    message_id: str = dataclasses.field(default_factory=messages_._gen_id)

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

            case FileEvent(block_id=bid, media_type=mt, data=d):
                self._files[bid] = (mt, d)

            case MessageDone(usage=usage):
                self._is_done = True
                self._usage = usage
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

        # File parts (inline images/videos from LLMs like Gemini, GPT-5)
        for _bid, (media_type, data) in self._files.items():
            parts.append(messages_.FilePart(data=data, media_type=media_type))

        return messages_.Message(
            id=self.message_id,
            role="assistant",
            parts=parts,
            usage=self._usage if self._is_done else None,
        )


class LanguageModel(abc.ABC):
    @abc.abstractmethod
    async def stream_events(
        self,
        messages: list[messages_.Message],
        tools: Sequence[tools_.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
    ) -> AsyncGenerator[StreamEvent]:
        raise NotImplementedError
        yield

    async def stream(
        self,
        messages: list[messages_.Message],
        tools: Sequence[tools_.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
    ) -> AsyncGenerator[messages_.Message]:
        """Stream Messages (uses StreamHandler internally)."""
        handler = StreamHandler()
        msg: messages_.Message | None = None
        async for event in self.stream_events(messages, tools, output_type=output_type):
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
            msg = msg.model_copy()
            msg.parts = [*msg.parts, part]
            yield msg

    async def buffer(
        self,
        messages: list[messages_.Message],
        tools: Sequence[tools_.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
    ) -> messages_.Message:
        """Drain the stream and return the final message."""
        final = None
        async for msg in self.stream(messages, tools, output_type=output_type):
            final = msg
        if final is None:
            raise ValueError("LLM produced no messages")
        return final
