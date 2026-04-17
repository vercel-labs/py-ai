from __future__ import annotations

import importlib
import uuid
from typing import Annotated, Any, Literal, overload

import pydantic


def generate_id(prefix: str | None = None) -> str:
    """Generate a short random ID for messages and parts."""
    raw = uuid.uuid4().hex[:12]
    return f"{prefix}_{raw}" if prefix else raw


class TextPart(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    id: str = pydantic.Field(default_factory=generate_id)
    text: str
    type: Literal["text"] = "text"


class ToolCallPart(pydantic.BaseModel):
    """A tool invocation requested by the LLM.

    Lives inside ``role="assistant"`` messages.  The corresponding result
    (if any) will appear as a :class:`ToolResultPart` in a separate
    ``role="tool"`` message, linked by ``tool_call_id``.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str = pydantic.Field(default_factory=generate_id)
    tool_call_id: str
    tool_name: str
    tool_args: str
    type: Literal["tool_call"] = "tool_call"


class ToolResultPart(pydantic.BaseModel):
    """The result of executing a tool call.

    Lives inside ``role="tool"`` messages.  Back-references the
    originating call via ``tool_call_id``.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str = pydantic.Field(default_factory=generate_id)
    tool_call_id: str
    tool_name: str
    result: Any = None
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


class ReasoningPart(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    id: str = pydantic.Field(default_factory=generate_id)
    text: str
    type: Literal["reasoning"] = "reasoning"
    # Anthropic's thinking blocks include a signature for cache/verification.
    # This must be preserved and sent back in multi-turn conversations.
    signature: str | None = None


class HookPart(pydantic.BaseModel):
    """Part representing a hook suspension point in the agent's turn."""

    model_config = pydantic.ConfigDict(frozen=True)

    id: str = pydantic.Field(default_factory=generate_id)
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

    model_config = pydantic.ConfigDict(frozen=True)

    id: str = pydantic.Field(default_factory=generate_id)
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


class FilePart(pydantic.BaseModel):
    """File, image, or audio content part.

    Covers images (``image/*``), documents (``application/pdf``, ``text/*``),
    and audio (``audio/*``).  The ``media_type`` field tells provider
    converters how to format this part for each API.

    ``data`` accepts:

    * **str** -- a URL (``http(s)://...`` or ``data:...``) *or* raw base-64 text.
    * **bytes** -- raw binary data (will be base-64 encoded when serialized
      to JSON for providers that need it).
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str = pydantic.Field(default_factory=generate_id)
    data: str | bytes
    media_type: str  # IANA media type, e.g. "image/png", "audio/wav"
    filename: str | None = None
    type: Literal["file"] = "file"

    @classmethod
    def from_url(cls, url: str, *, media_type: str | None = None) -> FilePart:
        """Create from a URL, inferring ``media_type`` from the URL if omitted.

        Inference handles ``data:`` URLs (the media type is embedded in the
        prefix) and ``http(s)://`` URLs (via :func:`mimetypes.guess_type`).
        Raises :class:`ValueError` if inference fails and no explicit
        ``media_type`` is provided.
        """
        if media_type is None:
            from . import media as media_helpers

            media_type = media_helpers.infer_media_type(url)
        return cls(data=url, media_type=media_type)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        media_type: str | None = None,
        filename: str | None = None,
    ) -> FilePart:
        """Create from raw bytes, detecting ``media_type`` via magic bytes.

        Attempts image detection first, then audio.  Raises
        :class:`ValueError` if no ``media_type`` is provided and
        detection fails.
        """
        if media_type is None:
            from . import media as media_helpers

            media_type = media_helpers.detect_image_media_type(
                data
            ) or media_helpers.detect_audio_media_type(data)
        if media_type is None:
            raise ValueError(
                "Cannot detect media_type from bytes. Provide media_type explicitly."
            )
        return cls(data=data, media_type=media_type, filename=filename)


Part = Annotated[
    TextPart
    | ToolCallPart
    | ToolResultPart
    | ReasoningPart
    | HookPart
    | StructuredOutputPart
    | FilePart,
    pydantic.Field(discriminator="type"),
]


class Usage(pydantic.BaseModel):
    """Normalized token usage from a single LLM call.

    Provides a provider-agnostic view of token consumption. Fields that a
    provider does not report are left as ``None`` (not zero) so callers
    can distinguish "not reported" from "zero tokens used".
    """

    model_config = pydantic.ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0

    # Optional breakdowns — not all providers report these.
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None

    # Pass-through of the raw provider usage payload so callers can access
    # provider-specific fields (e.g. OpenAI's accepted_prediction_tokens).
    raw: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        """input_tokens + output_tokens (always consistent)."""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        """Accumulate usage across multiple LLM calls."""

        def _add_optional(a: int | None, b: int | None) -> int | None:
            """Add two optional ints. Returns None only if both are None."""
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=_add_optional(
                self.reasoning_tokens, other.reasoning_tokens
            ),
            cache_read_tokens=_add_optional(
                self.cache_read_tokens, other.cache_read_tokens
            ),
            cache_write_tokens=_add_optional(
                self.cache_write_tokens, other.cache_write_tokens
            ),
            # Don't merge raw — it's per-call and provider-specific.
        )


class ToolDelta(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    tool_call_id: str
    tool_name: str
    args_delta: str


# ---------------------------------------------------------------------------
# Streaming sidecar — transient state excluded from persistence.
# ---------------------------------------------------------------------------


class PartOpened(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    part_id: str
    type: Literal["part_opened"] = "part_opened"


class PartDelta(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    part_id: str
    chunk: str
    type: Literal["part_delta"] = "part_delta"


class PartClosed(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    part_id: str
    type: Literal["part_closed"] = "part_closed"


StreamEvent = Annotated[
    PartOpened | PartDelta | PartClosed,
    pydantic.Field(discriminator="type"),
]


class MessageStreamState(pydantic.BaseModel):
    """Transient streaming state attached to a Message during streaming.

    ``events`` contains the events since the previous yield — never cumulative.
    ``is_done`` is True once the stream has finished.
    """

    events: list[StreamEvent] = pydantic.Field(default_factory=list)
    is_done: bool = False


class Message(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    role: Literal["user", "assistant", "system", "tool", "internal"]
    parts: list[Part]
    id: str = pydantic.Field(default_factory=generate_id)
    run_id: str | None = None
    agent: str | None = None
    usage: Usage | None = None
    stream: MessageStreamState | None = pydantic.Field(default=None, exclude=True)

    @overload
    def replace(self, new: Part, /) -> Message: ...
    @overload
    def replace(self, old: Part, new: Part, /) -> Message: ...
    def replace(self, *args: Part) -> Message:
        """Return a copy with a part replaced.

        Single arg: ``msg.replace(updated_part)`` — matches by ``id``.
        Two args: ``msg.replace(old, new)`` — matches by identity.

        Raises ValueError if the target part is not found.
        """
        if len(args) == 1:
            (new,) = args
            match_id: str | None = new.id
            match_ref = None
        elif len(args) == 2:
            old, new = args
            match_id = None
            match_ref = old
        else:
            raise TypeError(f"replace() takes 1 or 2 arguments ({len(args)} given)")
        found = False
        new_parts: list[Part] = []
        for p in self.parts:
            if not found and (
                (match_id is not None and p.id == match_id)
                or (match_ref is not None and p is match_ref)
            ):
                found = True
                new_parts.append(new)
            else:
                new_parts.append(p)
        if not found:
            if match_id is not None:
                raise ValueError(f"No part with id '{match_id}' in message")
            raise ValueError("Part not found in message")
        return self.model_copy(update={"parts": new_parts})

    @property
    def output(self) -> Any:
        """Return the validated structured output, or None."""
        for part in self.parts:
            if isinstance(part, StructuredOutputPart):
                return part.value
        return None

    @property
    def is_done(self) -> bool:
        """No sidecar (persisted/restored) means done. Otherwise ``stream.is_done``."""
        if self.stream is None:
            return True
        return self.stream.is_done

    def _parts_by_id(self) -> dict[str, Part]:
        return {p.id: p for p in self.parts}

    @property
    def text_delta(self) -> str:
        """Derive from ``stream.events`` — first PartDelta whose part is TextPart."""
        if self.stream is None:
            return ""
        parts_map = self._parts_by_id()
        for ev in self.stream.events:
            if isinstance(ev, PartDelta):
                part = parts_map.get(ev.part_id)
                if isinstance(part, TextPart):
                    return ev.chunk
        return ""

    @property
    def reasoning_delta(self) -> str:
        """First PartDelta whose part is a ReasoningPart."""
        if self.stream is None:
            return ""
        parts_map = self._parts_by_id()
        for ev in self.stream.events:
            if isinstance(ev, PartDelta):
                part = parts_map.get(ev.part_id)
                if isinstance(part, ReasoningPart):
                    return ev.chunk
        return ""

    @property
    def tool_deltas(self) -> list[ToolDelta]:
        """Derive from ``stream.events`` — PartDeltas whose parts are ToolCallPart."""
        if self.stream is None:
            return []
        parts_map = self._parts_by_id()
        deltas: list[ToolDelta] = []
        for ev in self.stream.events:
            if isinstance(ev, PartDelta):
                part = parts_map.get(ev.part_id)
                if isinstance(part, ToolCallPart):
                    deltas.append(
                        ToolDelta(
                            tool_call_id=part.tool_call_id,
                            tool_name=part.tool_name,
                            args_delta=ev.chunk,
                        )
                    )
        return deltas

    @property
    def files(self) -> list[FilePart]:
        """All file parts in the message."""
        return [p for p in self.parts if isinstance(p, FilePart)]

    @property
    def images(self) -> list[FilePart]:
        """File parts with ``image/*`` media types."""
        return [
            p
            for p in self.parts
            if isinstance(p, FilePart) and p.media_type.startswith("image/")
        ]

    @property
    def videos(self) -> list[FilePart]:
        """File parts with ``video/*`` media types."""
        return [
            p
            for p in self.parts
            if isinstance(p, FilePart) and p.media_type.startswith("video/")
        ]

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
    def tool_calls(self) -> list[ToolCallPart]:
        """All tool-call parts in this message."""
        return [part for part in self.parts if isinstance(part, ToolCallPart)]

    @property
    def tool_results(self) -> list[ToolResultPart]:
        """All tool-result parts in this message."""
        return [part for part in self.parts if isinstance(part, ToolResultPart)]

    def get_hook_part(self, hook_id: str | None = None) -> HookPart | None:
        """Find a HookPart by hook_id, or return the first HookPart if no id given."""
        for part in self.parts:
            if isinstance(part, HookPart) and (
                hook_id is None or part.hook_id == hook_id
            ):
                return part
        return None
