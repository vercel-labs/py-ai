"""
Outbound: internal message stream -> AI SDK UI stream.

Converts the internal runtime stream into AI SDK UI stream protocol parts
and optionally serializes them as SSE payloads.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncGenerator, AsyncIterable

from ...agents.hooks import TOOL_APPROVAL_HOOK_TYPE
from ...types import messages as messages_
from . import protocol


def _to_camel_case(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def serialize_part(part: protocol.UIMessageStreamPart) -> str:
    """Serialize a stream part to JSON with camelCase keys."""
    d = dataclasses.asdict(part)
    if isinstance(part, protocol.DataPart):
        d["type"] = part.type
        del d["data_type"]
    camel_dict = {_to_camel_case(k): v for k, v in d.items() if v is not None}
    return json.dumps(camel_dict)


def format_sse(part: protocol.UIMessageStreamPart) -> str:
    """Format a stream part as an SSE data line."""
    return f"data: {serialize_part(part)}\n\n"


class _StreamState:
    """Tracks state for UI message stream event sequencing."""

    def __init__(self) -> None:
        self.text_id: str | None = None
        self.reasoning_id: str | None = None
        self.label: str | None = None
        self.message_id: str | None = None
        self.emitted_start = False
        self.in_step = False
        self.started_tool_calls: set[str] = set()
        self.emitted_tool_results: set[str] = set()
        self.pending_tool_calls: set[str] = set()
        self.emitted_approval_requests: set[str] = set()

    def close_open_blocks(self) -> list[protocol.UIMessageStreamPart]:
        parts: list[protocol.UIMessageStreamPart] = []
        if self.reasoning_id:
            parts.append(protocol.ReasoningEndPart(id=self.reasoning_id))
            self.reasoning_id = None
        if self.text_id:
            parts.append(protocol.TextEndPart(id=self.text_id))
            self.text_id = None
        return parts

    def finish_step(self) -> list[protocol.UIMessageStreamPart]:
        parts = self.close_open_blocks()
        if self.in_step:
            parts.append(protocol.FinishStepPart())
            self.in_step = False
        return parts

    def reset_tool_tracking(self) -> None:
        self.started_tool_calls = set()
        self.emitted_tool_results = set()
        self.pending_tool_calls = set()
        self.emitted_approval_requests = set()

    def begin_message(self, msg: messages_.Message) -> list[protocol.UIMessageStreamPart]:
        parts: list[protocol.UIMessageStreamPart] = []
        is_new_message = self.message_id is not None and msg.id != self.message_id

        if not self.emitted_start or (msg.label and msg.label != self.label):
            parts.extend(self.finish_step())
            if self.emitted_start:
                parts.append(protocol.FinishPart(finish_reason="stop"))

            parts.append(protocol.StartPart(message_id=msg.id))
            parts.append(protocol.StartStepPart())
            self.emitted_start = True
            self.in_step = True
            self.label = msg.label
            self.message_id = msg.id
            self.reset_tool_tracking()
        elif is_new_message:
            parts.extend(self.finish_step())
            parts.append(protocol.StartStepPart())
            self.in_step = True
            self.message_id = msg.id

        return parts


def _tool_call_id_from_approval_hook(hook_part: messages_.HookPart) -> str | None:
    """Extract tool_call_id from a ToolApproval HookPart."""
    if hook_part.hook_type != TOOL_APPROVAL_HOOK_TYPE:
        return None
    prefix = "approve_"
    if hook_part.hook_id.startswith(prefix):
        return hook_part.hook_id[len(prefix) :]
    return None


def _is_tool_approval_hook_message(msg: messages_.Message) -> bool:
    """True if this message contains only ToolApproval HookParts."""
    if not msg.parts:
        return False
    return all(
        isinstance(p, messages_.HookPart)
        and _tool_call_id_from_approval_hook(p) is not None
        for p in msg.parts
    )


def _tool_error_text(part: messages_.ToolResultPart) -> str:
    """Best-effort error text for failed tool executions."""
    if isinstance(part.result, str) and part.result:
        return part.result
    if isinstance(part.result, dict):
        for key in ("error", "message", "detail"):
            value = part.result.get(key)
            if isinstance(value, str) and value:
                return value
    return "Tool execution failed"


async def to_ui_message_stream(
    messages: AsyncIterable[messages_.Message],
) -> AsyncGenerator[protocol.UIMessageStreamPart]:
    """
    Convert an internal message stream into AI SDK UI stream parts.
    """
    state = _StreamState()

    async for msg in messages:
        if msg.role == "tool" and state.message_id:
            msg = msg.model_copy(update={"id": state.message_id})

        if _is_tool_approval_hook_message(msg) and state.message_id:
            msg = msg.model_copy(update={"id": state.message_id})

        for part in state.begin_message(msg):
            yield part

        if delta := msg.reasoning_delta:
            if not state.reasoning_id:
                state.reasoning_id = messages_.generate_id("reasoning")
                yield protocol.ReasoningStartPart(id=state.reasoning_id)
            yield protocol.ReasoningDeltaPart(id=state.reasoning_id, delta=delta)

        if delta := msg.text_delta:
            if state.reasoning_id:
                yield protocol.ReasoningEndPart(id=state.reasoning_id)
                state.reasoning_id = None

            if not state.text_id:
                state.text_id = messages_.generate_id("text")
                yield protocol.TextStartPart(id=state.text_id)
            yield protocol.TextDeltaPart(id=state.text_id, delta=delta)

        for tool_delta in msg.tool_deltas:
            if tool_delta.tool_call_id not in state.started_tool_calls:
                state.started_tool_calls.add(tool_delta.tool_call_id)
                yield protocol.ToolInputStartPart(
                    tool_call_id=tool_delta.tool_call_id,
                    tool_name=tool_delta.tool_name,
                )
            yield protocol.ToolInputDeltaPart(
                tool_call_id=tool_delta.tool_call_id,
                input_text_delta=tool_delta.args_delta,
            )

        if msg.is_done:
            had_active_text = state.text_id is not None
            for part in state.close_open_blocks():
                yield part

            has_new_pending_tools = any(
                isinstance(p, messages_.ToolCallPart)
                and p.tool_call_id not in state.pending_tool_calls
                for p in msg.parts
            )
            has_new_tool_results = any(
                isinstance(p, messages_.ToolResultPart)
                and p.tool_call_id not in state.emitted_tool_results
                for p in msg.parts
            )

            for msg_part in msg.parts:
                match msg_part:
                    case messages_.TextPart(text=text) if (
                        text
                        and not had_active_text
                        and not has_new_pending_tools
                        and not has_new_tool_results
                    ):
                        text_id = messages_.generate_id("text")
                        yield protocol.TextStartPart(id=text_id)
                        yield protocol.TextEndPart(id=text_id)
                    case messages_.ToolCallPart(
                        tool_call_id=tc_id,
                        tool_name=name,
                        tool_args=args,
                    ):
                        if tc_id not in state.started_tool_calls:
                            state.started_tool_calls.add(tc_id)
                            yield protocol.ToolInputStartPart(
                                tool_call_id=tc_id,
                                tool_name=name,
                            )
                        if tc_id not in state.pending_tool_calls:
                            state.pending_tool_calls.add(tc_id)
                            yield protocol.ToolInputAvailablePart(
                                tool_call_id=tc_id,
                                tool_name=name,
                                input=args,
                            )

            if has_new_tool_results:
                for msg_part in msg.parts:
                    if not isinstance(msg_part, messages_.ToolResultPart):
                        continue
                    if msg_part.tool_call_id in state.emitted_tool_results:
                        continue

                    state.emitted_tool_results.add(msg_part.tool_call_id)
                    state.pending_tool_calls.discard(msg_part.tool_call_id)

                    if msg_part.is_error:
                        yield protocol.ToolOutputErrorPart(
                            tool_call_id=msg_part.tool_call_id,
                            error_text=_tool_error_text(msg_part),
                        )
                    else:
                        yield protocol.ToolOutputAvailablePart(
                            tool_call_id=msg_part.tool_call_id,
                            output=msg_part.result,
                        )

            for msg_part in msg.parts:
                if not isinstance(msg_part, messages_.HookPart):
                    continue
                approval_tc_id = _tool_call_id_from_approval_hook(msg_part)
                if approval_tc_id is None:
                    continue

                if msg_part.status == "pending":
                    if approval_tc_id not in state.emitted_approval_requests:
                        state.emitted_approval_requests.add(approval_tc_id)
                        yield protocol.ToolApprovalRequestPart(
                            approval_id=msg_part.hook_id,
                            tool_call_id=approval_tc_id,
                        )
                elif msg_part.status == "resolved":
                    resolution = msg_part.resolution or {}
                    if not resolution.get("granted", False):
                        yield protocol.ToolOutputDeniedPart(
                            tool_call_id=approval_tc_id,
                        )
                elif msg_part.status == "cancelled":
                    yield protocol.ToolOutputErrorPart(
                        tool_call_id=approval_tc_id,
                        error_text="Hook cancelled",
                    )

    for part in state.finish_step():
        yield part
    if state.emitted_start:
        yield protocol.FinishPart(finish_reason="stop")


async def filter_by_label(
    messages: AsyncIterable[messages_.Message],
    label: str | None = None,
) -> AsyncGenerator[messages_.Message]:
    """Filter a message stream to a single agent label."""
    async for msg in messages:
        if label is None:
            label = msg.label
        if msg.label == label:
            yield msg


async def to_sse_stream(
    messages: AsyncIterable[messages_.Message],
) -> AsyncGenerator[str]:
    """Convert an internal message stream into SSE strings."""
    async for part in to_ui_message_stream(messages):
        yield format_sse(part)


# Backward-compatible aliases for the current package surface.
stream_to_ui = to_ui_message_stream
stream_to_sse = to_sse_stream
