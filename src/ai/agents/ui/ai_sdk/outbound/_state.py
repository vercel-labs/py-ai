"""Stream state bookkeeping for the event-first outbound walk."""

from __future__ import annotations

from typing import Any

from .....types import events as events_
from .....types import messages as messages_
from .. import _approvals, protocol


def _tool_error_text(part: messages_.ToolResultPart) -> str:
    """Best-effort error text extraction from a failed tool result."""
    if isinstance(part.result, str) and part.result:
        return part.result
    if isinstance(part.result, dict):
        for key in ("error", "message", "detail"):
            value = part.result.get(key)
            if isinstance(value, str) and value:
                return value
    return "Tool execution failed"


class _StreamState:
    """Single-pass state across one ``to_stream()`` call."""

    def __init__(self) -> None:
        self.current_turn_id: str | None = None
        self.current_agent: str | None = None
        self.ui_message_id: str | None = None
        self.emitted_start: bool = False
        self.in_step: bool = False

        self.seen_done: set[str] = set()
        self.skip_current_message: bool = False
        self.started_current_message: bool = False

        self.started_tool_inputs: set[str] = set()
        self.tool_names: dict[str, str] = {}
        self.input_available_emitted: set[str] = set()
        self.emitted_tool_results: set[str] = set()
        self.emitted_approval_requests: set[str] = set()

        self.open_text_ids: set[str] = set()
        self.open_reasoning_ids: set[str] = set()
        self.completed_text_ids: set[str] = set()
        self.completed_reasoning_ids: set[str] = set()
        self.text_delta_ids: set[str] = set()
        self.reasoning_delta_ids: set[str] = set()

    # -- boundary helpers ----------------------------------------------------

    def _close_open_blocks(self) -> list[protocol.UIMessageStreamPart]:
        parts: list[protocol.UIMessageStreamPart] = []
        for rid in list(self.open_reasoning_ids):
            parts.append(protocol.ReasoningEndPart(id=rid))
            self.completed_reasoning_ids.add(rid)
        self.open_reasoning_ids.clear()
        for tid in list(self.open_text_ids):
            parts.append(protocol.TextEndPart(id=tid))
            self.completed_text_ids.add(tid)
        self.open_text_ids.clear()
        return parts

    def _finish_step(self) -> list[protocol.UIMessageStreamPart]:
        parts = self._close_open_blocks()
        if self.in_step:
            parts.append(protocol.FinishStepPart())
            self.in_step = False
        return parts

    def _reset_step_tracking(self) -> None:
        self.started_tool_inputs.clear()
        self.tool_names.clear()
        self.input_available_emitted.clear()
        self.emitted_tool_results.clear()
        self.emitted_approval_requests.clear()

    @staticmethod
    def _is_visible_message(msg: messages_.Message) -> bool:
        return msg.role not in ("user", "system")

    # -- phase: message start ------------------------------------------------

    def on_message_start(
        self, msg: messages_.Message | None
    ) -> list[protocol.UIMessageStreamPart]:
        self.started_current_message = False
        self.skip_current_message = False
        if msg is None:
            return []
        if msg.id in self.seen_done or not self._is_visible_message(msg):
            self.skip_current_message = True
            return []
        self.started_current_message = True
        return self.on_message(msg)

    def on_message(self, msg: messages_.Message) -> list[protocol.UIMessageStreamPart]:
        """Emit UIMessage/step boundary parts for *msg*."""
        if not self._is_visible_message(msg):
            return []

        parts: list[protocol.UIMessageStreamPart] = []
        agent_changed = (
            self.emitted_start
            and msg.source_label is not None
            and msg.source_label != self.current_agent
        )

        if not self.emitted_start or agent_changed:
            parts.extend(self._finish_step())
            if self.emitted_start:
                parts.append(protocol.FinishPart(finish_reason="stop"))

            self.ui_message_id = msg.id
            parts.append(protocol.StartPart(message_id=msg.id))
            parts.append(protocol.StartStepPart())
            self.emitted_start = True
            self.in_step = True
            self.current_agent = msg.source_label
            self.current_turn_id = msg.turn_id
            self._reset_step_tracking()
            return parts

        if (
            msg.turn_id is not None
            and self.current_turn_id is not None
            and msg.turn_id != self.current_turn_id
        ):
            parts.extend(self._finish_step())
            parts.append(protocol.StartStepPart())
            self.in_step = True
            self._reset_step_tracking()
            self.current_turn_id = msg.turn_id
        elif msg.turn_id is not None and self.current_turn_id is None:
            self.current_turn_id = msg.turn_id

        return parts

    # -- phase: streaming events --------------------------------------------

    def on_event(self, event: events_.Event) -> list[protocol.UIMessageStreamPart]:
        if self.skip_current_message:
            return []

        match event:
            case events_.TextStart(block_id=pid):
                self.open_text_ids.add(pid)
                return [protocol.TextStartPart(id=pid)]

            case events_.TextDelta(block_id=pid, chunk=chunk):
                out: list[protocol.UIMessageStreamPart] = []
                if pid not in self.open_text_ids:
                    self.open_text_ids.add(pid)
                    out.append(protocol.TextStartPart(id=pid))
                self.text_delta_ids.add(pid)
                out.append(protocol.TextDeltaPart(id=pid, delta=chunk))
                return out

            case events_.TextEnd(block_id=pid):
                if pid in self.open_text_ids:
                    self.open_text_ids.discard(pid)
                    self.completed_text_ids.add(pid)
                    return [protocol.TextEndPart(id=pid)]
                return []

            case events_.ReasoningStart(block_id=pid):
                self.open_reasoning_ids.add(pid)
                return [protocol.ReasoningStartPart(id=pid)]

            case events_.ReasoningDelta(block_id=pid, chunk=chunk):
                out = []
                if pid not in self.open_reasoning_ids:
                    self.open_reasoning_ids.add(pid)
                    out.append(protocol.ReasoningStartPart(id=pid))
                self.reasoning_delta_ids.add(pid)
                out.append(protocol.ReasoningDeltaPart(id=pid, delta=chunk))
                return out

            case events_.ReasoningEnd(block_id=pid):
                if pid in self.open_reasoning_ids:
                    self.open_reasoning_ids.discard(pid)
                    self.completed_reasoning_ids.add(pid)
                    return [protocol.ReasoningEndPart(id=pid)]
                return []

            case events_.ToolStart(tool_call_id=tcid, tool_name=name):
                self.tool_names[tcid] = name
                if tcid in self.started_tool_inputs:
                    return []
                self.started_tool_inputs.add(tcid)
                return [
                    protocol.ToolInputStartPart(
                        tool_call_id=tcid,
                        tool_name=name,
                    )
                ]

            case events_.ToolDelta(tool_call_id=tcid, chunk=chunk):
                out = []
                if tcid not in self.started_tool_inputs:
                    self.started_tool_inputs.add(tcid)
                    out.append(
                        protocol.ToolInputStartPart(
                            tool_call_id=tcid,
                            tool_name=self.tool_names.get(tcid, ""),
                        )
                    )
                out.append(
                    protocol.ToolInputDeltaPart(
                        tool_call_id=tcid,
                        input_text_delta=chunk,
                    )
                )
                return out

            case events_.ToolEnd():
                return []

        return []

    # -- phase: terminal message --------------------------------------------

    def _static_content(
        self, msg: messages_.Message
    ) -> list[protocol.UIMessageStreamPart]:
        out: list[protocol.UIMessageStreamPart] = []

        for part in msg.parts:
            if isinstance(part, messages_.ReasoningPart):
                if part.id not in self.completed_reasoning_ids:
                    if part.id not in self.open_reasoning_ids:
                        out.append(protocol.ReasoningStartPart(id=part.id))
                    if part.text and part.id not in self.reasoning_delta_ids:
                        out.append(
                            protocol.ReasoningDeltaPart(id=part.id, delta=part.text)
                        )
                    out.append(protocol.ReasoningEndPart(id=part.id))
                    self.open_reasoning_ids.discard(part.id)
                    self.completed_reasoning_ids.add(part.id)

            elif isinstance(part, messages_.TextPart):
                if part.id not in self.completed_text_ids:
                    if part.id not in self.open_text_ids:
                        out.append(protocol.TextStartPart(id=part.id))
                    if part.text and part.id not in self.text_delta_ids:
                        out.append(protocol.TextDeltaPart(id=part.id, delta=part.text))
                    out.append(protocol.TextEndPart(id=part.id))
                    self.open_text_ids.discard(part.id)
                    self.completed_text_ids.add(part.id)

            elif isinstance(part, messages_.FilePart):
                out.append(
                    protocol.FilePart(
                        url=part.data if isinstance(part.data, str) else "",
                        media_type=part.media_type,
                    )
                )

        return out

    def on_terminal(self, msg: messages_.Message) -> list[protocol.UIMessageStreamPart]:
        if msg.id in self.seen_done or not self._is_visible_message(msg):
            self.seen_done.add(msg.id)
            return []

        out: list[protocol.UIMessageStreamPart] = []
        if not self.started_current_message:
            out.extend(self.on_message(msg))

        out.extend(self._static_content(msg))

        for part in msg.parts:
            if isinstance(part, messages_.ToolCallPart):
                if part.tool_call_id in self.input_available_emitted:
                    continue
                self.input_available_emitted.add(part.tool_call_id)
                if part.tool_call_id not in self.started_tool_inputs:
                    self.started_tool_inputs.add(part.tool_call_id)
                    out.append(
                        protocol.ToolInputStartPart(
                            tool_call_id=part.tool_call_id,
                            tool_name=part.tool_name,
                        )
                    )
                out.append(
                    protocol.ToolInputAvailablePart(
                        tool_call_id=part.tool_call_id,
                        tool_name=part.tool_name,
                        input=part.tool_args,
                    )
                )

            elif isinstance(part, messages_.ToolResultPart):
                if part.tool_call_id in self.emitted_tool_results:
                    continue
                self.emitted_tool_results.add(part.tool_call_id)
                if part.is_error:
                    out.append(
                        protocol.ToolOutputErrorPart(
                            tool_call_id=part.tool_call_id,
                            error_text=_tool_error_text(part),
                        )
                    )
                else:
                    out.append(
                        protocol.ToolOutputAvailablePart(
                            tool_call_id=part.tool_call_id,
                            output=part.result,
                        )
                    )

            elif isinstance(part, messages_.HookPart):
                tc_id = _approvals.tool_call_id_for(part)
                if tc_id is None:
                    continue

                if part.status == "pending":
                    if tc_id in self.emitted_approval_requests:
                        continue
                    self.emitted_approval_requests.add(tc_id)
                    out.append(
                        protocol.ToolApprovalRequestPart(
                            approval_id=part.hook_id,
                            tool_call_id=tc_id,
                        )
                    )
                elif part.status == "resolved":
                    resolution: dict[str, Any] = part.resolution or {}
                    if not resolution.get("granted", False):
                        out.append(protocol.ToolOutputDeniedPart(tool_call_id=tc_id))
                elif part.status == "cancelled":
                    out.append(
                        protocol.ToolOutputErrorPart(
                            tool_call_id=tc_id,
                            error_text="Hook cancelled",
                        )
                    )

        self.seen_done.add(msg.id)
        self.skip_current_message = False
        self.started_current_message = False
        return out

    # -- phase: stream finish ------------------------------------------------

    def finish(self) -> list[protocol.UIMessageStreamPart]:
        parts = self._finish_step()
        if self.emitted_start:
            parts.append(protocol.FinishPart(finish_reason="stop"))
        return parts
