"""Replay middleware for serverless re-entry across approval round-trips."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

import ai
import pydantic

from replay_types import (
    PendingHookInfo,
    RecordedModelStep,
    RecordedToolResult,
    ReplayMetadata,
    ReplayState,
)

logger = logging.getLogger(__name__)

REPLAY_VERSION = 1


class ReplayMismatchError(RuntimeError):
    """Replay state does not match the live execution path."""


class ReplayMiddleware(ai.Middleware):
    """Replays recorded model/tool work and records new work after re-entry."""

    def __init__(
        self,
        *,
        session_id: str,
        fingerprint: str,
        model: ai.Model,
        tools: list[ai.Tool[..., Any]],
        input_message_count: int,
        replay: ReplayState | None,
    ) -> None:
        self._session_id = session_id
        self._fingerprint = fingerprint
        self._model = model
        self._tools = list(tools)
        self._input_message_count = input_message_count

        if replay is not None and (
            replay.version != REPLAY_VERSION or replay.fingerprint != fingerprint
        ):
            logger.info(
                "Ignoring stale replay for session %s (version/fingerprint mismatch)",
                session_id,
            )
            replay = None

        self._replay = replay
        self._replay_model_index = 0
        self._replay_tool_index = 0
        self._new_model_steps: list[RecordedModelStep] = []
        self._new_tool_results: list[RecordedToolResult] = []
        self._pending_hooks: dict[str, PendingHookInfo] = {}
        self._last_assistant_message_id: str | None = None
        self._replayed_outbound_messages: deque[str] = deque()

    async def wrap_model(
        self,
        call: ai.middleware.ModelContext,
        next: Callable[[ai.middleware.ModelContext], Awaitable[ai.StreamResultLike]],
    ) -> ai.StreamResultLike:
        """Replay a stored model stream when available, else record a live one."""
        replayed = self._next_replayed_model_step()
        if replayed is not None:
            logger.info(
                "Replaying model step %s for session %s",
                replayed.ordinal,
                self._session_id,
            )

            async def _from_replay() -> AsyncGenerator[ai.Message]:
                for message in deserialize_messages(replayed.messages):
                    if message.role == "assistant":
                        self._last_assistant_message_id = message.id
                    self._replayed_outbound_messages.append(
                        serialize_message_key(message)
                    )
                    yield message

            return ai.StreamResult.from_generator(_from_replay())

        stream = await next(call)
        ordinal = self._recorded_model_count()

        async def _record() -> AsyncGenerator[ai.Message]:
            snapshots: list[dict[str, Any]] = []
            async for message in stream:
                if message.role == "assistant":
                    self._last_assistant_message_id = message.id
                snapshots.append(serialize_message(message))
                yield message

            self._new_model_steps.append(
                RecordedModelStep(ordinal=ordinal, messages=snapshots)
            )

        return ai.StreamResult.from_generator(_record())

    async def wrap_tool(
        self,
        call: ai.middleware.ToolContext,
        next: Callable[[ai.middleware.ToolContext], Awaitable[ai.Message]],
    ) -> ai.Message:
        """Replay a stored tool result when available, else record the live result."""
        replayed = self._next_replayed_tool_result(call.tool_call_id, call.tool_name)
        if replayed is not None:
            logger.info(
                "Replaying tool result %s (%s) for session %s",
                replayed.tool_call_id,
                replayed.tool_name,
                self._session_id,
            )
            message = deserialize_message(replayed.message)
            self._replayed_outbound_messages.append(serialize_message_key(message))
            return message

        message = await next(call)
        self._new_tool_results.append(
            RecordedToolResult(
                ordinal=self._recorded_tool_count(),
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                message=serialize_message(message),
            )
        )
        return message

    async def wrap_hook(
        self,
        call: ai.middleware.HookContext,
        next: Callable[[ai.middleware.HookContext], Awaitable[pydantic.BaseModel]],
    ) -> pydantic.BaseModel:
        """Track interrupting hooks so the run can be resumed later."""
        try:
            result = await next(call)
        except asyncio.CancelledError:
            if call.interrupt_loop:
                self._pending_hooks[call.label] = PendingHookInfo(
                    label=call.label,
                    hook_type=call.payload.__name__,
                    metadata=dict(call.metadata),
                )
                logger.info(
                    "Recorded pending hook %s for session %s",
                    call.label,
                    self._session_id,
                )
            raise

        self._pending_hooks.pop(call.label, None)
        return result

    def has_pending_hooks(self) -> bool:
        """Return whether the run ended at an interrupting hook."""
        return bool(self._pending_hooks)

    def should_persist(self) -> bool:
        """Persist only interrupted runs that need serverless re-entry."""
        return self.has_pending_hooks()

    def build_state(self) -> ReplayState:
        """Build the next replay payload from prior prefix + newly recorded work."""
        base_model_steps = self._replay.model_steps if self._replay is not None else []
        base_tool_results = (
            self._replay.tool_results if self._replay is not None else []
        )
        return ReplayState(
            version=REPLAY_VERSION,
            session_id=self._session_id,
            fingerprint=self._fingerprint,
            model_steps=[*base_model_steps, *self._new_model_steps],
            tool_results=[*base_tool_results, *self._new_tool_results],
            pending_hooks=list(self._pending_hooks.values()),
            metadata=ReplayMetadata(
                model_id=self._model.id,
                tool_names=[tool.name for tool in self._tools],
                input_message_count=self._input_message_count,
                last_assistant_message_id=self._last_assistant_message_id,
            ),
        )

    def consume_replayed_outbound(self, message: ai.Message) -> bool:
        """Return True when a message belongs to the replayed prefix."""
        if not self._replayed_outbound_messages:
            return False
        key = serialize_message_key(message)
        if self._replayed_outbound_messages[0] != key:
            return False
        self._replayed_outbound_messages.popleft()
        return True

    def _recorded_model_count(self) -> int:
        base = len(self._replay.model_steps) if self._replay is not None else 0
        return base + len(self._new_model_steps)

    def _recorded_tool_count(self) -> int:
        base = len(self._replay.tool_results) if self._replay is not None else 0
        return base + len(self._new_tool_results)

    def _next_replayed_model_step(self) -> RecordedModelStep | None:
        if self._replay is None or self._replay_model_index >= len(
            self._replay.model_steps
        ):
            return None
        step = self._replay.model_steps[self._replay_model_index]
        self._replay_model_index += 1
        return step

    def _next_replayed_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
    ) -> RecordedToolResult | None:
        if self._replay is None or self._replay_tool_index >= len(
            self._replay.tool_results
        ):
            return None

        recorded = self._replay.tool_results[self._replay_tool_index]
        if recorded.tool_call_id != tool_call_id or recorded.tool_name != tool_name:
            raise ReplayMismatchError(
                "Replay tool mismatch: "
                f"expected {recorded.tool_name}/{recorded.tool_call_id}, "
                f"got {tool_name}/{tool_call_id}"
            )

        self._replay_tool_index += 1
        return recorded


def serialize_message_key(message: ai.Message) -> str:
    """Stable key used to match replayed outbound messages in-order."""
    return json.dumps(serialize_message(message), sort_keys=True, separators=(",", ":"))
