from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from typing import Any, get_type_hints

import pydantic

from ..telemetry import events as telemetry_
from ..types import messages as messages_
from . import checkpoint as checkpoint_
from . import context as context_
from . import hooks as hooks_
from . import mcp
from . import streams as streams_
from . import tools as tools_

logger = logging.getLogger(__name__)


# ── EventLog ──────────────────────────────────────────────────────
#
# Pure bookkeeping: replay from checkpoint + record new events.
# No asyncio, no queues — just data in, data out.
#


class EventLog:
    """Replay/record layer backed by a Checkpoint.

    Holds replay cursors (read pointer into the checkpoint) and
    recording lists (new events produced during this run).
    Completely synchronous — no queues, no async.
    """

    def __init__(self, checkpoint: checkpoint_.Checkpoint | None = None) -> None:
        self._checkpoint = checkpoint or checkpoint_.Checkpoint()

        # Replay cursors
        self._step_index: int = 0
        self._tool_replay: dict[str, checkpoint_.ToolEvent] = {
            t.tool_call_id: t for t in self._checkpoint.tools
        }
        self._hook_replay: dict[str, dict[str, Any]] = {
            h.label: h.resolution for h in self._checkpoint.hooks
        }

        # Recording lists (new events from this run)
        self._step_log: list[checkpoint_.StepEvent] = []
        self._tool_log: list[checkpoint_.ToolEvent] = []
        self._hook_log: list[checkpoint_.HookEvent] = []

    # ── Steps ─────────────────────────────────────────────────

    @property
    def step_index(self) -> int:
        return self._step_index

    def try_replay_step(self) -> streams_.StreamResult | None:
        if self._step_index < len(self._checkpoint.steps):
            event = self._checkpoint.steps[self._step_index]
            self._step_index += 1
            logger.info("Replaying step %d from checkpoint", event.index)
            return event.to_stream_result()
        return None

    def record_step(self, result: streams_.StreamResult) -> None:
        event = checkpoint_.StepEvent(
            index=self._step_index,
            messages=list(result.messages),
        )
        self._step_log.append(event)
        self._step_index += 1

    # ── Tools ─────────────────────────────────────────────────

    def try_replay_tool(self, tool_call_id: str) -> checkpoint_.ToolEvent | None:
        event = self._tool_replay.get(tool_call_id)
        if event is not None:
            logger.info(
                "Replaying tool %s (call_id=%s) from checkpoint",
                event.tool_call_id,
                tool_call_id,
            )
        return event

    def record_tool(
        self, tool_call_id: str, result: Any, *, status: str = "result"
    ) -> None:
        self._tool_log.append(
            checkpoint_.ToolEvent(
                tool_call_id=tool_call_id, result=result, status=status
            )
        )

    # ── Hooks ─────────────────────────────────────────────────

    def get_hook_resolution(self, label: str) -> dict[str, Any] | None:
        resolution = self._hook_replay.get(label)
        if resolution is not None:
            logger.info("Resolving hook '%s' from checkpoint", label)
        return resolution

    def record_hook(self, label: str, resolution: dict[str, Any]) -> None:
        self._hook_log.append(checkpoint_.HookEvent(label=label, resolution=resolution))

    # ── Snapshot ──────────────────────────────────────────────

    def checkpoint(
        self, pending_hooks: list[checkpoint_.PendingHookInfo] | None = None
    ) -> checkpoint_.Checkpoint:
        """Build a full Checkpoint merging prior state + new recordings."""
        return checkpoint_.Checkpoint(
            steps=list(self._checkpoint.steps) + self._step_log,
            tools=list(self._checkpoint.tools) + self._tool_log,
            hooks=list(self._checkpoint.hooks) + self._hook_log,
            pending_hooks=pending_hooks or [],
        )


# ── LoopExecutor ─────────────────────────────────────────────────
#
# Async coordination: queues that let graph code (streams, hooks,
# tools) talk to the driver loop. Pure mailbox — no replay, no
# checkpoint awareness.
#


@dataclasses.dataclass
class HookSuspension:
    """Submitted to the step queue when a hook needs resolution."""

    label: str
    hook_type: str
    metadata: dict[str, Any]
    future: asyncio.Future[Any]
    cancels_future: bool = False


class LoopExecutor:
    """Async coordination layer between graph code and the driver loop.

    Graph code (``@stream`` decorators, hooks, tool execution) submits
    work via the producer methods.  The driver loop consumes via
    ``next()`` and ``drain_messages()``.
    """

    class _Sentinel:
        pass

    _SENTINEL = _Sentinel()

    def __init__(self) -> None:
        self._step_queue: asyncio.Queue[
            tuple[streams_.Stream, asyncio.Future[streams_.StreamResult]]
            | HookSuspension
            | LoopExecutor._Sentinel
        ] = asyncio.Queue()

        self._message_queue: asyncio.Queue[messages_.Message] = asyncio.Queue()

        # Pending hooks (unresolved during this run)
        self._pending_hooks: dict[str, HookSuspension] = {}

        # Track hook labels registered in this run for cleanup
        self._hook_labels: set[str] = set()

    # ── Producers (called by graph code) ──────────────────────

    async def put_step(
        self, step_fn: streams_.Stream, future: asyncio.Future[streams_.StreamResult]
    ) -> None:
        await self._step_queue.put((step_fn, future))

    async def put_hook(self, suspension: HookSuspension) -> None:
        await self._step_queue.put(suspension)

    async def put_message(self, message: messages_.Message) -> None:
        await self._message_queue.put(message)

    async def done(self) -> None:
        await self._step_queue.put(self._SENTINEL)

    # ── Consumer (called by driver loop) ──────────────────────

    async def next(
        self, timeout: float = 0.1
    ) -> (
        tuple[streams_.Stream, asyncio.Future[streams_.StreamResult]]
        | HookSuspension
        | None
    ):
        """Pull the next item from the step queue.

        Returns ``None`` on timeout (no item available).
        Returns the sentinel's semantic equivalent by raising StopIteration
        when the graph signals completion.
        """
        try:
            item = await asyncio.wait_for(self._step_queue.get(), timeout=timeout)
        except TimeoutError:
            return None

        if isinstance(item, LoopExecutor._Sentinel):
            raise _LoopDone
        return item

    def drain_messages(self) -> list[messages_.Message]:
        msgs: list[messages_.Message] = []
        while not self._message_queue.empty():
            try:
                msgs.append(self._message_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return msgs

    # ── Hook label tracking ───────────────────────────────────

    def track_hook_label(self, label: str) -> None:
        self._hook_labels.add(label)

    def pending_hook_infos(self) -> list[checkpoint_.PendingHookInfo]:
        return [
            checkpoint_.PendingHookInfo(
                label=sus.label,
                hook_type=sus.hook_type,
                metadata=sus.metadata,
            )
            for sus in self._pending_hooks.values()
        ]


class _LoopDone(Exception):
    """Internal signal: the loop function has finished."""


# ── Runtime ───────────────────────────────────────────────────────
#
# Thin composition of EventLog + LoopExecutor.
# The context var points here; graph code accesses rt.log and
# rt.executor directly.
#


class Runtime:
    """Central coordinator — composes EventLog and LoopExecutor.

    Graph code accesses ``rt.log`` for replay/record and
    ``rt.executor`` for async coordination.
    """

    def __init__(self, checkpoint: checkpoint_.Checkpoint | None = None) -> None:
        self.log = EventLog(checkpoint)
        self.executor = LoopExecutor()

    def checkpoint(self) -> checkpoint_.Checkpoint:
        return self.log.checkpoint(
            pending_hooks=self.executor.pending_hook_infos(),
        )


_runtime: contextvars.ContextVar[Runtime] = contextvars.ContextVar("runtime")


def get_checkpoint() -> checkpoint_.Checkpoint:
    """Get the current checkpoint from the active Runtime."""
    return _runtime.get().checkpoint()


def _find_runtime_param(fn: Callable[..., Any]) -> str | None:
    """Find a parameter typed as Runtime, return its name or None."""
    try:
        hints = get_type_hints(fn)
    except Exception:
        return None
    for name, hint in hints.items():
        if hint is Runtime:
            return name
    return None


async def execute_tool(
    tool_call: messages_.ToolPart,
    message: messages_.Message | None = None,
) -> Any:
    """Execute a single tool call with replay support.

    Looks up the tool by name — first from the active Context (if any),
    then from the global registry.  Executes it and updates the ToolPart
    (and parent Message) with the result.  Emits the updated message to
    the LoopExecutor queue so the UI sees the transition from
    status="pending" to status="result" (or "error").

    If a checkpoint exists with a cached result for this tool_call_id,
    returns the cached result without re-executing.
    """
    rt = _runtime.get(None)

    # Replay: return cached result if available
    if rt:
        cached = rt.log.try_replay_tool(tool_call.tool_call_id)
        if cached is not None:
            if cached.status == "error":
                tool_call.set_error(cached.result)
            else:
                tool_call.set_result(cached.result)
            return cached.result

    telemetry_.handle(
        telemetry_.ToolCallStartEvent(
            tool_name=tool_call.tool_name,
            tool_call_id=tool_call.tool_call_id,
            args=tool_call.tool_args,
        )
    )
    t0 = telemetry_.time_ms()

    # Fresh execution — resolve from Context first, then global registry
    tool: tools_.Tool[..., Any] | None = None
    ctx = context_._context.get(None)
    if ctx is not None:
        tool = ctx.get_tool(tool_call.tool_name)
    if tool is None:
        tool = tools_.get_tool(tool_call.tool_name)
    if tool is None:
        raise ValueError(f"Tool not found in registry: {tool_call.tool_name}")

    error_str: str | None = None
    try:
        result = await tool.validate_and_call(tool_call.tool_args, rt)
        tool_call.set_result(result)
    except (json.JSONDecodeError, pydantic.ValidationError) as exc:
        result = f"{type(exc).__name__}: {exc}"
        error_str = result
        tool_call.set_error(result)

    telemetry_.handle(
        telemetry_.ToolCallFinishEvent(
            tool_name=tool_call.tool_name,
            tool_call_id=tool_call.tool_call_id,
            result=result,
            error=error_str,
            duration_ms=telemetry_.time_ms() - t0,
        )
    )

    # Record for checkpoint
    if rt:
        rt.log.record_tool(tool_call.tool_call_id, result, status=tool_call.status)

    # Emit updated message so UI sees status change
    if rt and message:
        await rt.executor.put_message(message.model_copy(deep=True))

    return result


# ── RunResult ─────────────────────────────────────────────────────


@dataclasses.dataclass
class HookInfo:
    """Info about a pending (unresolved) hook, exposed on RunResult."""

    label: str
    hook_type: str
    metadata: dict[str, Any]


class RunResult:
    """Returned by run(). Async-iterate for messages, then check state.

    Usage:
        result = ai.run(my_graph, llm, query)
        async for msg in result:
            ...
        result.checkpoint    # Checkpoint with all completed work
        result.pending_hooks # dict of unresolved hooks (empty if graph completed)
    """

    def __init__(self) -> None:
        self._messages: AsyncGenerator[messages_.Message] | None = None
        self._runtime: Runtime | None = None

    @property
    def checkpoint(self) -> checkpoint_.Checkpoint:
        if self._runtime is None:
            return checkpoint_.Checkpoint()
        return self._runtime.checkpoint()

    @property
    def pending_hooks(self) -> dict[str, HookInfo]:
        if self._runtime is None:
            return {}
        return {
            label: HookInfo(
                label=sus.label,
                hook_type=sus.hook_type,
                metadata=sus.metadata,
            )
            for label, sus in self._runtime.executor._pending_hooks.items()
        }

    async def __aiter__(self) -> AsyncGenerator[messages_.Message]:
        if self._messages is not None:
            async for msg in self._messages:
                yield msg


# ── run() ─────────────────────────────────────────────────────────


async def _stop_when_done(executor: LoopExecutor, task: Awaitable[None]) -> None:
    try:
        await task
    finally:
        await executor.done()


def run(
    root: Callable[..., Coroutine[Any, Any, Any]],
    *args: Any,
    checkpoint: checkpoint_.Checkpoint | None = None,
    context: context_.Context | None = None,
) -> RunResult:
    """Main entry point.

    1. Starts the root function as a background task
    2. Pulls steps and hook suspensions from the LoopExecutor queue
    3. Executes each step, yielding messages
    4. Resolves or suspends hooks depending on the hook's cancels_future
    5. Returns RunResult with .checkpoint and .pending_hooks

    Args:
        root: The loop function to execute.
        *args: Positional arguments forwarded to ``root``.
        checkpoint: Checkpoint to resume from.
        context: LLM prompt context (tools, system prompt, messages).
            If ``None``, an empty Context is created automatically.
    """
    result = RunResult()

    # Discard stale checkpoints: if the checkpoint has pending hooks but
    # none of them have been resolved, this isn't a resume.
    effective_checkpoint = checkpoint
    if checkpoint and checkpoint.pending_hooks:
        pending_labels = [ph.label for ph in checkpoint.pending_hooks]
        has_resolution = any(
            label in hooks_._pending_resolutions for label in pending_labels
        )
        if not has_resolution:
            logger.info(
                "Discarding stale checkpoint: pending hooks %s have no "
                "matching resolutions",
                pending_labels,
            )
            effective_checkpoint = None
        else:
            logger.info(
                "Resuming from checkpoint with %d pending hook(s): %s",
                len(pending_labels),
                pending_labels,
            )

    async def _generate() -> AsyncGenerator[messages_.Message]:
        rt = Runtime(checkpoint=effective_checkpoint)
        result._runtime = rt
        token_runtime = _runtime.set(rt)

        ctx = context or context_.Context()
        token_context = context_._context.set(ctx)

        token_run_id = telemetry_.start_run()

        telemetry_.handle(telemetry_.RunStartEvent())

        mcp_pool: dict[str, mcp.client._Connection] = {}
        mcp_token = mcp.client._pool.set(mcp_pool)

        kwargs: dict[str, Any] = {}
        if runtime_param := _find_runtime_param(root):
            kwargs[runtime_param] = rt

        run_error: BaseException | None = None
        total_usage: messages_.Usage | None = None

        try:
            async with asyncio.TaskGroup() as tg:
                _task: asyncio.Task[None] = tg.create_task(
                    _stop_when_done(rt.executor, root(*args, **kwargs))
                )

                while True:
                    # Drain pending messages
                    for msg in rt.executor.drain_messages():
                        yield msg

                    # Pull next item from the graph executor
                    try:
                        item = await rt.executor.next()
                    except _LoopDone:
                        for msg in rt.executor.drain_messages():
                            yield msg
                        break

                    if item is None:
                        # Timeout — no item available, loop again
                        continue

                    # ── Hook suspension ────────────────────────
                    if isinstance(item, HookSuspension):
                        resolution = rt.log.get_hook_resolution(item.label)
                        if resolution is not None:
                            item.future.set_result(resolution)
                            rt.log.record_hook(item.label, resolution)
                        else:
                            rt.executor._pending_hooks[item.label] = item
                            if item.cancels_future:
                                item.future.cancel()

                            yield messages_.Message(
                                role="assistant",
                                parts=[
                                    messages_.HookPart(
                                        hook_id=item.label,
                                        hook_type=item.hook_type,
                                        status="pending",
                                        metadata=item.metadata,
                                    )
                                ],
                            )

                        await asyncio.sleep(0)
                        for msg in rt.executor.drain_messages():
                            yield msg
                        continue

                    # ── Regular step ───────────────────────────
                    step_fn, future = item

                    telemetry_.handle(
                        telemetry_.StepStartEvent(
                            step_index=rt.log.step_index,
                        )
                    )

                    for tool_msg in rt.executor.drain_messages():
                        yield tool_msg

                    result_messages: list[messages_.Message] = []

                    async for msg in step_fn():
                        msg_copy = msg.model_copy(deep=True)
                        yield msg_copy
                        result_messages.append(msg)

                        for tool_msg in rt.executor.drain_messages():
                            yield tool_msg

                    step_result = streams_.StreamResult(messages=result_messages)
                    future.set_result(step_result)

                    telemetry_.handle(
                        telemetry_.StepFinishEvent(
                            step_index=rt.log.step_index,
                            result=step_result,  # type: ignore[arg-type]
                        )
                    )

                    # Accumulate usage for run-level telemetry
                    step_usage = step_result.usage
                    if step_usage is not None:
                        total_usage = (
                            step_usage
                            if total_usage is None
                            else total_usage + step_usage
                        )

                    await asyncio.sleep(0)
                    for tool_msg in rt.executor.drain_messages():
                        yield tool_msg

        except BaseException as exc:
            run_error = exc
            raise

        finally:
            telemetry_.handle(
                telemetry_.RunFinishEvent(
                    usage=total_usage,
                    error=run_error,
                )
            )
            telemetry_.end_run(token_run_id)

            hooks_._cleanup_run(rt.executor._hook_labels)

            if mcp_token is not None:
                await mcp.client.close_connections()
                mcp.client._pool.reset(mcp_token)

            context_._context.reset(token_context)
            _runtime.reset(token_runtime)

    result._messages = _generate()
    return result
