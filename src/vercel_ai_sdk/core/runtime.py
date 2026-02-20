from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Sequence
from typing import Any, get_type_hints

import pydantic

from .. import mcp
from . import checkpoint as checkpoint_
from . import hooks as hooks_
from . import llm as llm_
from . import messages as messages_
from . import streams as streams_
from . import tools as tools_

# ── Queue item types ──────────────────────────────────────────────


@dataclasses.dataclass
class HookSuspension:
    """Submitted to the step queue when a hook needs resolution."""

    label: str
    hook_type: str
    metadata: dict[str, Any]
    future: asyncio.Future[Any]


# ── Runtime ───────────────────────────────────────────────────────


class Runtime:
    """
    Central coordinator for the agent loop.

    Functions decorated with @stream submit step functions to the queue.
    Hooks submit HookSuspension items to the same queue.
    run() pulls items, processes them, yields messages, and resolves futures.
    """

    class _Sentinel:
        pass

    _SENTINEL = _Sentinel()

    def __init__(
        self,
        checkpoint: checkpoint_.Checkpoint | None = None,
    ) -> None:
        self._step_queue: asyncio.Queue[
            tuple[streams_.Stream, asyncio.Future[streams_.StreamResult]]
            | HookSuspension
            | Runtime._Sentinel
        ] = asyncio.Queue()

        # Message queue for streaming tool results and hook messages
        self._message_queue: asyncio.Queue[messages_.Message] = asyncio.Queue()

        # Checkpoint: replay state from previous run
        self._checkpoint = checkpoint or checkpoint_.Checkpoint()

        # Replay cursors
        self._step_index: int = 0
        self._tool_replay: dict[str, checkpoint_.ToolEvent] = {
            t.tool_call_id: t for t in self._checkpoint.tools
        }
        self._hook_replay: dict[str, dict[str, Any]] = {
            h.label: h.resolution for h in self._checkpoint.hooks
        }

        # Recording: new events from this run
        self._step_log: list[checkpoint_.StepEvent] = []
        self._tool_log: list[checkpoint_.ToolEvent] = []
        self._hook_log: list[checkpoint_.HookEvent] = []

        # Pending hooks (unresolved during this run)
        self._pending_hooks: dict[str, HookSuspension] = {}

        # Track hook labels registered in this run for cleanup
        self._hook_labels: set[str] = set()

    # ── Step queue ────────────────────────────────────────────────

    async def put_step(
        self, step_fn: streams_.Stream, future: asyncio.Future[streams_.StreamResult]
    ) -> None:
        await self._step_queue.put((step_fn, future))

    async def put_hook_suspension(self, suspension: HookSuspension) -> None:
        await self._step_queue.put(suspension)

    async def signal_done(self) -> None:
        await self._step_queue.put(self._SENTINEL)

    # ── Message queue ─────────────────────────────────────────────

    async def put_message(self, message: messages_.Message) -> None:
        await self._message_queue.put(message)

    def get_all_messages(self) -> list[messages_.Message]:
        msgs = []
        while not self._message_queue.empty():
            try:
                msgs.append(self._message_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return msgs

    # ── Replay / record: steps ────────────────────────────────────

    def try_replay_step(self) -> streams_.StreamResult | None:
        if self._step_index < len(self._checkpoint.steps):
            event = self._checkpoint.steps[self._step_index]
            self._step_index += 1
            return event.to_stream_result()
        return None

    def record_step(self, result: streams_.StreamResult) -> None:
        event = checkpoint_.StepEvent(
            index=self._step_index,
            messages=list(result.messages),
        )
        self._step_log.append(event)
        self._step_index += 1

    # ── Replay / record: tools ────────────────────────────────────

    def try_replay_tool(self, tool_call_id: str) -> checkpoint_.ToolEvent | None:
        """Return the cached ToolEvent if available, else None."""
        return self._tool_replay.get(tool_call_id)

    def record_tool(
        self, tool_call_id: str, result: Any, *, status: str = "result"
    ) -> None:
        self._tool_log.append(
            checkpoint_.ToolEvent(
                tool_call_id=tool_call_id, result=result, status=status
            )
        )

    # ── Replay / record: hooks ────────────────────────────────────

    def get_hook_resolution(self, label: str) -> dict[str, Any] | None:
        return self._hook_replay.get(label)

    def record_hook(self, label: str, resolution: dict[str, Any]) -> None:
        self._hook_log.append(checkpoint_.HookEvent(label=label, resolution=resolution))

    def track_hook_label(self, label: str) -> None:
        """Track a hook label for cleanup when the run completes."""
        self._hook_labels.add(label)

    # ── Checkpoint ────────────────────────────────────────────────

    def get_checkpoint(self) -> checkpoint_.Checkpoint:
        return checkpoint_.Checkpoint(
            steps=list(self._checkpoint.steps) + self._step_log,
            tools=list(self._checkpoint.tools) + self._tool_log,
            hooks=list(self._checkpoint.hooks) + self._hook_log,
        )


_runtime: contextvars.ContextVar[Runtime] = contextvars.ContextVar("runtime")


def get_checkpoint() -> checkpoint_.Checkpoint:
    """Get the current checkpoint from the active Runtime."""
    return _runtime.get().get_checkpoint()


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


# ── Convenience functions ─────────────────────────────────────────


@streams_.stream
async def stream_step(
    llm: llm_.LanguageModel,
    messages: list[messages_.Message],
    tools: Sequence[tools_.ToolLike] | None = None,
    label: str | None = None,
) -> AsyncGenerator[messages_.Message]:
    """Single LLM call that streams to Runtime."""
    async for msg in llm.stream(messages=messages, tools=tools):
        msg.label = label
        yield msg


async def execute_tool(
    tool_call: messages_.ToolPart,
    message: messages_.Message | None = None,
) -> Any:
    """
    Execute a single tool call with replay support.

    Looks up the tool by name from the global registry, executes it,
    and updates the ToolPart (and parent Message) with the result.
    Emits the updated message to the Runtime queue so the UI sees
    the transition from status="pending" to status="result" (or "error").

    If a checkpoint exists with a cached result for this tool_call_id,
    returns the cached result without re-executing.
    """
    rt = _runtime.get(None)

    # Replay: return cached result if available
    if rt:
        cached = rt.try_replay_tool(tool_call.tool_call_id)
        if cached is not None:
            if cached.status == "error":
                tool_call.set_error(cached.result)
            else:
                tool_call.set_result(cached.result)
            return cached.result

    # Fresh execution
    tool = tools_.get_tool(tool_call.tool_name)
    if tool is None:
        raise ValueError(f"Tool not found in registry: {tool_call.tool_name}")

    try:
        result = await tool.validate_and_call(tool_call.tool_args, rt)
        tool_call.set_result(result)
    except (json.JSONDecodeError, pydantic.ValidationError) as exc:
        # LLM produced malformed JSON or args that don't match the schema.
        # Report back as a tool error so the model can retry.
        result = f"{type(exc).__name__}: {exc}"
        tool_call.set_error(result)

    # Record for checkpoint
    if rt:
        rt.record_tool(tool_call.tool_call_id, result, status=tool_call.status)

    # Emit updated message so UI sees status change
    if rt and message:
        await rt.put_message(message.model_copy(deep=True))

    return result


async def stream_loop(
    llm: llm_.LanguageModel,
    messages: list[messages_.Message],
    tools: Sequence[tools_.ToolLike],
    label: str | None = None,
) -> streams_.StreamResult:
    """Agent loop: stream LLM, execute tools, repeat until done."""
    local_messages = list(messages)

    while True:
        result = await stream_step(llm, local_messages, tools, label=label)

        if not result.tool_calls:
            return result

        last_msg = result.last_message
        if last_msg is not None:
            local_messages.append(last_msg)

        await asyncio.gather(
            *(execute_tool(tc, message=last_msg) for tc in result.tool_calls)
        )


# ── RunResult ─────────────────────────────────────────────────────


@dataclasses.dataclass
class HookInfo:
    """Info about a pending (unresolved) hook, exposed on RunResult."""

    label: str
    hook_type: str
    metadata: dict[str, Any]


class RunResult:
    """
    Returned by run(). Async-iterate for messages, then check state.

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
        return self._runtime.get_checkpoint()

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
            for label, sus in self._runtime._pending_hooks.items()
        }

    async def __aiter__(self) -> AsyncGenerator[messages_.Message]:
        if self._messages is not None:
            async for msg in self._messages:
                yield msg


# ── run() ─────────────────────────────────────────────────────────


async def _stop_when_done(runtime: Runtime, task: Awaitable[None]) -> None:
    try:
        await task
    finally:
        await runtime.signal_done()


def run(
    root: Callable[..., Coroutine[Any, Any, Any]],
    *args: Any,
    checkpoint: checkpoint_.Checkpoint | None = None,
    cancel_on_hooks: bool = False,
) -> RunResult:
    """
    Main entry point.

    1. Starts the root function as a background task
    2. Pulls steps and hook suspensions from the Runtime queue
    3. Executes each step, yielding messages
    4. Resolves or suspends hooks depending on mode:
       - cancel_on_hooks=True  (serverless): cancel the future, branch dies,
         caller inspects result.pending_hooks and result.checkpoint to resume
       - cancel_on_hooks=False (long-running, default): future stays alive,
         external code calls Hook.resolve() / Hook.cancel() to unblock
    5. Returns RunResult with .checkpoint and .pending_hooks
    """
    result = RunResult()

    async def _generate() -> AsyncGenerator[messages_.Message]:
        runtime = Runtime(checkpoint=checkpoint)
        result._runtime = runtime
        token_runtime = _runtime.set(runtime)

        mcp_pool: dict[str, mcp.client._Connection] = {}
        mcp_token = mcp.client._pool.set(mcp_pool)

        kwargs: dict[str, Any] = {}
        if runtime_param := _find_runtime_param(root):
            kwargs[runtime_param] = runtime

        try:
            async with asyncio.TaskGroup() as tg:
                _task: asyncio.Task[None] = tg.create_task(
                    _stop_when_done(runtime, root(*args, **kwargs))
                )

                while True:
                    # Drain pending messages
                    for msg in runtime.get_all_messages():
                        yield msg

                    # Wait for next queue item
                    try:
                        step_item = await asyncio.wait_for(
                            runtime._step_queue.get(), timeout=0.1
                        )
                    except TimeoutError:
                        continue

                    if isinstance(step_item, Runtime._Sentinel):
                        for msg in runtime.get_all_messages():
                            yield msg
                        break

                    # ── Hook suspension ────────────────────────
                    if isinstance(step_item, HookSuspension):
                        resolution = runtime.get_hook_resolution(step_item.label)
                        if resolution is not None:
                            # Resolve immediately
                            step_item.future.set_result(resolution)
                            runtime.record_hook(step_item.label, resolution)
                        else:
                            # No resolution available
                            runtime._pending_hooks[step_item.label] = step_item
                            if cancel_on_hooks:
                                # Serverless: cancel the future so the branch
                                # dies with CancelledError. Caller inspects
                                # result.pending_hooks to resume later.
                                step_item.future.cancel()
                            # else: long-running — future stays alive,
                            # external code calls Hook.resolve() to unblock.

                            # Yield pending hook message
                            yield messages_.Message(
                                role="assistant",
                                parts=[
                                    messages_.HookPart(
                                        hook_id=step_item.label,
                                        hook_type=step_item.hook_type,
                                        status="pending",
                                        metadata=step_item.metadata,
                                    )
                                ],
                            )

                        # Let resolved branches resume and submit their
                        # next steps before we pull from the queue again.
                        await asyncio.sleep(0)

                        # Drain messages after hook processing
                        for msg in runtime.get_all_messages():
                            yield msg
                        continue

                    # ── Regular step ───────────────────────────
                    step_fn, future = step_item

                    for tool_msg in runtime.get_all_messages():
                        yield tool_msg

                    result_messages: list[messages_.Message] = []

                    async for msg in step_fn():
                        msg_copy = msg.model_copy(deep=True)
                        yield msg_copy
                        result_messages.append(msg)

                        for tool_msg in runtime.get_all_messages():
                            yield tool_msg

                    step_result = streams_.StreamResult(messages=result_messages)
                    future.set_result(step_result)

                    await asyncio.sleep(0)
                    for tool_msg in runtime.get_all_messages():
                        yield tool_msg

        finally:
            # Clean up module-level hook registries for this run
            hooks_._cleanup_run(runtime._hook_labels)

            if mcp_token is not None:
                await mcp.client.close_connections()
                mcp.client._pool.reset(mcp_token)

            _runtime.reset(token_runtime)

    result._messages = _generate()
    return result
