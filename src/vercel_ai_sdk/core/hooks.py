from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar

import pydantic

from . import messages as messages_

if TYPE_CHECKING:
    from . import runtime as runtime_


# ---------------------------------------------------------------------------
# Module-level hook registries
#
# _live_hooks:
#   Populated by Hook.create() when a hook suspends inside a running graph.
#   Maps hook label -> (future, metadata dict, Runtime).
#   Consumed by Hook.resolve() / Hook.cancel() to unblock the awaiting
#   coroutine.  Entries are removed when the hook resolves, cancels, or
#   the run completes.
#
# _pending_resolutions:
#   Populated by Hook.resolve() when no live hook exists yet (serverless
#   re-entry: the user calls resolve() *before* ai.run() replays the graph).
#   Maps hook label -> validated resolution dict.
#   Consumed by Hook.create() at the start of graph execution — if a
#   pre-registered resolution exists for the label, the hook returns
#   immediately without suspending.  Entries are removed on consumption.
# ---------------------------------------------------------------------------

_live_hooks: dict[
    str, tuple[asyncio.Future[Any], dict[str, Any], runtime_.Runtime]
] = {}

_pending_resolutions: dict[str, dict[str, Any]] = {}
# label -> validated resolution dict


def _cleanup_run(labels: set[str]) -> None:
    """Remove all registry entries associated with a finished run."""
    for label in labels:
        _live_hooks.pop(label, None)
        _pending_resolutions.pop(label, None)


class Hook[T: pydantic.BaseModel]:
    """
    Hook: a suspension point that requires external input to continue.

    Usage in graph code:

        approval = await ToolApproval.create("approve_delete", metadata={...})
        if approval.granted:
            ...

    Resolution from outside the graph:

        ToolApproval.resolve("approve_delete", {"granted": True, ...})

    Behavior depends on the cancel_on_hooks flag passed to ai.run():

    cancel_on_hooks=False (default, long-running): the await blocks until
    Hook.resolve() is called from outside the graph (e.g., websocket
    handler, API endpoint).

    cancel_on_hooks=True (serverless): if no resolution is available, the
    hook's future is cancelled by run(). The branch receives CancelledError
    and dies cleanly. On re-entry, call Hook.resolve() before ai.run() to
    pre-register the resolution, then pass checkpoint= to replay.
    """

    _schema: ClassVar[type[pydantic.BaseModel]]
    _hook_type: ClassVar[str]

    @classmethod
    async def create(cls, label: str, metadata: dict[str, Any] | None = None) -> T:
        """
        Create a hook and await its resolution.

        The hook is submitted to the Runtime's step queue. run() will either:
        - Resolve immediately (if a resolution is available from checkpoint
          or pre-registered via Hook.resolve())
        - Cancel the future (cancel_on_hooks=True, serverless mode)
        - Hold the future (cancel_on_hooks=False, long-running mode)

        Args:
            label: Stable identifier for this hook. Used to match resolutions
                   across requests in serverless mode. Must be unique within
                   a single run.
            metadata: Optional metadata surfaced in the pending HookPart message.
        """
        from . import runtime as rt_mod

        rt = rt_mod._runtime.get(None)
        if rt is None:
            raise ValueError("No Runtime context - must be called within ai.run()")

        # Check pre-registered resolutions (serverless re-entry path)
        pre_registered = _pending_resolutions.pop(label, None)
        if pre_registered is not None:
            rt.record_hook(label, pre_registered)
            return cls._schema(**pre_registered)  # type: ignore[return-value]

        # Check checkpoint for a previously resolved value
        resolution = rt.get_hook_resolution(label)
        if resolution is not None:
            rt.record_hook(label, resolution)
            return cls._schema(**resolution)  # type: ignore[return-value]

        # Submit to step queue — run() decides what to do
        future: asyncio.Future[dict[str, Any]] = asyncio.Future()
        suspension = rt_mod.HookSuspension(
            label=label,
            hook_type=cls._hook_type,
            metadata=metadata or {},
            future=future,
        )
        await rt.put_hook_suspension(suspension)

        # Register in module-level registry for external resolution
        hook_metadata = metadata or {}
        _live_hooks[label] = (future, hook_metadata, rt)
        rt.track_hook_label(label)

        # Await resolution — may be resolved immediately by run(),
        # cancelled by run() (serverless), or resolved later by
        # Hook.resolve() (long-running).
        resolution = await future

        # Clean up
        _live_hooks.pop(label, None)

        # Record for checkpoint
        rt.record_hook(label, resolution)

        # Emit resolved message
        await rt.put_message(
            messages_.Message(
                role="assistant",
                parts=[
                    messages_.HookPart(
                        hook_id=label,
                        hook_type=cls._hook_type,
                        status="resolved",
                        metadata=hook_metadata,
                        resolution=resolution,
                    )
                ],
            )
        )

        return cls._schema(**resolution)  # type: ignore[return-value]

    @classmethod
    def resolve(cls, label: str, data: T | dict[str, Any]) -> None:
        """
        Resolve a hook by label.

        Works in two modes:

        1. Live hook exists (long-running): validates data, resolves the
           future immediately, unblocking the awaiting coroutine.

        2. No live hook yet (serverless re-entry): validates data and
           stashes it in the pre-registration registry. When ai.run()
           replays the graph and Hook.create() executes, it finds the
           pre-registered resolution and returns without suspending.

        Args:
            label: The hook label to resolve.
            data: Resolution payload (dict or pydantic model). Validated
                  against the hook's schema immediately.
        """
        # Validate and normalize to dict
        if isinstance(data, dict):
            # Validate by constructing the schema model
            validated = cls._schema(**data)
            resolution = validated.model_dump()
        else:
            # Already a model instance — validate it's the right type
            if not isinstance(data, cls._schema):
                raise TypeError(
                    f"Expected {cls._schema.__name__} or dict, "
                    f"got {type(data).__name__}"
                )
            resolution = data.model_dump()

        # Path 1: live hook — resolve the future directly
        if label in _live_hooks:
            future, _, _rt = _live_hooks[label]
            future.set_result(resolution)
            return

        # Path 2: no live hook — pre-register for later consumption
        _pending_resolutions[label] = resolution

    @classmethod
    async def cancel(cls, label: str, reason: str | None = None) -> None:
        """Cancel a pending hook.

        Only works for live hooks (long-running mode). Raises if the
        hook is not currently pending.
        """
        if label not in _live_hooks:
            raise ValueError(f"No pending hook with label: {label}")

        future, hook_metadata, rt = _live_hooks.pop(label)
        future.cancel(reason)

        await rt.put_message(
            messages_.Message(
                role="assistant",
                parts=[
                    messages_.HookPart(
                        hook_id=label,
                        hook_type=cls._hook_type,
                        status="cancelled",
                        metadata=hook_metadata,
                    )
                ],
            )
        )


def hook[T: pydantic.BaseModel](cls: type[T]) -> type[Hook[T]]:
    """
    Decorator to create a Hook type from a pydantic model.

    The pydantic model defines the schema for the hook's resolution payload.
    """
    hook_impl = type(
        cls.__name__,
        (Hook,),
        {
            "_schema": cls,
            "_hook_type": cls.__name__,
            "__doc__": cls.__doc__,
        },
    )

    return hook_impl
