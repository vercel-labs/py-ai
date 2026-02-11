from __future__ import annotations

import asyncio
from typing import Any, ClassVar, Generic, Literal, TypeVar

import pydantic

from . import messages as messages_

T = TypeVar("T", bound=pydantic.BaseModel)


class Hook(Generic[T]):
    """
    Hook: a suspension point that requires external input to continue.

    Usage in graph code (identical in all modes):

        approval = await ToolApproval.create("approve_delete", metadata={...})
        if approval.granted:
            ...

    Behavior depends on the cancel_on_hooks flag passed to ai.run():

    cancel_on_hooks=False (default, long-running): the await blocks until
    Hook.resolve() is called from outside the graph (e.g., websocket
    handler, API endpoint).

    cancel_on_hooks=True (serverless): if no resolution is available, the
    hook's future is cancelled by run(). The branch receives CancelledError
    and dies cleanly. On re-entry, pass checkpoint= and resolutions= to
    run() to replay completed work and resolve the hooks.
    """

    _schema: ClassVar[type[pydantic.BaseModel]]
    _hook_type: ClassVar[str]

    def __init__(self, id: str, metadata: dict[str, Any] | None = None):
        self.id = id
        self.metadata = metadata or {}

    def to_message(
        self,
        status: Literal["pending", "resolved", "cancelled"],
        resolution: dict[str, Any] | None = None,
    ) -> messages_.Message:
        return messages_.Message(
            role="assistant",
            parts=[
                messages_.HookPart(
                    hook_id=self.id,
                    hook_type=self._hook_type,
                    status=status,
                    metadata=self.metadata,
                    resolution=resolution,
                )
            ],
        )

    @classmethod
    async def create(cls, label: str, metadata: dict[str, Any] | None = None) -> T:
        """
        Create a hook and await its resolution.

        The hook is submitted to the Runtime's step queue. run() will either:
        - Resolve immediately (if a resolution is available from checkpoint/resolutions)
        - Cancel the future (cancel_on_hooks=True, serverless mode)
        - Hold the future (cancel_on_hooks=False, long-running mode)

        Args:
            label: Stable identifier for this hook. Used to match resolutions
                   across requests in serverless mode. Must be unique within
                   a single run.
            metadata: Optional metadata surfaced in the pending HookPart message.
        """
        from . import runtime as runtime_

        rt = runtime_._runtime.get(None)
        if rt is None:
            raise ValueError("No Runtime context - must be called within ai.run()")

        # Check checkpoint for a previously resolved value
        resolution = rt.get_hook_resolution(label)
        if resolution is not None:
            rt.record_hook(label, resolution)
            return cls._schema(**resolution)  # type: ignore[return-value]

        # Submit to step queue — run() decides what to do
        future: asyncio.Future[dict[str, Any]] = asyncio.Future()
        suspension = runtime_.HookSuspension(
            label=label,
            hook_type=cls._hook_type,
            metadata=metadata or {},
            future=future,
        )
        await rt.put_hook_suspension(suspension)

        # Also register for long-running external resolution
        instance = cls(id=label, metadata=metadata)
        await rt.put_hook(instance, future)

        # Await resolution — may be resolved immediately by run(),
        # cancelled by run() (serverless), or resolved later by
        # Hook.resolve() (long-running).
        resolution = await future

        # Clean up long-running registration
        rt._hook_futures.pop(label, None)

        # Record for checkpoint
        rt.record_hook(label, resolution)

        # Emit resolved message
        await rt.put_message(
            instance.to_message(
                status="resolved",
                resolution=resolution,
            )
        )

        return cls._schema(**resolution)  # type: ignore[return-value]

    @classmethod
    def resolve(cls, label: str, data: T | dict[str, Any]) -> None:
        """
        Resolve a pending hook by label.

        Can be called from outside the graph (API handler, websocket, etc.)
        to unblock the awaiting coroutine in long-running mode.
        """
        from . import runtime as runtime_

        rt = runtime_._runtime.get(None)
        if rt is None:
            raise ValueError("No Runtime context")

        if label not in rt._hook_futures:
            raise ValueError(f"No pending hook with label: {label}")

        future, _instance = rt._hook_futures[label]

        # Convert model to dict if needed
        if isinstance(data, dict):
            resolution = data
        else:
            resolution = data.model_dump()

        future.set_result(resolution)

    @classmethod
    def cancel(cls, label: str, reason: str | None = None) -> None:
        """Cancel a pending hook."""
        from . import runtime as runtime_

        rt = runtime_._runtime.get(None)
        if rt is None:
            raise ValueError("No Runtime context")

        if label not in rt._hook_futures:
            raise ValueError(f"No pending hook with label: {label}")

        future, instance = rt._hook_futures[label]
        future.cancel(reason)

        asyncio.create_task(rt.put_message(instance.to_message(status="cancelled")))
        rt._hook_futures.pop(label, None)


def hook(cls: type[T]) -> type[Hook[T]]:
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

    return hook_impl  # type: ignore[return-value]
