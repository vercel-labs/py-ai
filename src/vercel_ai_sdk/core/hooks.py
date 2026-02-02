from __future__ import annotations

import asyncio
import contextvars
import uuid
from typing import Any, ClassVar, Generic, TypeVar

import pydantic

from . import messages as messages_

T = TypeVar("T", bound=pydantic.BaseModel)

# Context var for pre-loaded hook resolutions (used in serverless patterns)
_hook_resolutions: contextvars.ContextVar[dict[str, dict[str, Any]]] = (
    contextvars.ContextVar("hook_resolutions", default={})
)


# TODO this should be properly typed
class HookPending(Exception):
    """
    Raised when a hook is places using .create_or_raise() and
    doesn't have a resolution in the context.
    """

    def __init__(
        self,
        hook_id: str,
        hook_type: str,
        metadata: dict[str, Any] | None = None,
    ):
        self.hook_id = hook_id
        self.hook_type = hook_type
        self.metadata = metadata or {}
        super().__init__(f"Hook pending: {hook_type}:{hook_id}")


def _make_hook_message(
    hook_id: str,
    hook_type: str,
    status: str,  # TODO type this with an enum or something
    metadata: dict[str, Any],
    resolution: dict[str, Any] | None = None,  # TODO should have the payload type
) -> messages_.Message:
    """Create a Message containing a HookPart."""
    return messages_.Message(
        role="assistant",
        parts=[
            messages_.HookPart(
                hook_id=hook_id,
                hook_type=hook_type,
                status=status,
                metadata=metadata,
                resolution=resolution,
            )
        ],
        is_done=True,
    )


class Hook(Generic[T]):
    """
    Mixin for hooks that adds hook-related classmethods
    to a Pydantic BaseModel that represents the hook's payload
    """

    _schema: ClassVar[type[pydantic.BaseModel]]
    _hook_type: ClassVar[str]

    def __init__(self, id: str, metadata: dict[str, Any] | None = None):
        self.id = id
        self.metadata = metadata or {}
        self._future: asyncio.Future[T] | None = None

    @classmethod
    async def create(cls, metadata: dict[str, Any] | None = None) -> T:
        """
        Create a hook and await its resolution.

        This emits a Message with HookPart(status="pending") to the stream, then
        blocks until resolve() is called. Upon resolution, emits another Message
        with HookPart(status="resolved").
        """

        # this is for a long-running type of application, where the hook
        # suspends execution until it is resolved from outside of the function

        from . import runtime as runtime_

        rt = runtime_._runtime.get(None)
        if rt is None:
            raise ValueError("No Runtime context - must be called within ai.run()")

        hook_id = uuid.uuid4().hex[:12]
        instance = cls(id=hook_id, metadata=metadata)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        instance._future = future

        # Register in runtime's pending hooks
        rt._pending_hooks[hook_id] = (future, instance)

        # Emit pending message through the stream
        await rt.put_message(
            _make_hook_message(
                hook_id=hook_id,
                hook_type=cls._hook_type,
                status="pending",
                metadata=metadata or {},
            )
        )

        # Block until resolved
        result = await future

        # Emit resolved message
        await rt.put_message(
            _make_hook_message(
                hook_id=hook_id,
                hook_type=cls._hook_type,
                status="resolved",
                metadata=metadata or {},
                resolution=result.model_dump() if hasattr(result, "model_dump") else {},
            )
        )

        # Clean up
        rt._pending_hooks.pop(hook_id, None)

        return result

    # TODO prohibit dict for a payload
    @classmethod
    def resolve(cls, hook_id: str, data: T | dict[str, Any]) -> None:
        """
        Resolve a pending hook by ID.

        Can be called from outside the graph (API handler, websocket, etc.)
        to unblock the awaiting coroutine.
        """
        from . import runtime as runtime_

        rt = runtime_._runtime.get(None)
        if rt is None:
            raise ValueError("No Runtime context")

        if hook_id not in rt._pending_hooks:
            raise ValueError(f"No pending hook with id: {hook_id}")

        future, _instance = rt._pending_hooks[hook_id]

        # Convert dict to model if needed
        resolved: T
        if isinstance(data, dict):
            resolved = cls._schema(**data)  # type: ignore[assignment]
        else:
            resolved = data

        future.set_result(resolved)

    @classmethod
    def cancel(cls, hook_id: str, reason: str | None = None) -> None:
        """
        Cancel a pending hook.
        """
        from . import runtime as runtime_

        rt = runtime_._runtime.get(None)
        if rt is None:
            raise ValueError("No Runtime context")

        if hook_id not in rt._pending_hooks:
            raise ValueError(f"No pending hook with id: {hook_id}")

        future, instance = rt._pending_hooks[hook_id]
        future.cancel(reason)

        # Emit cancelled message
        asyncio.create_task(
            rt.put_message(
                _make_hook_message(
                    hook_id=hook_id,
                    hook_type=cls._hook_type,
                    status="cancelled",
                    metadata=instance.metadata,
                )
            )
        )

        rt._pending_hooks.pop(hook_id, None)

    @classmethod
    def get_pending(cls, hook_id: str) -> Hook[T]:
        """
        Get a pending hook instance by ID.
        """
        from . import runtime as runtime_

        rt = runtime_._runtime.get(None)
        if rt is None:
            raise ValueError("No Runtime context")

        if hook_id not in rt._pending_hooks:
            raise ValueError(f"No pending hook with id: {hook_id}")

        return rt._pending_hooks[hook_id][1]

    @classmethod
    def create_or_raise(cls, hook_id: str, metadata: dict[str, Any] | None = None) -> T:
        """
        Get a resolved hook value or raise HookPending.
        """

        # this is for serverless applications where the resolution
        # is being provided in a different process, so instead of blocking
        # it needs to raise to exit the function

        resolutions = _hook_resolutions.get()

        if hook_id in resolutions:
            return cls._schema(**resolutions[hook_id])  # type: ignore[return-value]

        raise HookPending(
            hook_id=hook_id,
            hook_type=cls._hook_type,
            metadata=metadata or {},
        )


def hook(cls: type[T]) -> type[Hook[T]]:
    """
    Decorator to create a Hook type from a pydantic model.

    The pydantic model defines the schema for the hook's payload
    """
    # Create a new class that inherits from Hook
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
