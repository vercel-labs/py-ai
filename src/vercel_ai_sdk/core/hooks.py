"""
Hook system for human-in-the-loop agent flows.

Hooks are first-class suspension points in the execution graph. When you await
a hook, it blocks until resolve() is called from outside (e.g., an API endpoint,
websocket handler, or another coroutine).

Example:
    @ai.hook
    class CommunicationApproval(pydantic.BaseModel):
        granted: bool
        reason: str

    # In your graph:
    approval = await CommunicationApproval.create(
        metadata={"tool_call_id": tc.tool_call_id}
    )
    if approval.granted:
        await ai.execute_tool(tc, tools, message)
    else:
        # handle rejection

    # From outside (API handler, websocket, etc.):
    CommunicationApproval.resolve(hook_id, {"granted": True, "reason": "Approved"})
"""

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


def get_hook_resolutions() -> dict[str, dict[str, Any]]:
    """Get the current hook resolutions from context."""
    return _hook_resolutions.get()


def set_hook_resolutions(
    resolutions: dict[str, dict[str, Any]],
) -> contextvars.Token[dict[str, dict[str, Any]]]:
    """Set hook resolutions in context. Returns token for reset."""
    return _hook_resolutions.set(resolutions)


def reset_hook_resolutions(
    token: contextvars.Token[dict[str, dict[str, Any]]],
) -> None:
    """Reset hook resolutions context."""
    _hook_resolutions.reset(token)


class HookPending(Exception):
    """
    Raised when a hook needs resolution in serverless context.

    Use with Hook.get_or_raise() for suspend/resume patterns.

    Attributes:
        hook_id: Unique identifier for this hook instance.
        hook_type: The hook class name (e.g., "CommunicationApproval").
        metadata: Context data about the hook (e.g., tool_call_id, args).
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API response."""
        return {
            "hook_id": self.hook_id,
            "hook_type": self.hook_type,
            "metadata": self.metadata,
        }


def _make_hook_message(
    hook_id: str,
    hook_type: str,
    status: messages_.HookPart.model_fields["status"].annotation,  # type: ignore[name-defined]
    metadata: dict[str, Any],
    resolution: dict[str, Any] | None = None,
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
    Base class for hooks - suspension points in the execution graph.

    Hooks are created via the @hook decorator on a pydantic model.
    The model defines the schema for the resolution data.
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

        Args:
            metadata: Optional context data (e.g., tool_call_id, args).
                     This is included in the HookPart for the UI/handler.

        Returns:
            The resolved pydantic model instance.

        Raises:
            ValueError: If called outside of an ai.run() context.
        """
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

    @classmethod
    def resolve(cls, hook_id: str, data: T | dict[str, Any]) -> None:
        """
        Resolve a pending hook by ID.

        Can be called from outside the graph (API handler, websocket, etc.)
        to unblock the awaiting coroutine.

        Args:
            hook_id: The hook's unique identifier (from HookPart.hook_id).
            data: The resolution data - either a dict or the pydantic model instance.

        Raises:
            ValueError: If no pending hook exists with the given ID.
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

        Args:
            hook_id: The hook's unique identifier.
            reason: Optional cancellation reason.

        Raises:
            ValueError: If no pending hook exists with the given ID.
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

        Useful for inspecting metadata or other properties.

        Args:
            hook_id: The hook's unique identifier.

        Returns:
            The Hook instance.

        Raises:
            ValueError: If no pending hook exists with the given ID.
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

        For serverless suspend/resume patterns. Looks up hook_id in the
        context's resolutions (set via ai.run(..., hook_resolutions={...})).

        If found, returns the resolved model instance.
        If not found, raises HookPending for the caller to handle.

        Args:
            hook_id: Unique identifier for this hook instance.
            metadata: Context data to include in HookPending if raised.

        Returns:
            The resolved pydantic model instance.

        Raises:
            HookPending: If hook_id is not in resolutions.

        Example:
            # In serverless graph:
            approval = CommunicationApproval.get_or_raise(
                f"approval_{tc.tool_call_id}",
                metadata={"tool_name": tc.tool_name, "args": tc.tool_args}
            )
            if approval.granted:
                await ai.execute_tool(tc, tools, msg)
        """
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

    The pydantic model defines the schema for the hook's resolution data.

    Example:
        @ai.hook
        class Approval(pydantic.BaseModel):
            granted: bool
            reason: str

        # Use in graph:
        approval = await Approval.create(metadata={...})
        if approval.granted:
            ...
    """
    # Create a new class that inherits from Hook
    # We use type() to avoid TypeVar shadowing issues
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
