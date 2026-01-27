from __future__ import annotations

import contextvars
import types
from typing import Generic, Self, TypeVar

import pydantic

T = TypeVar("T", bound=pydantic.BaseModel)

# Context var holding pre-resolved hook states (populated from request)
_hooks: contextvars.ContextVar[dict[str, dict]] = contextvars.ContextVar(
    "hook_resolutions", default={}
)


def set_hook_resolutions(
    resolutions: dict[str, dict],
) -> contextvars.Token[dict[str, dict]]:
    return _hooks.set(resolutions)


def reset_hook_resolutions(token: contextvars.Token[dict[str, dict]]) -> None:
    _hooks.reset(token)


def get_hook_resolutions() -> dict[str, dict]:
    return _hooks.get()


class HookPending(Exception):
    """Raised when a hook is not yet resolved and execution should suspend."""

    def __init__(
        self,
        token: str,
        hook_type: str,
        metadata: dict | None = None,
    ):
        self.token = token
        self.hook_type = hook_type
        self.metadata = metadata
        super().__init__(f"Hook pending: {hook_type}:{token}")


class BaseHookImpl(Generic[T]):
    def __init__(self, name: str, schema: type[T]):
        self.name = name
        self.schema = schema

    @classmethod
    def create(cls, name: str, schema: type[T]) -> Self:
        return cls(name=name, schema=schema)

    @classmethod
    def resume(cls, token: str) -> Self:
        return cls(name=cls.name, schema=cls.schema)

    async def __await__(self) -> T:
        return await self.schema.check(token=self.token)


# decorator for declaring hooks
def hook(cls: type[T]) -> type[BaseHookImpl[T]]:
    name = f"{cls.__name__}Hook"
    return types.new_class(name, (BaseHookImpl[T], cls))
