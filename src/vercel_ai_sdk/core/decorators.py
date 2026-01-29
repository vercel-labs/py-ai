"""Decorators for wiring functions into the Runtime execution model."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncGenerator, Callable
from typing import Any, TypeVar

from . import messages as messages_
from . import runtime as runtime_
from . import step as step_

T = TypeVar("T")


def stream(fn: Callable[..., AsyncGenerator[messages_.Message, None]]) -> Callable[..., Any]:
    """
    Decorator: wraps an async generator to submit as a step to Runtime.
    
    The decorated function submits its work to the Runtime queue and
    blocks until execute() processes it, then returns the StepResult.
    """

    @functools.wraps(fn)
    async def wrapped(*args: Any, **kwargs: Any) -> step_.StepResult:
        rt: runtime_.Runtime | None = runtime_._runtime.get(None)
        if rt is None:
            raise ValueError("No Runtime context - must be called within ai.execute()")

        future: asyncio.Future[step_.StepResult] = asyncio.Future()

        async def step_fn() -> AsyncGenerator[messages_.Message, None]:
            async for msg in fn(*args, **kwargs):
                yield msg

        await rt.submit_step(step_fn, future)
        return await future

    return wrapped
