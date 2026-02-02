from __future__ import annotations

import asyncio
import dataclasses
import functools
from collections.abc import AsyncGenerator, Callable
from typing import Any

from . import messages as messages_


@dataclasses.dataclass
class StreamResult:
    messages: list[messages_.Message] = dataclasses.field(
        default_factory=list
    )  # TODO is there ever more than one?
    # tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def tool_calls(self) -> None:
        # extract tool parts, maybe resolve into tools
        # for better ergonomics, i.e. for tool_call in stream_result.tool_calls: tool_call.execute()
        # TODO would that work with durable @step?
        pass

    # @property
    # def last_message(self) -> messages_.Message | None:
    #     return self.messages[-1] if self.messages else None

    # @property
    # def text(self) -> str:
    #     if self.last_message:
    #         return self.last_message.text
    #     return ""


Stream = Callable[[], AsyncGenerator[messages_.Message, None]]
# maybe it should have a name and an id inferred from LLM outputs


def stream(
    fn: Callable[..., AsyncGenerator[messages_.Message, None]],
) -> Callable[..., Any]:
    """
    Decorator to put an async generator into the Runtime queue.

    The decorated function submits its work to the Runtime queue and
    blocks until run() processes it, then returns the StreamResult.
    """

    from . import runtime as runtime_

    @functools.wraps(fn)
    async def wrapped(*args: Any, **kwargs: Any) -> StreamResult:
        rt: runtime_.Runtime | None = runtime_._runtime.get(None)
        if rt is None:
            raise ValueError("No Runtime context - must be called within ai.execute()")

        future: asyncio.Future[StreamResult] = asyncio.Future()

        async def stream_fn() -> AsyncGenerator[messages_.Message, None]:
            async for msg in fn(*args, **kwargs):
                yield msg

        await rt.put_step(stream_fn, future)
        return await future

    return wrapped
