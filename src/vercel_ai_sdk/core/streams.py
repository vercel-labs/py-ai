from __future__ import annotations

import asyncio
import dataclasses
import functools
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from . import messages as messages_


@dataclasses.dataclass
class StreamResult:
    messages: list[messages_.Message] = dataclasses.field(default_factory=list)

    @property
    def last_message(self) -> messages_.Message | None:
        return self.messages[-1] if self.messages else None

    @property
    def tool_calls(self) -> list[messages_.ToolPart]:
        """Get tool calls from the last message."""
        if self.last_message:
            return self.last_message.tool_calls
        return []

    @property
    def text(self) -> str:
        if self.last_message:
            return self.last_message.text
        return ""

    @property
    def output(self) -> Any:
        """Parsed structured output from the last message, if available."""
        if self.last_message:
            return self.last_message.output
        return None

    @property
    def usage(self) -> messages_.Usage | None:
        """Token usage from the last (most recent) LLM call."""
        if self.last_message:
            return self.last_message.usage
        return None

    @property
    def total_usage(self) -> messages_.Usage | None:
        """Accumulated token usage across all LLM calls in this result.

        Sums usage from every message that carries it (i.e. assistant
        messages produced by LLM calls).  Returns ``None`` if no message
        reported usage.
        """
        total: messages_.Usage | None = None
        for msg in self.messages:
            if msg.usage is not None:
                total = msg.usage if total is None else total + msg.usage
        return total


Stream = Callable[[], AsyncGenerator[messages_.Message]]
# maybe it should have a name and an id inferred from LLM outputs


def stream[**P](
    fn: Callable[P, AsyncGenerator[messages_.Message]],
) -> Callable[P, Awaitable[StreamResult]]:
    """
    Decorator to put an async generator into the Runtime queue.

    The decorated function submits its work to the Runtime queue and
    blocks until run() processes it, then returns the StreamResult.

    If a checkpoint exists with a cached result for this step index,
    returns the cached result without submitting to the queue (replay).
    """

    from . import runtime as runtime_

    @functools.wraps(fn)
    async def wrapped(*args: Any, **kwargs: Any) -> StreamResult:
        rt: runtime_.Runtime | None = runtime_._runtime.get(None)
        if rt is None:
            raise ValueError("No Runtime context - must be called within ai.run()")

        # Replay: return cached result if available
        cached = rt.try_replay_step()
        if cached is not None:
            return cached

        # Fresh execution: submit to queue and wait
        future: asyncio.Future[StreamResult] = asyncio.Future()

        async def stream_fn() -> AsyncGenerator[messages_.Message]:
            async for msg in fn(*args, **kwargs):
                yield msg

        await rt.put_step(stream_fn, future)
        result = await future

        # Record for checkpoint
        rt.record_step(result)
        return result

    return wrapped
