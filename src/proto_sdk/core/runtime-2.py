from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncGenerator, Awaitable
from typing import Callable


# decorator that introspects functions
# and turns them into tool schemae
def tool():
    pass


# all data withing the framework gets normalized to this
# one message type that is made of of these parts
@dataclasses.dataclass
class Part:
    content: str


@dataclasses.dataclass
class Message:
    parts: list[Part]


class LanguageModel:
    async def stream(self) -> AsyncGenerator[Message, None]:
        yield Message(parts=[Part(content="foo")])


class _Sentinel:
    pass


_SENTINEL = _Sentinel()


# owns message queue
# exposes .stream() that yields messages from queue
# is consumer and also transforms whatever is consumed into a generator
class Loop:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[Message | _Sentinel] = asyncio.Queue()
        self._done: bool = False

    async def stream(
        self, root: Callable[[Runtime], Awaitable[None]]
    ) -> AsyncGenerator[Message]:
        runtime = Runtime(self)

        async with asyncio.TaskGroup() as g:
            _ = g.create_task(self._stop_when_done(root(runtime)))

            while True:
                message = await self.queue.get()
                if isinstance(message, _Sentinel):
                    break
                yield message

    # checks if the task has quit and gracefully
    # shuts down the queue (why do we need this?)
    async def _stop_when_done(self, task: Awaitable[None]):
        try:
            await task
        finally:
            self.queue.put_nowait(_SENTINEL)  # FIXME: swap out for a proper sentinel

    def close(self) -> None:
        self._done = True
        self.queue.put_nowait(_SENTINEL)


# gets injected into tools so they can push to the Loop queue
# is produces
class Runtime:
    def __init__(self, loop: Loop) -> None:
        self._loop: Loop = loop
        self.llm: LanguageModel = LanguageModel()

    async def push(self, message: Message) -> None:
        await self._loop.queue.put(message)

    async def stream(self) -> AsyncGenerator[Message, None]:
        async for message in self.llm.stream():
            await self.push(message)
            yield message


# defines the execution graph for the root
# calls the LLM, invokes tools, etc
async def loop_root():
    pass


async def main():
    pass


if __name__ == "__main__":
    asyncio.run(main())
