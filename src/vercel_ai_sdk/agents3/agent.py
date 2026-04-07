"""Agent, Context, StreamResult, and the stream() function."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable, Sequence
from typing import Any, Protocol

import pydantic

from .. import models, types
from . import runtime, tools as tools_


class Context(pydantic.BaseModel):
    """Everything that goes into the LLM"""

    messages: list[types.Message]
    tools: list[
        tools_.Tool[..., Any]
    ]  # TODO should be something serializable like schema

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


class LoopFn(Protocol):
    def __call__(self, context: Context) -> AsyncGenerator[types.Message]: ...


async def _default_loop(context: Context) -> AsyncGenerator[types.Message]:
    while True:
        stream = models.stream(context.model, context.messages)
        async for message in stream:
            yield message

        async with asyncio.TaskGroup() as tg:
            pass
            # todo call tools
            # yield tool messages


class Agent:
    """Bag of configuration: model + tools + loop."""

    def __init__(
        self,
        *,
        tools: list[tools_.Tool[..., Any]] | None = None,
    ) -> None:
        self._tools: list[tools_.Tool[..., Any]] = tools or []
        self._loop_fn: LoopFn = _default_loop

    def loop(self, fn: LoopFn) -> LoopFn:
        """Decorator: override the default loop function."""
        self._loop_fn = fn
        return fn

    async def run(
        self, model: models.Model, messages: list[types.Message]
    ) -> AsyncGenerator[types.Message]:
        """Run the agent loop, yielding messages to the consumer."""

        # todo: validate messages, maybe inject standard system message
        # todo: check tools, allow passing filtered list of tools

        context = Context(messages=messages, tools=self._tools)

        async for message in runtime.run(self._loop_fn(context)):
            yield message


def agent(
    *,
    model: models.Model,
    tools: list[Tool[..., Any]] | None = None,
    system: str | None = None,
) -> Agent:
    """Create an Agent."""
    return Agent(tools=tools)
