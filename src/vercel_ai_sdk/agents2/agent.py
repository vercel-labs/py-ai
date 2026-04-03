"""Agent — the primary user-facing API.

Bundles model, system prompt, and tools into a reusable, composable
unit.  Provides a default tool-calling loop and a decorator to
override it.

Usage::

    agent = ai.agent(
        model=my_model,
        system="You are a helpful assistant.",
        tools=[get_weather, get_population],
    )

    # stream messages
    async for msg in agent.run(messages):
        print(msg.text_delta, end="")

    # or collect the final result
    result = await agent.run(messages).collect()
    print(result.text)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from .. import models2
from ..types import messages as messages_
from . import checkpoint as checkpoint_
from . import context as context_
from . import runtime as runtime_
from . import streams as streams_
from . import tools as tools_

# ── Types ─────────────────────────────────────────────────────────

LoopFn = Callable[["Agent", list[messages_.Message]], Awaitable[streams_.StreamResult]]


# ── Default loop primitives ───────────────────────────────────────


@streams_.stream
async def _stream_step(
    model: models2.Model,
    messages: list[messages_.Message],
    tools: list[tools_.Tool[..., Any]],
) -> AsyncGenerator[messages_.Message]:
    """Single LLM call that streams into the Runtime queue."""
    async for msg in models2.stream(model, messages, tools=tools):
        yield msg


async def _default_loop(
    agent: Agent, messages: list[messages_.Message]
) -> streams_.StreamResult:
    """Default agent loop: stream LLM, execute tools, repeat."""
    local_messages = list(messages)

    while True:
        result = await _stream_step(agent.model, local_messages, agent.tools)

        if not result.tool_calls:
            return result

        last_msg = result.last_message
        if last_msg is not None:
            local_messages.append(last_msg)

        await asyncio.gather(
            *(runtime_.execute_tool(tc, message=last_msg) for tc in result.tool_calls)
        )


# ── AgentRun ──────────────────────────────────────────────────────


class AgentRun:
    """Returned by ``agent.run()``.  Async-iterate for messages, then
    inspect post-run state.

    Usage::

        run = agent.run(messages)

        # streaming
        async for msg in run:
            print(msg.text_delta, end="")
        run.checkpoint       # checkpoint after iteration
        run.pending_hooks    # unresolved hooks (empty if completed)

        # non-streaming
        result = await agent.run(messages).collect()
        print(result.text)
    """

    def __init__(self, inner: runtime_.RunResult) -> None:
        self._inner = inner

    async def __aiter__(self) -> AsyncGenerator[messages_.Message]:
        async for msg in self._inner:
            yield msg

    async def collect(self) -> streams_.StreamResult:
        """Drain the stream and return a :class:`StreamResult`."""
        msgs: list[messages_.Message] = []
        async for msg in self._inner:
            msgs.append(msg)
        return streams_.StreamResult(messages=msgs)

    @property
    def checkpoint(self) -> checkpoint_.Checkpoint:
        return self._inner.checkpoint

    @property
    def pending_hooks(self) -> dict[str, runtime_.HookInfo]:
        return self._inner.pending_hooks


# ── Agent ─────────────────────────────────────────────────────────


class Agent:
    """An agent — bundles model, system prompt, tools, and loop logic.

    Create via :func:`agent`::

        weather = ai.agent(
            model=my_model,
            system="Answer questions about weather.",
            tools=[get_weather],
        )

    Tools default to all globally registered tools when ``None``
    (the default).  Pass ``tools=[]`` to explicitly disable tools.

    Override the default tool-calling loop with ``@agent.loop``::

        @weather.loop
        async def custom(agent, messages):
            ...
    """

    def __init__(
        self,
        model: models2.Model,
        system: str = "",
        tools: list[tools_.Tool[..., Any]] | None = None,
    ) -> None:
        self._model = model
        self._system = system
        self._tools = tools
        self._custom_loop: LoopFn | None = None

    @property
    def model(self) -> models2.Model:
        return self._model

    @property
    def system(self) -> str:
        return self._system

    @property
    def tools(self) -> list[tools_.Tool[..., Any]]:
        """Registered tools.  ``None`` at init resolves to all globally
        registered tools at access time."""
        if self._tools is None:
            return list(tools_._tool_registry.values())
        return list(self._tools)

    def loop(self, fn: LoopFn) -> LoopFn:
        """Decorator to override the default agent loop.

        The decorated function receives the :class:`Agent` instance and
        the per-run messages::

            @my_agent.loop
            async def custom(
                agent: ai.Agent, messages: list[ai.Message],
            ) -> ai.StreamResult:
                ...
        """
        self._custom_loop = fn
        return fn

    def run(
        self,
        messages: list[messages_.Message],
        *,
        checkpoint: checkpoint_.Checkpoint | None = None,
    ) -> AgentRun:
        """Run the agent.

        Returns an :class:`AgentRun` — async-iterate for streamed
        messages, or call ``.collect()`` for the final result.

        Args:
            messages: Conversation messages (user, assistant, etc.).
            checkpoint: Resume from a previous checkpoint.
        """
        # Prepend system prompt
        full_messages: list[messages_.Message] = []
        if self._system:
            full_messages.append(
                messages_.Message(
                    role="system",
                    parts=[messages_.TextPart(text=self._system)],
                )
            )
        full_messages.extend(messages)

        loop_fn = self._custom_loop or _default_loop
        ctx = context_.Context(tools=self.tools)

        # Build the graph function that runtime_.run() expects
        async def _graph() -> streams_.StreamResult:
            return await loop_fn(self, full_messages)

        inner = runtime_.run(
            _graph,
            checkpoint=checkpoint,
            context=ctx,
        )
        return AgentRun(inner)


# ── Factory ───────────────────────────────────────────────────────


def agent(
    model: models2.Model,
    system: str = "",
    tools: list[tools_.Tool[..., Any]] | None = None,
) -> Agent:
    """Create an :class:`Agent`.

    Args:
        model: The language model to use.
        system: System prompt.
        tools: Tools available to the agent.  ``None`` (default) means
            all globally registered tools.  Pass ``[]`` to disable.
    """
    return Agent(model=model, system=system, tools=tools)
