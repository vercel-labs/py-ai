"""Middleware-based durability: a Middleware subclass backed by Temporal activities.

The agent uses the **default loop** unchanged.  A ``TemporalMiddleware`` is
passed to ``agent.run(middleware=[...])`` and intercepts ``wrap_model`` and
``wrap_tool`` to route all I/O through Temporal activities.  Temporal's event
history provides automatic replay — on crash recovery, activities return
cached results without re-executing.

This is the recommended pattern for Temporal durability: no custom loop, no
special framework hooks — just a middleware that replaces I/O with activities.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import AsyncGenerator
from typing import Any

import temporalio.common
import temporalio.workflow

with temporalio.workflow.unsafe.imports_passed_through():
    import activities

    import ai


# ── Tools ────────────────────────────────────────────────────────
#
# Defined with @tool for schema extraction.  The default loop will
# try to call them via ToolCall.__call__(), but the middleware intercepts
# and routes to Temporal activities instead.


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    raise RuntimeError("should not be called inside workflow")


@ai.tool
async def get_population(city: str) -> int:
    """Get population of a city."""
    raise RuntimeError("should not be called inside workflow")


# ── Temporal middleware ──────────────────────────────────────────


class TemporalMiddleware(ai.Middleware):
    """Middleware that routes LLM and tool I/O through Temporal activities.

    Temporal's event history provides durability — on replay, activities
    return cached results automatically.  No checkpoint needed on our side.

    The middleware maintains its own copy of the conversation so it can
    serialize the correct messages to the LLM activity at each step.
    """

    def __init__(
        self,
        initial_messages: list[ai.Message],
        tool_schemas: list[dict[str, Any]],
    ) -> None:
        self._messages: list[ai.Message] = list(initial_messages)
        self._tool_schemas = tool_schemas

    async def wrap_model(self, call: ai.middleware.ModelContext, next: Any) -> Any:
        """Call the LLM via a Temporal activity instead of the real adapter."""
        result = await temporalio.workflow.execute_activity(
            activities.llm_call_activity,
            activities.LLMCallParams(
                messages=[m.model_dump() for m in self._messages],
                tool_schemas=self._tool_schemas,
            ),
            start_to_close_timeout=datetime.timedelta(minutes=5),
            retry_policy=temporalio.common.RetryPolicy(maximum_attempts=3),
        )
        msg = ai.Message.model_validate(result.message)
        # Track the assistant response.
        self._messages.append(msg)

        # Return a StreamResult-compatible wrapper so the default loop
        # can iterate it like a normal stream.
        return _BufferedStreamResult(msg)

    async def wrap_tool(self, call: ai.middleware.ToolContext, next: Any) -> Any:
        """Execute a tool via a Temporal activity instead of the real function."""
        result = await temporalio.workflow.execute_activity(
            activities.tool_dispatch_activity,
            activities.ToolDispatchParams(
                tool_name=call.tool_name,
                tool_args=json.dumps(call.kwargs),
            ),
            start_to_close_timeout=datetime.timedelta(minutes=2),
        )

        tool_result_msg = ai.tool_message(
            ai.ToolResultPart(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                result=result.result,
                is_error=result.is_error,
            )
        )
        # Track the tool result so the next LLM call has the full conversation.
        self._messages.append(tool_result_msg)
        return tool_result_msg


class _BufferedStreamResult:
    """StreamResult-like wrapper around a single buffered message."""

    def __init__(self, message: ai.Message) -> None:
        self._message = message

    def __aiter__(self) -> AsyncGenerator[ai.Message]:
        return self._generate()

    async def _generate(self) -> AsyncGenerator[ai.Message]:
        yield self._message

    @property
    def tool_calls(self) -> list[ai.ToolCallPart]:
        return self._message.tool_calls

    @property
    def text(self) -> str:
        return self._message.text

    @property
    def usage(self) -> ai.Usage | None:
        return self._message.usage

    @property
    def output(self) -> Any:
        return self._message.output


# ── Agent (uses default loop) ────────────────────────────────────

weather_agent = ai.agent(tools=[get_weather, get_population])


# ── Workflow ─────────────────────────────────────────────────────


@temporalio.workflow.defn
class ProviderWorkflow:
    @temporalio.workflow.run
    async def run(self, user_query: str) -> str:
        model = ai.model("ai-gateway", "anthropic/claude-sonnet-4")
        messages: list[ai.Message] = [
            ai.system_message(
                "Answer questions using the weather and population tools."
            ),
            ai.user_message(user_query),
        ]

        tool_schemas = [
            {
                "name": t.name,
                "description": t.description,
                "param_schema": t.param_schema,
            }
            for t in weather_agent._tools
        ]

        temporal_mw = TemporalMiddleware(messages, tool_schemas)

        final_text = ""
        async for msg in weather_agent.run(model, messages, middleware=[temporal_mw]):
            if msg.text:
                final_text = msg.text
        return final_text
