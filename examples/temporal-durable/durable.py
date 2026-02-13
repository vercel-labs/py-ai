"""Durable execution primitives — thin wrappers that route I/O through Temporal activities.

Two primitives:
  1. TemporalLanguageModel  — LanguageModel whose stream() calls an activity
  2. temporal_tool()        — wraps a Tool so its fn calls an activity

The agent code stays identical to the non-durable version.  These wrappers
are the only things that know about Temporal.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import timedelta
from typing import Any, override

from temporalio import workflow
from temporalio.common import RetryPolicy

import vercel_ai_sdk as ai
from vercel_ai_sdk.core.tools import _tool_registry

from activities import (
    LLMCallParams,
    LLMCallResult,
    ToolCallParams,
    llm_stream_activity,
    register_original_fn,
    tool_call_activity,
)


# ── TemporalLanguageModel ────────────────────────────────────────


class TemporalLanguageModel(ai.LanguageModel):
    """LanguageModel that delegates to a Temporal activity for each LLM call.

    Instead of hitting the real API, stream() calls workflow.execute_activity()
    which runs the real LLM call inside an activity (with retry, timeout, etc.).

    The activity drains the full stream and returns the final messages.  We
    yield them as-if they arrived all at once — the "buffering wrapper" that
    bridges streaming to Temporal's request/response model.
    """

    @override
    async def stream(
        self,
        messages: list[ai.Message],
        tools: list[ai.Tool] | None = None,
    ) -> AsyncGenerator[ai.Message, None]:
        # Serialize for the activity boundary
        tool_schemas = [
            {"name": t.name, "description": t.description, "schema": t.schema}
            for t in (tools or [])
        ]

        params = LLMCallParams(
            messages=[m.model_dump() for m in messages],
            tool_schemas=tool_schemas,
        )

        # Call the activity — this is the durable boundary.
        # On replay, Temporal returns the cached result without re-executing.
        result: LLMCallResult = await workflow.execute_activity(
            llm_stream_activity,
            params,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # Yield the final message(s) — the @ai.stream decorator and
        # Runtime pump see this as a (very fast) stream that completes
        # in one shot.
        for msg_dict in result.messages:
            yield ai.Message.model_validate(msg_dict)


# ── temporal_tool ────────────────────────────────────────────────


def temporal_tool(t: ai.Tool) -> ai.Tool:
    """Wrap a Tool so its execution goes through a Temporal activity.

    The original function is stashed in activities._original_tool_fns
    so the activity can find and call it.  The tool's .fn is replaced
    with a wrapper that calls workflow.execute_activity().

    Returns a *new* Tool object — the original is not mutated.
    """
    # Stash the original so the activity can find it
    register_original_fn(t.name, t.fn)

    tool_name = t.name  # capture for closure

    async def activity_calling_fn(**kwargs: Any) -> Any:
        return await workflow.execute_activity(
            tool_call_activity,
            ToolCallParams(tool_name=tool_name, tool_args=kwargs),
            start_to_close_timeout=timedelta(minutes=2),
        )

    wrapped = ai.Tool(
        name=t.name,
        description=t.description,
        schema=t.schema,
        fn=activity_calling_fn,
    )

    # HACK: replace the global registry entry so ai.execute_tool() finds
    # the wrapped version.  The activity uses _original_tool_fns to find
    # the real function.  This is the "dumbest version" — we'll extract
    # a proper framework primitive once the overall shape settles.
    _tool_registry[t.name] = wrapped

    return wrapped


def temporal_tools(tools: list[ai.Tool]) -> list[ai.Tool]:
    """Wrap a list of tools for durable execution."""
    return [temporal_tool(t) for t in tools]
