"""Temporal workflow — the durable agent loop.

Same logic as examples/samples/custom_loop.py.  The differences:
  1. LLM calls go through a Temporal activity (TemporalLanguageModel)
  2. Tool calls go through a Temporal activity (direct execute_activity)
  3. ai.run() drains fully inside the workflow (no streaming out)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import timedelta
from typing import Any, override

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    import vercel_ai_sdk as ai

    from activities import (
        TOOLS,
        LLMCallParams,
        LLMCallResult,
        ToolCallParams,
        llm_stream_activity,
        tool_call_activity,
    )


# ── TemporalLanguageModel ────────────────────────────────────────


class TemporalLanguageModel(ai.LanguageModel):
    """LanguageModel that delegates to a Temporal activity.

    The activity drains the full LLM stream and returns the final message.
    On replay, Temporal returns the cached result without re-executing.
    """

    @override
    async def stream(
        self,
        messages: list[ai.Message],
        tools: list[ai.Tool] | None = None,
    ) -> AsyncGenerator[ai.Message, None]:
        tool_schemas = [
            {"name": t.name, "description": t.description, "schema": t.schema}
            for t in (tools or [])
        ]
        result: LLMCallResult = await workflow.execute_activity(
            llm_stream_activity,
            LLMCallParams(
                messages=[m.model_dump() for m in messages],
                tool_schemas=tool_schemas,
            ),
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        for msg_dict in result.messages:
            yield ai.Message.model_validate(msg_dict)


# ── Tool execution via activity ──────────────────────────────────


async def execute_tool_via_activity(
    tool_call: ai.ToolPart, message: ai.Message
) -> None:
    """Execute a tool call through a Temporal activity, then update the ToolPart in-place."""
    args = json.loads(tool_call.tool_args)
    result = await workflow.execute_activity(
        tool_call_activity,
        ToolCallParams(tool_name=tool_call.tool_name, tool_args=args),
        start_to_close_timeout=timedelta(minutes=2),
    )
    # ToolPart.result is typed as dict[str, Any] — wrap raw values so
    # the message survives Pydantic serialize/deserialize at the activity boundary.
    if not isinstance(result, dict):
        result = {"value": result}
    tool_call.set_result(result)


# ── Agent function ───────────────────────────────────────────────


async def agent(llm: ai.LanguageModel, user_query: str) -> ai.StreamResult:
    tools = list(TOOLS.values())
    messages = ai.make_messages(
        system="Answer questions using the weather and population tools.",
        user=user_query,
    )

    while True:
        result = await ai.stream_step(llm, messages, tools, label="agent")

        if not result.tool_calls:
            return result

        messages.append(result.last_message)
        await asyncio.gather(
            *(
                execute_tool_via_activity(tc, result.last_message)
                for tc in result.tool_calls
            )
        )


# ── Workflow ─────────────────────────────────────────────────────


@workflow.defn
class AgentWorkflow:
    @workflow.run
    async def run(self, user_query: str) -> str:
        llm = TemporalLanguageModel()
        final_text = ""
        async for msg in ai.run(agent, llm, user_query):
            if msg.text:
                final_text = msg.text
        return final_text
