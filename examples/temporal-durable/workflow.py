"""Temporal workflow — the durable agent loop."""

from __future__ import annotations

import datetime
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from typing import override

import temporalio.common
import temporalio.workflow

with temporalio.workflow.unsafe.imports_passed_through():
    import vercel_ai_sdk as ai

    import activities


class DurableModel(ai.LanguageModel):
    def __init__(
        self,
        call_fn: Callable[
            [activities.LLMCallParams], Awaitable[activities.LLMCallResult]
        ],
    ) -> None:
        self.call_fn = call_fn

    @override
    async def stream(
        self,
        messages: list[ai.Message],
        tools: Sequence[ai.ToolLike] | None = None,
    ) -> AsyncGenerator[ai.Message, None]:
        result = await self.call_fn(
            activities.LLMCallParams(
                messages=[m.model_dump() for m in messages],
                tool_schemas=[
                    {
                        "name": t.name,
                        "description": t.description,
                        "param_schema": t.param_schema,
                    }
                    for t in (tools or [])
                ],
            )
        )
        yield ai.Message.model_validate(result.message)


# ── Durable tools ────────────────────────────────────────────────
#
# Plain @ai.tool — the decorator builds the JSON schema from the
# signature.  The body calls execute_activity, making each tool
# invocation a durable Temporal activity.


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return await temporalio.workflow.execute_activity(
        activities.get_weather_activity,
        args=[city],
        start_to_close_timeout=datetime.timedelta(minutes=2),
    )


@ai.tool
async def get_population(city: str) -> int:
    """Get population of a city."""
    return await temporalio.workflow.execute_activity(
        activities.get_population_activity,
        args=[city],
        start_to_close_timeout=datetime.timedelta(minutes=2),
    )


# ── Agent ────────────────────────────────────────────────────────


async def agent(llm: ai.LanguageModel, user_query: str) -> ai.StreamResult:
    """Agent loop — identical to the non-Temporal version."""
    messages = ai.make_messages(
        system="Answer questions using the weather and population tools.",
        user=user_query,
    )
    return await ai.stream_loop(llm, messages, [get_weather, get_population])


# ── Workflow ─────────────────────────────────────────────────────


@temporalio.workflow.defn
class AgentWorkflow:
    @temporalio.workflow.run
    async def run(self, user_query: str) -> str:
        llm = DurableModel(
            lambda params: temporalio.workflow.execute_activity(
                activities.llm_call_activity,
                params,
                start_to_close_timeout=datetime.timedelta(minutes=5),
                retry_policy=temporalio.common.RetryPolicy(maximum_attempts=3),
            )
        )

        final_text = ""
        async for msg in ai.run(agent, llm, user_query):
            if msg.text:
                final_text = msg.text
        return final_text
