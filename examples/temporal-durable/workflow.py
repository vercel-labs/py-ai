"""Temporal workflow — the durable agent loop.

NOTE: This example still uses the old models.LanguageModel ABC because
it wraps Temporal activities as a custom model. When the models layer
is fully migrated to models2, this will need a custom adapter instead.
"""

from __future__ import annotations

import asyncio
import datetime
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from typing import Any, override

import pydantic
import temporalio.common
import temporalio.workflow

with temporalio.workflow.unsafe.imports_passed_through():
    import activities

    import vercel_ai_sdk as ai


class DurableModel(ai.models.LanguageModel):
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
        output_type: type[pydantic.BaseModel] | None = None,
    ) -> AsyncGenerator[ai.Message]:
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
#
# TODO: This example uses the old LanguageModel ABC and ai.run() /
# ai.stream_loop free-function patterns. Once the models layer is
# migrated, convert to use ai.agent() + models2.Model with a custom
# adapter for Temporal activity-based LLM calls.


async def agent(llm: Any, user_query: str) -> ai.StreamResult:
    """Agent loop — uses old-style stream_loop via models.LanguageModel.

    This is a transitional pattern. The old ai.stream_loop and ai.run
    are no longer part of the public API. This example needs a custom
    models2 adapter to work with the new Agent API.
    """
    messages = ai.make_messages(
        system="Answer questions using the weather and population tools.",
        user=user_query,
    )

    # Manually implement the loop since we can't use Agent with LanguageModel
    tools = [get_weather, get_population]
    local_messages = list(messages)

    while True:
        result_messages: list[ai.Message] = []
        async for msg in llm.stream(local_messages, tools=tools):
            result_messages.append(msg)
        result = ai.StreamResult(messages=result_messages)

        if not result.tool_calls:
            return result

        last_msg = result.last_message
        if last_msg is not None:
            local_messages.append(last_msg)

        await asyncio.gather(
            *(ai.execute_tool(tc, message=last_msg) for tc in result.tool_calls)
        )


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

        # TODO: This uses the old free-function pattern. Once models2
        # supports custom adapters for Temporal, use Agent.run() instead.
        from vercel_ai_sdk.agents2 import run

        final_text = ""
        async for msg in run(agent, llm, user_query):
            if msg.text:
                final_text = msg.text
        return final_text
