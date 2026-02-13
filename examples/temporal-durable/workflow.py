"""Temporal workflow — the durable agent loop.

The agent function is essentially identical to custom_loop.py.  The only
differences are:

  1. The LLM is a TemporalLanguageModel (activity-backed)
  2. The tools are wrapped with temporal_tool() (activity-backed)
  3. ai.run() drains fully inside the workflow (no streaming out)

Everything else — ai.run(), @ai.stream, ai.execute_tool, asyncio.gather
for parallel tool calls — works unchanged inside the Temporal workflow.
"""

from __future__ import annotations

import asyncio

from temporalio import workflow

# Pass our modules through the Temporal sandbox — they don't need
# to be re-imported on each replay.  This is required because the
# sandbox would otherwise try to re-exec them in a restricted env
# where network imports etc. are blocked.
with workflow.unsafe.imports_passed_through():
    import vercel_ai_sdk as ai
    from durable import TemporalLanguageModel, temporal_tools
    from tools import get_population, get_weather


# ── The agent function — same as custom_loop.py ──────────────────


async def agent(llm: ai.LanguageModel, user_query: str) -> ai.StreamResult:
    """Custom agent loop with manual tool execution.

    This is the same logic as examples/samples/custom_loop.py.
    The only difference is that llm and tools are temporal-wrapped,
    so all I/O goes through durable activities.
    """
    tools = temporal_tools([get_weather, get_population])
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
                ai.execute_tool(tc, message=result.last_message)
                for tc in result.tool_calls
            )
        )


# ── Temporal workflow definition ─────────────────────────────────


@workflow.defn
class AgentWorkflow:
    """Durable agent workflow.

    The workflow body drives ai.run() which pumps the Runtime event loop.
    All I/O (LLM calls, tool calls) is routed through Temporal activities
    by the TemporalLanguageModel and temporal_tools wrappers.

    On replay after a crash, Temporal replays the activity results from
    its event history — the workflow re-executes deterministically and
    each workflow.execute_activity() returns the cached result.
    """

    @workflow.run
    async def run(self, user_query: str) -> str:
        llm = TemporalLanguageModel()

        # ai.run() creates a Runtime, pumps the step queue, yields messages.
        # Inside a workflow we can't stream out, so we drain everything and
        # return the final text.
        run_result = ai.run(agent, llm, user_query)

        final_text = ""
        async for msg in run_result:
            # Collect the last non-empty text — that's the final answer
            if msg.text:
                final_text = msg.text

        return final_text
