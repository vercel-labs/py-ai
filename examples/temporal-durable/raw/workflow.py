"""Temporal workflow — the durable agent loop.

No framework. Plain dicts for messages, raw Anthropic response format.
The entire agent loop is ~30 lines.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities import (
        LLMCallParams,
        LLMCallResult,
        ToolCallParams,
        llm_call_activity,
        tool_call_activity,
    )


@workflow.defn
class AgentWorkflow:
    @workflow.run
    async def run(self, user_query: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_query},
        ]

        while True:
            # ── LLM call (durable) ───────────────────────────
            result: LLMCallResult = await workflow.execute_activity(
                llm_call_activity,
                LLMCallParams(messages=messages),
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            response = result.response

            # Append assistant message
            messages.append(
                {
                    "role": "assistant",
                    "content": response["content"],
                }
            )

            # Extract tool_use blocks
            tool_calls = [
                block for block in response["content"] if block["type"] == "tool_use"
            ]

            if not tool_calls:
                # No tools — extract final text and return
                text_blocks = [
                    block["text"]
                    for block in response["content"]
                    if block["type"] == "text"
                ]
                return "\n".join(text_blocks)

            # ── Parallel tool execution (each call is durable) ─
            tool_results = await asyncio.gather(
                *(
                    workflow.execute_activity(
                        tool_call_activity,
                        ToolCallParams(tool_name=tc["name"], tool_args=tc["input"]),
                        start_to_close_timeout=timedelta(minutes=2),
                    )
                    for tc in tool_calls
                )
            )

            # Append tool results as a single user message
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tc["id"],
                            "content": str(result),
                        }
                        for tc, result in zip(tool_calls, tool_results)
                    ],
                }
            )
