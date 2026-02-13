"""Entry point — starts a Temporal worker and executes the agent workflow.

Prerequisites:
    1. Temporal dev server: temporal server start-dev
    2. AI_GATEWAY_API_KEY environment variable set

Usage:
    uv run python with_sdk/main.py
    uv run python with_sdk/main.py "What is the weather in Tokyo?"
"""

from __future__ import annotations

import asyncio
import sys
import uuid

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import (
    SandboxRestrictions,
    SandboxedWorkflowRunner,
)

from activities import llm_stream_activity, tool_call_activity
from workflow import AgentWorkflow

TASK_QUEUE = "agent-sdk"


async def main(user_query: str) -> None:
    client = await Client.connect("localhost:7233")

    restrictions = SandboxRestrictions.default.with_passthrough_modules(
        "pydantic",
        "pydantic_core",
        "vercel_ai_sdk",
    )

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflow],
        activities=[llm_stream_activity, tool_call_activity],
        workflow_runner=SandboxedWorkflowRunner(restrictions=restrictions),
    ):
        workflow_id = f"agent-sdk-{uuid.uuid4().hex[:8]}"
        print(f"Workflow {workflow_id}")
        print(f"Query: {user_query}\n")

        result = await client.execute_workflow(
            AgentWorkflow.run,
            user_query,
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
        print(result)


if __name__ == "__main__":
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else ("What's the weather and population of New York and Los Angeles?")
    )
    asyncio.run(main(query))
