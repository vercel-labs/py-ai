"""Entry point — starts a Temporal worker and executes the agent workflow.

For simplicity, the worker and client run in the same process.  In production
you'd run the worker separately and use the client to start workflows.

Prerequisites:
    1. Temporal dev server running: temporal server start-dev
    2. AI_GATEWAY_API_KEY environment variable set

Usage:
    uv run python main.py
    uv run python main.py "What is the weather in Tokyo?"
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

TASK_QUEUE = "agent-durable"


async def main(user_query: str) -> None:
    # Connect to local Temporal server
    client = await Client.connect("localhost:7233")

    # Pass pydantic and vercel_ai_sdk modules through the sandbox so they
    # don't get re-imported in the restricted environment.
    sandbox_restrictions = SandboxRestrictions.default.with_passthrough_modules(
        "pydantic",
        "pydantic_core",
        "vercel_ai_sdk",
    )

    # Start the worker in the background — it picks up workflow and
    # activity tasks from the queue.
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflow],
        activities=[llm_stream_activity, tool_call_activity],
        workflow_runner=SandboxedWorkflowRunner(restrictions=sandbox_restrictions),
    ):
        # Execute the workflow and wait for the result.
        # Each execution gets a unique ID so we can re-run freely.
        workflow_id = f"agent-{uuid.uuid4().hex[:8]}"

        print(f"Starting workflow {workflow_id}")
        print(f"Query: {user_query}")
        print()

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
        else "What's the weather and population of New York and Los Angeles?"
    )
    asyncio.run(main(query))
