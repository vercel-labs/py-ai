"""Entry point — starts a Temporal worker and executes the agent workflow.

Prerequisites:
    1. Temporal dev server: temporal server start-dev
    2. AI_GATEWAY_API_KEY environment variable set

Usage:
    uv run python main.py
    uv run python main.py "What is the weather in Tokyo?"
"""

from __future__ import annotations

import asyncio
import sys
import uuid

import temporalio.client
import temporalio.worker

import activities
import workflow

TASK_QUEUE = "agent-durable"


async def main(user_query: str) -> None:
    temporal = await temporalio.client.Client.connect("localhost:7233")

    async with temporalio.worker.Worker(
        temporal,
        task_queue=TASK_QUEUE,
        workflows=[workflow.AgentWorkflow],
        activities=[
            activities.llm_call_activity,
            activities.get_weather_activity,
            activities.get_population_activity,
        ],
    ):
        workflow_id = f"agent-durable-{uuid.uuid4().hex[:8]}"
        print(f"Workflow {workflow_id}")
        print(f"Query: {user_query}\n")

        result = await temporal.execute_workflow(
            workflow.AgentWorkflow.run,
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
