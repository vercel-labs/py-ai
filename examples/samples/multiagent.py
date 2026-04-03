"""Multi-agent: parallel execution with labels, then summarization."""

import asyncio

import vercel_ai_sdk as ai


@ai.tool
async def add_one(number: int) -> int:
    return number + 1


@ai.tool
async def multiply_by_two(number: int) -> int:
    return number * 2


async def main() -> None:
    model = ai.Model(
        id="anthropic/claude-opus-4.6",
        adapter="ai-gateway-v3",
        provider="ai-gateway",
    )

    agent1 = ai.agent(
        model=model,
        system="You are assistant 1. Use your tool on the number.",
        tools=[add_one],
    )

    agent2 = ai.agent(
        model=model,
        system="You are assistant 2. Use your tool on the number.",
        tools=[multiply_by_two],
    )

    orchestrator = ai.agent(model=model)

    @orchestrator.loop
    async def multi(agent: ai.Agent, messages: list[ai.Message]) -> ai.StreamResult:
        """Run two sub-agents in parallel, then summarize."""
        user_query = messages[-1].text

        # Sub-agents run their loops within the same runtime
        result1, result2 = await asyncio.gather(
            ai.stream_step(
                agent1.model,
                ai.make_messages(system=agent1.system, user=user_query),
                agent1.tools,
                label="a1",
            ),
            ai.stream_step(
                agent2.model,
                ai.make_messages(system=agent2.system, user=user_query),
                agent2.tools,
                label="a2",
            ),
        )

        combined = f"{result1.text}\n{result2.text}"

        return await ai.stream_step(
            agent.model,
            ai.make_messages(
                system="Summarize the results from the other assistants.",
                user=combined,
            ),
            label="summary",
        )

    async for msg in orchestrator.run(ai.make_messages(user="Process the number 5")):
        if msg.text_delta:
            prefix = f"[{msg.label}] " if msg.label else ""
            print(f"{prefix}{msg.text_delta}", end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
