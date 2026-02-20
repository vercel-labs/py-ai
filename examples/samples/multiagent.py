"""Multi-agent: parallel execution with labels, then summarization."""

import asyncio
import os

import vercel_ai_sdk as ai


@ai.tool
async def add_one(number: int) -> int:
    return number + 1


@ai.tool
async def multiply_by_two(number: int) -> int:
    return number * 2


async def multiagent(llm: ai.LanguageModel, user_query: str) -> ai.StreamResult:
    """Run two agents in parallel, then combine their results."""

    result1, result2 = await asyncio.gather(
        ai.stream_loop(
            llm,
            messages=ai.make_messages(
                system="You are assistant 1. Use your tool on the number.",
                user=user_query,
            ),
            tools=[add_one],
            label="a1",
        ),
        ai.stream_loop(
            llm,
            messages=ai.make_messages(
                system="You are assistant 2. Use your tool on the number.",
                user=user_query,
            ),
            tools=[multiply_by_two],
            label="a2",
        ),
    )

    combined = f"{result1.messages[-1].text}\n{result2.messages[-1].text}"

    return await ai.stream_loop(
        llm,
        messages=ai.make_messages(
            system="Summarize the results from the other assistants.",
            user=combined,
        ),
        tools=[],
        label="summary",
    )


async def main() -> None:
    llm = ai.anthropic.AnthropicModel(
        model="anthropic/claude-haiku-4.5",
        base_url="https://ai-gateway.vercel.sh",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    async for msg in ai.run(multiagent, llm, "Process the number 5"):
        if msg.text_delta:
            prefix = f"[{msg.label}] " if msg.label else ""
            print(f"{prefix}{msg.text_delta}", end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
