"""Multi-agent example with parallel execution and tool usage."""

import asyncio
import os

import dotenv
import rich

import vercel_ai_sdk as ai

dotenv.load_dotenv()


@ai.tool
async def add_one(number: int) -> int:
    return number + 1


@ai.tool
async def multiply_by_two(number: int) -> int:
    return number * 2


async def multiagent(llm: ai.LanguageModel, user_query: str):
    """Run two agents in parallel, then combine their results."""
    stream1, stream2 = await asyncio.gather(
        ai.stream_loop(
            llm,
            messages=[
                ai.Message(role="system", parts=[ai.TextPart(text="You are assistant 1. Use your tool on the number.")]),
                ai.Message(role="user", parts=[ai.TextPart(text=user_query)]),
            ],
            tools=[add_one],
            label="a1",
        ),
        ai.stream_loop(
            llm,
            messages=[
                ai.Message(role="system", parts=[ai.TextPart(text="You are assistant 2. Use your tool on the number.")]),
                ai.Message(role="user", parts=[ai.TextPart(text=user_query)]),
            ],
            tools=[multiply_by_two],
            label="a2",
        ),
    )

    combined = "\n".join(msg.text for msg in stream1[-1:]) + "\n".join(msg.text for msg in stream2[-1:])

    return await ai.stream_text(
        llm,
        messages=[
            ai.Message(role="system", parts=[ai.TextPart(text="You are assistant 3. Summarize the results.")]),
            ai.Message(role="user", parts=[ai.TextPart(text=f"Results from the other assistants: {combined}")]),
        ],
        label="a3",
    )


async def main():
    llm = ai.openai.OpenAIModel(
        model="openai/gpt-5.2",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    colors = {"a1": "cyan", "a2": "magenta", "a3": "green"}

    async for msg in ai.execute(multiagent, llm, "Process the number 5"):
        label = msg.label or "unknown"
        color = colors.get(label, "white")
        rich.print(f"[{color}]■[/{color}]", end=" ", flush=True)
        rich.print(msg)


if __name__ == "__main__":
    asyncio.run(main())
