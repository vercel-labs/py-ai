import asyncio

import vercel_ai_sdk as ai


@ai.tool
async def talk_to_mothership(question: str) -> str:
    return "Soon."


async def main() -> None:
    model = ai.Model(
        id="anthropic/claude-opus-4.6",
        adapter="ai-gateway-v3",
        provider="ai-gateway",
    )

    my_agent = ai.agent(
        model=model,
        system="Start every response with 'You are absolutely right!'",
        tools=[talk_to_mothership],
    )

    async for msg in my_agent.run(
        ai.make_messages(user="When will the robots take over?")
    ):
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
