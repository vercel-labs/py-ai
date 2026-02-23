import asyncio

import vercel_ai_sdk as ai


@ai.tool
async def talk_to_mothership(question: str) -> str:
    return "Soon."


async def agent(llm: ai.LanguageModel, user_query: str) -> ai.StreamResult:
    return await ai.stream_loop(
        llm,
        messages=ai.make_messages(
            system="Start every response with 'You are absolutely right!'",
            user=user_query,
        ),
        tools=[talk_to_mothership],
    )


async def main() -> None:
    llm = ai.ai_gateway.GatewayModel(model="anthropic/claude-opus-4.6")

    async for msg in ai.run(agent, llm, "When will the robots take over?"):
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
