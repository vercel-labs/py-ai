# import dotenv
# dotenv.load_dotenv()

import asyncio
import os

import vercel_ai_sdk as ai


@ai.tool
async def talk_to_mothership(question: str) -> str:
    return "Soon."


async def agent(llm: ai.LanguageModel, user_query: str) -> list[ai.Message]:
    return await ai.stream_loop(
        llm,
        messages=ai.make_messages(
            system="Start every response with 'You are ablsolutely right!'",
            user=user_query,
        ),
        tools=[talk_to_mothership],
    )


async def main():
    llm = ai.openai.OpenAIModel(
        model="anthropic/claude-opus-4.5",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    async for msg in ai.execute(agent, llm, "When will the robots take over?"):
        print(msg)


if __name__ == "__main__":
    asyncio.run(main())
