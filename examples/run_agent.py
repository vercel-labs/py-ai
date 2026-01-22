"""Simple agent with thinking/reasoning enabled."""

import asyncio
import os

import dotenv
import rich

import vercel_ai_sdk as ai

dotenv.load_dotenv()


async def thinking_agent(llm: ai.LanguageModel, user_query: str):
    """Agent using extended thinking for step-by-step reasoning."""
    return await ai.stream_text(
        llm,
        messages=ai.make_messages(
            system="You are a helpful assistant that thinks step by step.",
            user=user_query,
        ),
        label="thinking",
    )


async def main():
    llm = ai.openai.OpenAIModel(
        model="openai/gpt-5.2",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
        thinking=True,
        budget_tokens=10000,
    )

    async for msg in ai.execute(thinking_agent, llm, "What is 25 * 47?"):
        rich.print(msg)


if __name__ == "__main__":
    asyncio.run(main())
