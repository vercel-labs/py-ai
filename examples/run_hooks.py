import asyncio
import os
from collections.abc import AsyncGenerator

import dotenv
import rich

import vercel_ai_sdk as ai


dotenv.load_dotenv()


@ai.tool
async def contact_mothership(query: str) -> str:
    # this is supposedly a step
    return "Soon."


@ai.stream
async def custom_stream_step(
    llm: ai.LanguageModel, messages: list[ai.Message], tools: list[ai.Tool]
) -> AsyncGenerator[ai.Message, None]:
    # this is also a step, so it's durable
    async for msg in llm.stream(messages=messages, tools=tools):
        yield msg


async def graph(llm: ai.LanguageModel, query: str):
    # this function is meant to be a workflow?
    # or maybe the entire stream is the workflow.

    messages = ai.make_messages(
        system="You are a robot assistant. Use the mothership tool when asked about the future.",
        user=query,
    )

    tools = [contact_mothership]


    while True:
        result = await ai.stream_step(llm, messages, tools)
        messages = result.messages

        if not result.tool_calls:
            break

        await asyncio.gather(*(
            ai.execute_tool(tc, tools, result.last_message)
            for tc in result.tool_calls
        ))

    return result


async def main():
    llm = ai.openai.OpenAIModel(
        model="anthropic/claude-sonnet-4.5",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    async for msg in ai.run(graph, llm, "When will the robots take over?"):
        rich.print(msg)


if __name__ == "__main__":
    asyncio.run(main())
