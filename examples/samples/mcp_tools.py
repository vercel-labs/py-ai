"""Agent with Context7 MCP integration for live documentation."""

import asyncio
import os

import rich

from typing import Any

import vercel_ai_sdk as ai


async def context7_agent(llm: ai.LanguageModel, user_query: str) -> ai.StreamResult:
    """Agent with Context7 MCP tools for up-to-date library documentation."""

    context7_tools: list[ai.Tool[..., Any]] = await ai.mcp.get_http_tools(
        "https://mcp.context7.com/mcp",
        headers={"CONTEXT7_API_KEY": os.environ.get("CONTEXT7_API_KEY", "")},
        tool_prefix="context7",
    )

    return await ai.stream_loop(
        llm,
        messages=ai.make_messages(
            system="You are a helpful assistant. Use context7 to look up documentation.",
            user=user_query,
        ),
        tools=context7_tools,
        label="context7",
    )


async def main() -> None:
    llm = ai.openai.OpenAIModel(
        model="openai/gpt-4.1",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    async for msg in ai.run(
        context7_agent, llm, "How do I create middleware in Next.js?"
    ):
        rich.print(msg)


if __name__ == "__main__":
    asyncio.run(main())
