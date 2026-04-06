"""Agent with Context7 MCP integration for live documentation."""

import asyncio
import os
from typing import Any

import rich

import vercel_ai_sdk as ai


async def main() -> None:
    model = ai.Model(
        id="anthropic/claude-opus-4.6",
        adapter="ai-gateway-v3",
        provider="ai-gateway",
    )

    context7_tools: list[ai.Tool[..., Any]] = await ai.mcp.get_http_tools(
        "https://mcp.context7.com/mcp",
        headers={"CONTEXT7_API_KEY": os.environ.get("CONTEXT7_API_KEY", "")},
        tool_prefix="context7",
    )

    my_agent = ai.agent(
        model=model,
        system="You are a helpful assistant. Use context7 to look up documentation.",
        tools=context7_tools,
    )

    async for msg in my_agent.run(
        ai.make_messages(user="How do I create middleware in Next.js?")
    ):
        rich.print(msg)


if __name__ == "__main__":
    asyncio.run(main())
