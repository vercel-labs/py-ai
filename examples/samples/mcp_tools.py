"""Agent with Context7 MCP integration for live documentation."""

import asyncio
import os
from typing import Any

import ai


async def main() -> None:
    model = ai.ai_gateway("anthropic/claude-sonnet-4")

    context7_tools: list[ai.Tool[..., Any]] = await ai.mcp.get_http_tools(
        "https://mcp.context7.com/mcp",
        headers={"CONTEXT7_API_KEY": os.environ.get("CONTEXT7_API_KEY", "")},
        tool_prefix="context7",
    )

    my_agent = ai.agent(tools=context7_tools)

    messages = [
        ai.system_message(
            "You are a helpful assistant. Use context7 to look up documentation."
        ),
        ai.user_message("How do I create middleware in Next.js?"),
    ]

    async for msg in my_agent.run(model, messages):
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
