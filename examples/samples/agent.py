"""Coding agent with local filesystem tools."""

import asyncio
import os

import vercel_ai_sdk as ai
import vercel_ai_sdk.agent as agent


async def main():
    llm = ai.openai.OpenAIModel(
        model="anthropic/claude-sonnet-4-20250514",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    coding_agent = agent.Agent(
        model=llm,
        filesystem=agent.local.LocalFilesystem(),
        system="You are a coding assistant. Use your tools to explore and modify the codebase.",
    )

    messages = ai.make_messages(user="List the files in this directory")

    async for msg in coding_agent.run(messages):
        # Auto-approve all tool calls
        if (hook := msg.get_hook_part()) and hook.status == "pending":
            agent.ToolApproval.resolve(hook.hook_id, {"granted": True})

        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
