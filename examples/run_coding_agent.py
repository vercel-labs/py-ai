import asyncio
import os
import sys

import vercel_ai_sdk as ai
import vercel_ai_sdk.agent as agent

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


async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "List the files in this directory"
    messages = ai.make_messages(user=query)

    async for msg in coding_agent.run(messages):
        # Auto-approve all tool calls
        hook = next((p for p in msg.parts if isinstance(p, ai.HookPart)), None)
        if hook and hook.status == "pending":
            agent.ToolApproval.resolve(hook.hook_id, {"granted": True})
            continue

        # Log tool calls and results
        for part in msg.parts:
            if isinstance(part, ai.ToolPart):
                if part.state == "done" and part.status == "pending":
                    print(
                        f"\n--- tool: {part.tool_name}({part.tool_args}) ---",
                        flush=True,
                    )
                elif part.status == "result":
                    result = str(part.result)
                    if len(result) > 500:
                        result = result[:500] + "..."
                    print(f"--- result: {result} ---\n", flush=True)

        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)

    print()


if __name__ == "__main__":
    asyncio.run(main())
