"""
Run the coding agent with either a local filesystem or a Vercel sandbox.

Usage:
    # Local filesystem (default):
    uv run python examples/run_coding_agent.py "List the files in this directory"

    # Vercel sandbox:
    uv run python examples/run_coding_agent.py --sandbox "What OS is this? Run uname -a"

Required env vars:
    AI_GATEWAY_API_KEY   — for the LLM

Additional env vars for --sandbox mode:
    VERCEL_TOKEN         — Vercel API token
    VERCEL_PROJECT_ID    — Vercel project ID
    VERCEL_TEAM_ID       — Vercel team ID
"""

import asyncio
import os
import sys

import vercel_ai_sdk as ai
import vercel_ai_sdk.agent as agent

import dotenv

_ = dotenv.load_dotenv(".env.local")


def make_filesystem(use_sandbox: bool) -> agent.proto.Filesystem:
    if use_sandbox:
        return agent.vercel.VercelSandbox()
    return agent.local.LocalFilesystem()


async def main():
    args = sys.argv[1:]
    use_sandbox = "--sandbox" in args
    if use_sandbox:
        args.remove("--sandbox")

    query = " ".join(args) if args else "List the files in this directory"

    llm = ai.openai.OpenAIModel(
        model="anthropic/claude-sonnet-4-20250514",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    fs = make_filesystem(use_sandbox)
    mode = "vercel sandbox" if use_sandbox else "local"
    print(f"[mode: {mode}]", flush=True)

    coding_agent = agent.Agent(
        model=llm,
        filesystem=fs,
        system="You are a coding assistant. Use your tools to explore and modify the codebase.",
    )

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

    # Clean up sandbox if used
    if use_sandbox and isinstance(fs, agent.vercel.VercelSandbox):
        await fs.stop()
        print("[sandbox stopped]", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
