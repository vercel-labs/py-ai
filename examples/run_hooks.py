import asyncio
import os
from typing import Literal
import dotenv

from pydantic import BaseModel

import vercel_ai_sdk as ai

dotenv.load_dotenv()


# Define the hook schema
@ai.hook
class Approval(BaseModel):
    decision: Literal["approved", "denied"]
    notes: str | None = None


@ai.tool
async def dangerous_action(target: str) -> str:
    """Perform a dangerous action that requires human approval."""

    approval = await Approval.check(token="foo")

    if approval.decision == "approved":
        return f"Successfully executed action on '{target}'"
    return f"Action denied: {approval.notes or 'No reason given'}"


@ai.tool
async def safe_action(message: str) -> str:
    """A safe action that doesn't require approval."""
    return f"Safe action completed: {message}"


async def agent(
    llm: ai.LanguageModel,
    messages: list[ai.Message],
    tools: list[ai.Tool],
) -> list[ai.Message]:
    """Single agent step."""
    return await ai.stream_loop(llm, messages=messages, tools=tools)


async def main():
    llm = ai.openai.OpenAIModel(
        model="anthropic/claude-sonnet-4",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    tools = [dangerous_action, safe_action]
    history = ai.make_messages(
        system="You are a helpful assistant. Use your tools when appropriate.",
        user="Please perform a dangerous action on 'production-database' and also do a safe action saying hello.",
    )

    while True:
        try:
            # the idea is that once you pass the message history to ai.execute,
            # it figures out the state and restores it from there. so if you have suspended tools,
            # it restores that, asks for resolution etc.
            async for msg in ai.execute(agent, llm, history, tools):
                if msg.text_delta:
                    print(msg.text_delta, end="", flush=True)
            print()
            break

        # maybe we should actually catch this somewhere midway and return a normal-looking
        # message that the user can accumulate and propagate. then they could passs the same message history
        # back to the ai.execute with the hook resolved, and ai.execute would be able to restore the state
        except ai.ExecutionSuspended as e:
            for hook in e.pending_hooks:
                print(f"\n🔒 Tool '{hook.tool_name}' requires approval")
                print(f"   Hook: {hook.hook_type} (token: {hook.token})")

                approval = input("   Approve? [y/n]: ").strip().lower()

                if approval == "y":
                    notes = input("   Notes (optional): ").strip() or None
                    hook.resolve({"decision": "approved", "notes": notes})
                    print("   ✅ Approved")
                else:
                    reason = input("   Reason for denial: ").strip() or "Not approved"
                    hook.resolve({"decision": "denied", "notes": reason})
                    print(f"   ❌ Denied: {reason}")

            print("\n--- Resuming ---\n")


if __name__ == "__main__":
    asyncio.run(main())
