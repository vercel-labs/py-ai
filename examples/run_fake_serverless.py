"""
Serverless hook example using checkpoint-based replay.

Demonstrates how the same graph code works in both long-running
and serverless modes. Hooks use `await` in both cases — no special
serverless API needed.

On first request:
  - Graph runs, hits a hook, branch is cancelled
  - RunResult.pending_hooks shows what needs resolving
  - Checkpoint is saved

On second request (with resolution):
  - Graph replays from checkpoint (no duplicate LLM calls or tool executions)
  - Hook resolves immediately from the provided resolution
  - Graph continues to completion
"""

import asyncio
import os
from typing import Any

import pydantic
import rich

import vercel_ai_sdk as ai


@ai.tool
async def contact_mothership(query: str) -> str:
    """Contact the mothership for important decisions."""
    return f"Mothership response: {query} -> Soon."


@ai.hook
class CommunicationApproval(pydantic.BaseModel):
    """Approval required before contacting the mothership."""

    granted: bool
    reason: str


# Same graph code as run_hooks.py — always use `await`
async def graph(
    llm: ai.LanguageModel,
    messages: list[ai.Message],
    tools: list[ai.Tool],
):
    local_messages = list(messages)

    while True:
        result = await ai.stream_step(llm, local_messages, tools)
        if not result.tool_calls:
            return result.text

        last_msg = result.last_message
        local_messages.append(last_msg)

        for tc in result.tool_calls:
            if tc.tool_name == "contact_mothership":
                # This awaits — in serverless mode, if no resolution is
                # available, the future is cancelled and the branch dies.
                approval = await CommunicationApproval.create(
                    f"approve_{tc.tool_call_id}",
                    metadata={"tool": tc.tool_name, "args": tc.tool_args},
                )

                if approval.granted:
                    await ai.execute_tool(tc, message=last_msg)
                else:
                    tc.set_result({"error": f"Rejected: {approval.reason}"})
            else:
                await ai.execute_tool(tc, message=last_msg)


# Simulated database
_db: dict[str, dict[str, Any]] = {}


async def pretend_endpoint(
    session_id: str,
    messages: list[ai.Message],
    resolutions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Simulated HTTP endpoint for chat.

    Args:
        session_id: Unique session identifier
        messages: Full message history from frontend
        resolutions: Optional hook resolutions from previous pending response
    """
    # Load checkpoint from "database"
    saved = _db.get(session_id)
    checkpoint = ai.Checkpoint.deserialize(saved) if saved else None

    llm = ai.openai.OpenAIModel(
        model="anthropic/claude-sonnet-4-20250514",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )
    tools = [contact_mothership]

    result = ai.run(
        graph,
        llm,
        messages,
        tools,
        checkpoint=checkpoint,
        resolutions=resolutions,
    )

    async for msg in result:
        if msg.text_delta:
            rich.print(msg.text_delta, end="", flush=True)
    rich.print()

    if result.pending_hooks:
        # Save checkpoint for resume
        _db[session_id] = result.checkpoint.serialize()
        return {
            "status": "pending",
            "hooks": {
                label: {"hook_type": info.hook_type, "metadata": info.metadata}
                for label, info in result.pending_hooks.items()
            },
        }
    else:
        # Clear state on completion
        _db.pop(session_id, None)
        return {"status": "complete"}


# --- Simulated Frontend Client ---


async def main():
    session_id = "sess-1"

    messages = ai.make_messages(
        system="You are a robot. Use contact_mothership when asked about the future.",
        user="When will the robots take over?",
    )

    # First request - will likely hit a hook and return pending
    rich.print("[bold cyan]--- First Request ---[/bold cyan]")
    result = await pretend_endpoint(session_id, messages)
    rich.print(f"[bold]Response:[/bold] {result}\n")

    if result["status"] == "pending":
        hooks = result["hooks"]
        for label, info in hooks.items():
            rich.print(f"[yellow]Hook pending: {label} ({info['hook_type']})[/yellow]")

        rich.print("[yellow]Simulating user approval...[/yellow]\n")

        # Second request - provide resolutions for all pending hooks
        rich.print("[bold cyan]--- Second Request (with approval) ---[/bold cyan]")
        resolutions = {
            label: {"granted": True, "reason": "User approved via UI"}
            for label in hooks
        }
        result = await pretend_endpoint(session_id, messages, resolutions=resolutions)
        rich.print(f"[bold]Response:[/bold] {result}")


if __name__ == "__main__":
    asyncio.run(main())
