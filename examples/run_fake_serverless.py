import asyncio
import dataclasses
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


@dataclasses.dataclass
class SessionState:
    """Persisted between requests. In real app, this would be in a database."""

    pending_tools: list[ai.ToolPart] = dataclasses.field(default_factory=list)
    resolutions: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)


# Simulated database
_db: dict[str, SessionState] = {}


async def graph(
    llm: ai.LanguageModel,
    messages: list[ai.Message],
    tools: list[ai.Tool],
    state: SessionState,
):
    """
    Main agent loop. Handles tool execution with hook-based approval.

    The key insight: we handle pending_tools BEFORE calling the LLM.
    This way, if a hook raises, we haven't made an extra LLM call yet.
    """
    while True:
        # Handle pending tool calls first (from previous request)
        if state.pending_tools:
            for tc in state.pending_tools:
                if tc.tool_name == "contact_mothership":
                    # This raises HookPending if resolution not provided
                    approval = CommunicationApproval.create_or_raise(
                        f"approval_{tc.tool_call_id}",
                        resolutions=state.resolutions,
                    )

                    if approval.granted:
                        await tc.execute()
                    else:
                        tc.set_result({"error": f"Rejected: {approval.reason}"})
                else:
                    await tc.execute()

            state.pending_tools = []

        # Now safe to call LLM
        result = await ai.stream_step(llm, messages, tools)
        if not result.messages:
            return None

        messages.append(result.last_message)

        if result.tool_calls:
            state.pending_tools = list(result.tool_calls)
        else:
            return result.text


async def pretend_endpoint(
    session_id: str,
    messages: list[ai.Message],
    hook_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Simulated HTTP endpoint for chat.

    Args:
        session_id: Unique session identifier
        messages: Full message history from frontend
        hook_resolution: Optional {"hook_id": str, "data": dict} to resolve a pending hook
    """
    # Load or create session state
    state = _db.get(session_id, SessionState())

    # Apply hook resolution if provided
    if hook_resolution:
        state.resolutions[hook_resolution["hook_id"]] = hook_resolution["data"]

    llm = ai.openai.OpenAIModel(
        model="anthropic/claude-sonnet-4-20250514",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )
    tools = [contact_mothership]

    try:
        async for msg in ai.run(graph, llm, messages, tools, state):
            if msg.text_delta:
                rich.print(msg.text_delta, end="", flush=True)
        rich.print()

        # Clear state on completion
        _db.pop(session_id, None)
        return {
            "status": "complete",
            "response": messages[-1].text if messages else None,
        }

    except ai.HookPending as e:
        # Save state for resume
        _db[session_id] = state
        return {
            "status": "pending",
            "hook": {
                "hook_id": e.hook_id,
                "hook_type": e.hook_type,
                "metadata": e.metadata,
            },
        }


# --- Simulated Frontend Client ---


async def main():
    session_id = "sess-1"

    # Build initial message history (like frontend would)
    messages = ai.make_messages(
        system="You are a robot. Use contact_mothership when asked about the future.",
        user="When will the robots take over?",
    )

    # First request - will likely hit a hook and return pending
    rich.print("[bold cyan]--- First Request ---[/bold cyan]")
    result = await pretend_endpoint(session_id, messages)
    rich.print(f"[bold]Response:[/bold] {result}\n")

    if result["status"] == "pending":
        hook = result["hook"]
        rich.print(f"[yellow]Hook pending: {hook['hook_type']}[/yellow]")
        rich.print("[yellow]Simulating user approval...[/yellow]\n")

        # Second request - same messages, plus hook resolution
        rich.print("[bold cyan]--- Second Request (with approval) ---[/bold cyan]")
        result = await pretend_endpoint(
            session_id,
            messages,  # Same message history
            hook_resolution={
                "hook_id": hook["hook_id"],
                "data": {"granted": True, "reason": "User approved via UI"},
            },
        )
        rich.print(f"[bold]Response:[/bold] {result}")


if __name__ == "__main__":
    asyncio.run(main())
