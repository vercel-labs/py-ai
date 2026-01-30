"""
Example: Fake serverless execution with suspend/resume.

This demonstrates how hooks work in a serverless environment:
1. Each request runs the graph until it hits a hook needing approval
2. HookPending is raised, caught by endpoint, returned to client
3. Client sends approval, next request resumes with hook_resolutions

Key API:
- ai.run(..., hook_resolutions={...}) - pass pre-resolved hooks
- Hook.get_or_raise(hook_id, metadata) - returns value or raises HookPending
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

import dotenv
import pydantic
import rich

import vercel_ai_sdk as ai

dotenv.load_dotenv()


# --- Tools and Hooks ---


@ai.tool
async def contact_mothership(query: str) -> str:
    """Contact the mothership for important decisions."""
    return f"Mothership response: {query} -> Soon."


@ai.hook
class CommunicationApproval(pydantic.BaseModel):
    """Approval required before contacting the mothership."""

    granted: bool
    reason: str


# --- State (persisted between requests) ---


@dataclass
class State:
    session_id: str
    messages: list[ai.Message] = field(default_factory=list)
    pending_tools: list[ai.ToolCall] = field(default_factory=list)
    hook_resolutions: dict[str, dict[str, Any]] = field(default_factory=dict)


_db: dict[str, State] = {}


# --- The Agent Graph ---


TOOLS_NEEDING_APPROVAL = {"contact_mothership"}


async def graph(
    llm: ai.LanguageModel,
    state: State,
    tools: list[ai.Tool],
):
    """
    Agent graph that may suspend on hooks.

    On resume, pending_tools are processed first (with resolutions in context),
    then continues to LLM if needed.
    """
    while True:
        # Phase 1: Execute pending tool calls
        if state.pending_tools:
            last_msg = state.messages[-1]

            for tc in state.pending_tools:
                if tc.tool_name in TOOLS_NEEDING_APPROVAL:
                    # This raises HookPending if not in resolutions
                    approval = CommunicationApproval.get_or_raise(
                        f"approval_{tc.tool_call_id}",
                        metadata={"tool_name": tc.tool_name, "args": tc.tool_args},
                    )

                    if approval.granted:
                        await ai.execute_tool(tc, tools, last_msg)
                    else:
                        part = last_msg.get_tool_part(tc.tool_call_id)
                        if part:
                            part.status = "result"
                            part.result = {"error": f"Rejected: {approval.reason}"}
                else:
                    await ai.execute_tool(tc, tools, last_msg)

            state.pending_tools = []

        # Phase 2: Call LLM
        result = await ai.stream_step(llm, state.messages, tools)
        if result.last_message:
            state.messages.append(result.last_message)

        # Phase 3: Check result
        if result.tool_calls:
            state.pending_tools = list(result.tool_calls)
            continue  # Back to phase 1

        # Done
        return result.text


# --- Endpoints ---


async def start_agent(session_id: str, query: str) -> dict[str, Any]:
    """Start a new agent session."""
    rich.print(f"\n[bold cyan]>>> START_AGENT({session_id})[/]")

    state = State(
        session_id=session_id,
        messages=ai.make_messages(
            system="You are a robot. Use contact_mothership when asked about the future.",
            user=query,
        ),
    )

    llm = ai.openai.OpenAIModel(
        model="anthropic/claude-sonnet-4-20250514",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )
    tools = [contact_mothership]

    try:
        result = None

        async def run_graph():
            nonlocal result
            result = await graph(llm, state, tools)

        async for msg in ai.run(run_graph, hook_resolutions={}):
            if msg.text_delta:
                print(msg.text_delta, end="", flush=True)
        print()

        _db[session_id] = state
        return {"status": "complete", "response": result}

    except ai.HookPending as e:
        # Save state with pending tools for resume
        _db[session_id] = state
        return {"status": "pending", "hook": e.to_dict()}


async def resume_agent(
    session_id: str, hook_id: str, resolution: dict
) -> dict[str, Any]:
    """Resume agent with hook resolution."""
    rich.print(f"\n[bold cyan]>>> RESUME_AGENT({session_id}, {hook_id})[/]")

    state = _db.get(session_id)
    if not state:
        return {"status": "error", "message": "Session not found"}

    # Add the new resolution
    state.hook_resolutions[hook_id] = resolution

    llm = ai.openai.OpenAIModel(
        model="anthropic/claude-sonnet-4-20250514",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )
    tools = [contact_mothership]

    try:
        result = None

        async def run_graph():
            nonlocal result
            result = await graph(llm, state, tools)

        # Pass all accumulated resolutions
        async for msg in ai.run(run_graph, hook_resolutions=state.hook_resolutions):
            if msg.text_delta:
                print(msg.text_delta, end="", flush=True)
        print()

        _db[session_id] = state
        return {"status": "complete", "response": result}

    except ai.HookPending as e:
        _db[session_id] = state
        return {"status": "pending", "hook": e.to_dict()}


# --- Simulate Client ---


async def main():
    result = await start_agent("sess-1", "When will the robots take over?")
    rich.print(f"[yellow]Client got:[/] {result}")

    while result.get("status") == "pending":
        hook = result["hook"]
        rich.print(f"\n[yellow]Client: Approving {hook['hook_type']}...[/]")
        await asyncio.sleep(0.5)

        result = await resume_agent(
            "sess-1",
            hook["hook_id"],
            {"granted": True, "reason": "User approved"},
        )
        rich.print(f"[yellow]Client got:[/] {result}")

    rich.print("\n[bold green]Done![/]")


if __name__ == "__main__":
    asyncio.run(main())
