import asyncio
import os
import dataclasses
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
class State:
    session_id: str
    messages: list[ai.Message] = dataclasses.field(default_factory=list)
    pending_tools: list[ai.ToolCall] = dataclasses.field(default_factory=list)
    hook_resolutions: dict[str, dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )


_db: dict[str, State] = {}


async def graph(
    llm: ai.LanguageModel,
    state: State,
    tools: list[ai.Tool],
):
    while True:
        # handle tool calls first
        if state.pending_tools:
            last_msg = state.messages[-1]

            for tc in state.pending_tools:
                if tc.tool_name == "contact_mothership":
                    # this raises HookPending if not in resolutions
                    approval = CommunicationApproval.create_or_raise(
                        f"approval_{tc.tool_call_id}"
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

        # now call LLM since there's no risk of accidentally calling it twice
        # due to hook raising
        result = await ai.stream_step(llm, state.messages, tools)
        if result.last_message:
            state.messages.append(result.last_message)

        # prepare tools for the next iteration
        if result.tool_calls:
            state.pending_tools = list(result.tool_calls)
        else:
            # if no pending tool calls, we're done
            return result.text


async def first_pretend_endpoint(session_id: str, query: str) -> dict[str, Any]:
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
        async for msg in ai.run(graph, llm, state, tools):
            if msg.text_delta:
                rich.print(msg.text_delta, end="", flush=True)
        rich.print()

        _db[session_id] = state
        return {"status": "complete", "response": state.messages[-1].text}

    except ai.HookPending as e:
        # Save state with pending tools for resume
        _db[session_id] = state
        return {"status": "pending", "hook": e.to_dict()}


async def pretend_endpoint(
    session_id: str, hook_id: str, resolution: dict
) -> dict[str, Any]:
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
                rich.print(msg.text_delta, end="", flush=True)
        rich.print()

        _db[session_id] = state
        return {"status": "complete", "response": result}

    except ai.HookPending as e:
        _db[session_id] = state
        return {"status": "pending", "hook": e.to_dict()}


# --- Simulate Client ---


async def main():
    # call and get hooks
    result = await pretend_endpoint("sess-1", "When will the robots take over?")

    # resolve hooks and continue execution
    await pretend_endpoint("sess-1", "When will the robots take over?")



if __name__ == "__main__":
    asyncio.run(main())
