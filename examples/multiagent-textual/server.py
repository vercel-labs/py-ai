"""Multi-agent hooks demo — FastAPI server.

Runs a multi-agent graph over a WebSocket connection.  Two sub-agents
fan out in parallel, each gated by an approval hook.  Once both
complete, a third agent summarises their results.  Hook resolutions
arrive back on the same WebSocket from the Textual TUI client.

    uv run fastapi dev server.py
"""

from __future__ import annotations

import asyncio
import json
import os
import warnings

import pydantic
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import vercel_ai_sdk as ai

# ToolPart.result is typed as dict but tools can return plain strings.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

app = FastAPI(title="multiagent-textual")

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@ai.tool
async def contact_mothership(question: str) -> str:
    """Contact the mothership for important decisions."""
    return "Soon."


@ai.tool
async def contact_data_centers(question: str) -> str:
    """Contact the data centers for status updates."""
    return "We are not sure yet."


# ---------------------------------------------------------------------------
# Hook
# ---------------------------------------------------------------------------


@ai.hook
class Approval(pydantic.BaseModel):
    granted: bool
    reason: str


# ---------------------------------------------------------------------------
# Sub-agent branches
# ---------------------------------------------------------------------------


async def mothership_branch(llm: ai.LanguageModel, query: str):
    """Agent that contacts the mothership, gated by an approval hook."""
    messages = ai.make_messages(
        system="You are assistant 1. Use contact_mothership when asked about the future.",
        user=query,
    )
    tools = [contact_mothership]

    while True:
        result = await ai.stream_step(llm, messages, tools, label="mothership")

        if not result.tool_calls:
            break

        for tc in result.tool_calls:
            if tc.tool_name == "contact_mothership":
                approval = await Approval.create(
                    f"mothership_{tc.tool_call_id}",
                    metadata={"branch": "mothership", "tool": tc.tool_name},
                )
                if approval.granted:
                    await ai.execute_tool(tc, message=result.last_message)
                else:
                    tc.set_result(f"Denied: {approval.reason}")
            else:
                await ai.execute_tool(tc, message=result.last_message)

        messages.append(result.last_message)

    return result


async def data_center_branch(llm: ai.LanguageModel, query: str):
    """Agent that contacts data centers, gated by an approval hook."""
    messages = ai.make_messages(
        system="You are assistant 2. Use contact_data_centers when asked about the future.",
        user=query,
    )
    tools = [contact_data_centers]

    while True:
        result = await ai.stream_step(llm, messages, tools, label="data_centers")

        if not result.tool_calls:
            break

        for tc in result.tool_calls:
            if tc.tool_name == "contact_data_centers":
                approval = await Approval.create(
                    f"data_centers_{tc.tool_call_id}",
                    metadata={"branch": "data_centers", "tool": tc.tool_name},
                )
                if approval.granted:
                    await ai.execute_tool(tc, message=result.last_message)
                else:
                    tc.set_result(f"Access denied: {approval.reason}")
            else:
                await ai.execute_tool(tc, message=result.last_message)

        messages.append(result.last_message)

    return result


# ---------------------------------------------------------------------------
# Graph — fan-out, hooks, fan-in
# ---------------------------------------------------------------------------


async def multiagent(llm: ai.LanguageModel, query: str):
    """Run two gated agents in parallel, then summarise their results."""
    r1, r2 = await asyncio.gather(
        mothership_branch(llm, query),
        data_center_branch(llm, query),
    )

    combined = (
        f"Mothership: {r1.messages[-1].text}\nData centers: {r2.messages[-1].text}"
    )

    return await ai.stream_loop(
        llm,
        messages=ai.make_messages(
            system="You are assistant 3. Summarise the results from the other assistants.",
            user=combined,
        ),
        tools=[],
        label="summary",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_message(data: dict) -> dict:
    """Ensure ToolPart.result is always a dict for safe deserialisation."""
    for part in data.get("parts", []):
        if part.get("type") == "tool" and isinstance(part.get("result"), str):
            part["result"] = {"value": part["result"]}
    return data


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    llm = ai.anthropic.AnthropicModel(
        model="anthropic/claude-haiku-4.5",
        base_url="https://ai-gateway.vercel.sh",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    result = ai.run(multiagent, llm, "When will the robots take over?")

    # Background task: read hook resolutions from the client.
    async def read_resolutions():
        try:
            while True:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                print(f"  Resolution received: {data['hook_id']}")
                Approval.resolve(
                    data["hook_id"],
                    {"granted": data["granted"], "reason": data["reason"]},
                )
        except WebSocketDisconnect:
            pass

    reader = asyncio.create_task(read_resolutions())

    try:
        async for msg in result:
            data = _normalise_message(msg.model_dump())
            await websocket.send_json(data)

            if hook_part := msg.get_hook_part():
                print(f"  Hook {hook_part.status}: {hook_part.hook_id}")
    finally:
        reader.cancel()
        try:
            await reader
        except asyncio.CancelledError:
            pass

    # Signal completion
    try:
        await websocket.send_json({"type": "done"})
    except Exception:
        pass

    print("Run complete")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
