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
import warnings
from typing import Any

import fastapi
import pydantic

import vercel_ai_sdk as ai

# ToolPart.result is typed as dict but tools can return plain strings.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

app = fastapi.FastAPI(title="multiagent-textual")

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
# Model
# ---------------------------------------------------------------------------

MODEL = ai.Model(
    id="anthropic/claude-opus-4.6",
    adapter="ai-gateway-v3",
    provider="ai-gateway",
)


# ---------------------------------------------------------------------------
# Sub-agent branches (implemented as custom loops on per-branch agents)
# ---------------------------------------------------------------------------


mothership_agent = ai.agent(
    model=MODEL,
    system="You are assistant 1. Use contact_mothership when asked about the future.",
    tools=[contact_mothership],
)


@mothership_agent.loop
async def mothership_loop(
    agent: ai.Agent, messages: list[ai.Message]
) -> ai.StreamResult:
    """Agent that contacts the mothership, gated by an approval hook."""
    local_messages = list(messages)

    while True:
        result = await ai.stream_step(
            agent.model, local_messages, agent.tools, label="mothership"
        )

        if not result.tool_calls:
            break

        for tc in result.tool_calls:
            if tc.tool_name == "contact_mothership":
                # TODO: mypy doesn't support class decorators that change the
                # class type — @ai.hook returns type[Hook[T]] but mypy still
                # sees the original BaseModel.
                approval = await Approval.create(  # type: ignore[attr-defined]
                    f"mothership_{tc.tool_call_id}",
                    metadata={"branch": "mothership", "tool": tc.tool_name},
                )
                if approval.granted:
                    await ai.execute_tool(tc, message=result.last_message)
                else:
                    tc.set_error(f"Denied: {approval.reason}")
            else:
                await ai.execute_tool(tc, message=result.last_message)

        if result.last_message is not None:
            local_messages.append(result.last_message)

    return result


data_center_agent = ai.agent(
    model=MODEL,
    system="You are assistant 2. Use contact_data_centers when asked about the future.",
    tools=[contact_data_centers],
)


@data_center_agent.loop
async def data_center_loop(
    agent: ai.Agent, messages: list[ai.Message]
) -> ai.StreamResult:
    """Agent that contacts data centers, gated by an approval hook."""
    local_messages = list(messages)

    while True:
        result = await ai.stream_step(
            agent.model, local_messages, agent.tools, label="data_centers"
        )

        if not result.tool_calls:
            break

        for tc in result.tool_calls:
            if tc.tool_name == "contact_data_centers":
                # TODO: mypy doesn't support class decorators that change the
                # class type — @ai.hook returns type[Hook[T]] but mypy still
                # sees the original BaseModel.
                approval = await Approval.create(  # type: ignore[attr-defined]
                    f"data_centers_{tc.tool_call_id}",
                    metadata={"branch": "data_centers", "tool": tc.tool_name},
                )
                if approval.granted:
                    await ai.execute_tool(tc, message=result.last_message)
                else:
                    tc.set_error(f"Access denied: {approval.reason}")
            else:
                await ai.execute_tool(tc, message=result.last_message)

        if result.last_message is not None:
            local_messages.append(result.last_message)

    return result


# ---------------------------------------------------------------------------
# Orchestrator — fan-out, hooks, fan-in
# ---------------------------------------------------------------------------


orchestrator = ai.agent(model=MODEL)


@orchestrator.loop
async def multiagent_loop(
    agent: ai.Agent, messages: list[ai.Message]
) -> ai.StreamResult:
    """Run two gated agents in parallel, then summarise their results."""
    query = messages[-1].text

    # Fan out: run both sub-agent loops within this runtime
    r1, r2 = await asyncio.gather(
        mothership_loop(mothership_agent, ai.make_messages(user=query)),
        data_center_loop(data_center_agent, ai.make_messages(user=query)),
    )

    combined = (
        f"Mothership: {r1.messages[-1].text}\nData centers: {r2.messages[-1].text}"
    )

    return await ai.stream_step(
        agent.model,
        ai.make_messages(
            system="You are assistant 3. Summarise the results from the other assistants.",
            user=combined,
        ),
        label="summary",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_message(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure ToolPart.result is always a dict for safe deserialisation."""
    for part in data.get("parts", []):
        if part.get("type") == "tool" and isinstance(part.get("result"), str):
            part["result"] = {"value": part["result"]}
    return data


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def ws_endpoint(websocket: fastapi.WebSocket) -> None:
    await websocket.accept()
    print("Client connected")

    result = orchestrator.run(ai.make_messages(user="When will the robots take over?"))

    # Background task: read hook resolutions from the client.
    async def read_resolutions() -> None:
        try:
            while True:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                print(f"  Resolution received: {data['hook_id']}")
                # TODO: mypy doesn't support class decorators that change the
                # class type — @ai.hook returns type[Hook[T]] but mypy still
                # sees the original BaseModel.
                Approval.resolve(  # type: ignore[attr-defined]
                    data["hook_id"],
                    {"granted": data["granted"], "reason": data["reason"]},
                )
        except fastapi.WebSocketDisconnect:
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
async def health() -> dict[str, str]:
    return {"status": "ok"}
