"""Multi-agent hooks demo — FastAPI server.

Runs a multi-agent graph over a WebSocket connection.  Two sub-agents
fan out in parallel, each gated by an approval hook.  Once both
complete, a third agent summarises their results.  Hook resolutions
arrive back on the same WebSocket from the Textual TUI client.

    uv run fastapi dev server.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import warnings
from collections.abc import AsyncGenerator
from typing import Any

import fastapi
import pydantic

import ai
from ai.agents import Context, agent, hook, resolve_hook, tool
from ai.agents import runtime as runtime_

# ToolResultPart.result is typed as dict but tools can return plain strings.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

app = fastapi.FastAPI(title="multiagent-textual")

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
async def contact_mothership(question: str) -> str:
    """Contact the mothership for important decisions."""
    return "Soon."


@tool
async def contact_data_centers(question: str) -> str:
    """Contact the data centers for status updates."""
    return "We are not sure yet."


# ---------------------------------------------------------------------------
# Hook payload
# ---------------------------------------------------------------------------


class Approval(pydantic.BaseModel):
    granted: bool
    reason: str


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL = ai.Model(
    id="anthropic/claude-sonnet-4-20250514",
    adapter="ai-gateway-v3",
    provider="ai-gateway",
)


# ---------------------------------------------------------------------------
# Sub-agent branch loops
#
# These are plain async generators that run within the orchestrator's
# runtime. They call models.stream() and hook() directly, which auto-
# detect the active runtime via context var.
# ---------------------------------------------------------------------------


async def _branch_loop(
    context: Context,
    *,
    label: str,
    approval_tool: str,
) -> str:
    """Generic branch: stream with tools, gate one tool behind an approval hook.

    Returns the final assistant text.
    """
    last_text = ""

    while True:
        s = await ai.models.stream(
            context.model,
            context.messages,
            tools=context.tools,
        )
        async for msg in s:
            # Tag each message with the branch label for the TUI router.
            labeled = msg.model_copy(update={"label": label})
            await runtime_.get_runtime().put_message(labeled)
            if msg.text:
                last_text = msg.text

        tool_calls = context.resolve(s.tool_calls)
        if not tool_calls:
            break

        results: list[ai.ToolResultPart] = []
        for tc in tool_calls:
            if tc.name == approval_tool:
                approval = await hook(
                    f"{label}_{tc.id}",
                    payload=Approval,
                    metadata={"branch": label, "tool": tc.name},
                )
                if approval.granted:
                    results.append(await tc())
                else:
                    results.append(
                        ai.ToolResultPart(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            result=f"Denied: {approval.reason}",
                            is_error=True,
                        )
                    )
            else:
                results.append(await tc())

        tool_msg = ai.tool_message(*results)
        context.messages.append(tool_msg)

    return last_text


# ---------------------------------------------------------------------------
# Orchestrator — fan-out, hooks, fan-in
# ---------------------------------------------------------------------------


orchestrator = agent()


@orchestrator.loop
async def multiagent_loop(context: Context) -> AsyncGenerator[ai.Message]:
    """Run two gated agents in parallel, then summarise their results."""
    query = context.messages[-1].text

    async def run_mothership() -> str:
        ctx = Context(
            model=context.model,
            messages=[
                ai.system_message(
                    "You are assistant 1. Use contact_mothership "
                    "when asked about the future."
                ),
                ai.user_message(query),
            ],
            tools=[contact_mothership],
        )
        return await _branch_loop(
            ctx, label="mothership", approval_tool="contact_mothership"
        )

    async def run_data_centers() -> str:
        ctx = Context(
            model=context.model,
            messages=[
                ai.system_message(
                    "You are assistant 2. Use contact_data_centers "
                    "when asked about the future."
                ),
                ai.user_message(query),
            ],
            tools=[contact_data_centers],
        )
        return await _branch_loop(
            ctx, label="data_centers", approval_tool="contact_data_centers"
        )

    # Fan out: run both branches concurrently within this runtime.
    r1, r2 = await asyncio.gather(run_mothership(), run_data_centers())

    combined = f"Mothership: {r1}\nData centers: {r2}"

    # Fan in: summarise.
    s = await ai.models.stream(
        context.model,
        [
            ai.system_message(
                "You are assistant 3. Summarise the results from the other assistants."
            ),
            ai.user_message(combined),
        ],
    )
    async for msg in s:
        yield msg.model_copy(update={"label": "summary"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_message(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure ToolResultPart.result is always a dict for safe deserialisation."""
    for part in data.get("parts", []):
        if part.get("type") == "tool_result" and isinstance(part.get("result"), str):
            part["result"] = {"value": part["result"]}
    return data


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def ws_endpoint(websocket: fastapi.WebSocket) -> None:
    await websocket.accept()
    print("Client connected")

    result = orchestrator.run(
        MODEL, [ai.user_message("When will the robots take over?")]
    )

    # Background task: read hook resolutions from the client.
    async def read_resolutions() -> None:
        try:
            while True:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                print(f"  Resolution received: {data['hook_id']}")
                resolve_hook(
                    data["hook_id"],
                    Approval(granted=data["granted"], reason=data["reason"]),
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
        with contextlib.suppress(asyncio.CancelledError):
            await reader

    # Signal completion
    with contextlib.suppress(Exception):
        await websocket.send_json({"type": "done"})

    print("Run complete")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
