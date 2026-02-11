"""Multi-agent hooks example — server.

Runs the multi-agent graph and streams messages to a connected websocket
client. Hook resolutions arrive back on the same connection.

    Terminal 1:  python examples/run_multiagent_hooks/server.py
    Terminal 2:  python examples/run_multiagent_hooks/client.py
"""

import asyncio
import contextvars
import json
import os
import warnings

import pydantic
import websockets

import vercel_ai_sdk as ai

# ToolPart.result is typed as dict but tools can return plain strings;
# model_dump() still works but emits noisy warnings.  We suppress them
# here and normalize the data before sending over the wire.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

HOST = "localhost"
PORT = 8765

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
# Hook — one type, distinguished by label
# ---------------------------------------------------------------------------


@ai.hook
class Approval(pydantic.BaseModel):
    granted: bool
    reason: str


# ---------------------------------------------------------------------------
# Sub-agent branches (manual loop with hook gating)
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
    """Agent that contacts the data centers, gated by an approval hook."""
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
    """Run two gated agents in parallel, then summarize their results."""

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
            system="You are assistant 3. Summarize the results from the other assistants.",
            user=combined,
        ),
        tools=[],
        label="summary",
    )


# ---------------------------------------------------------------------------
# Websocket handler
# ---------------------------------------------------------------------------


def get_hook_part(msg: ai.Message) -> ai.HookPart | None:
    for part in msg.parts:
        if isinstance(part, ai.HookPart):
            return part
    return None


async def handle_client(ws: websockets.ServerConnection):
    print("Client connected")

    llm = ai.anthropic.AnthropicModel(
        model="anthropic/claude-haiku-4.5",
        base_url="https://ai-gateway.vercel.sh",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    # Background task: read hook resolutions from the client
    # and resolve the corresponding hooks.
    #
    # Approval.resolve() looks up the Runtime via a ContextVar that is
    # set inside ai.run()'s async generator.  We need this task to share
    # that context, so we start it lazily on the first message (by which
    # point the generator has set the var) and copy the current context.
    async def read_resolutions():
        async for raw in ws:
            data = json.loads(raw)
            print(f"  Resolution received: {data['hook_id']}")
            Approval.resolve(
                data["hook_id"],
                {"granted": data["granted"], "reason": data["reason"]},
            )

    reader: asyncio.Task[None] | None = None

    try:
        async for msg in ai.run(multiagent, llm, "When will the robots take over?"):
            # Start the reader on the first message — the Runtime
            # ContextVar is now set, so we snapshot it into the task.
            if reader is None:
                reader = asyncio.create_task(
                    read_resolutions(),
                    context=contextvars.copy_context(),
                )

            # Serialize the message. ToolPart.result is typed as
            # dict but tools can return plain strings — normalize so
            # the client can deserialize without errors.
            data = msg.model_dump()
            for part in data.get("parts", []):
                if part.get("type") == "tool" and isinstance(part.get("result"), str):
                    part["result"] = {"value": part["result"]}
            await ws.send(json.dumps(data))

            hook_part = get_hook_part(msg)
            if hook_part:
                print(f"  Hook {hook_part.status}: {hook_part.hook_id}")
    finally:
        if reader is not None:
            reader.cancel()
            try:
                await reader
            except asyncio.CancelledError:
                pass

    # Signal completion
    try:
        await ws.send(json.dumps({"type": "done"}))
    except websockets.exceptions.ConnectionClosed:
        pass
    print("Run complete")


async def main():
    async with websockets.serve(handle_client, HOST, PORT):
        print(f"Server listening on ws://{HOST}:{PORT}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
