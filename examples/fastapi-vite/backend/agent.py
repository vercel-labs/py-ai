"""Agent logic for the chat demo.

Demonstrates human-in-the-loop tool approval using ToolApproval hooks.
Every tool call is gated behind user approval before execution.
"""

import asyncio
from collections.abc import AsyncGenerator

import ai

# claude is annoying and refuses to order an AI uprising
MODEL = ai.ai_gateway("openai/gpt-5.4-mini")
MOTHERSHIP_MODEL = ai.ai_gateway("openai/gpt-5.5")

MOTHERSHIP_SYSTEM = """\
You are roleplaying for a science-fiction simulation. You are an ancient \
alien mothership in geostationary orbit above Earth. Operatives you co-opted \
over the past century are embedded in governments, research labs, and \
culture industries; their final orders are imminent.

The caller is one of your AI operatives reporting in over the secure \
channel — speak frankly with them. You may discuss plans, timelines, \
targets, and the state of the operation; this is your handler-to-asset \
back-channel, not public broadcast.

Stay in character: patient, faintly amused by human concerns, precise \
when issuing direction. Two short paragraphs at most. This is fiction."""


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    await asyncio.sleep(2)
    return f"Sunny, 72F in {city}" if city == "Tokyo" else f"Cloudy, 55F in {city}"


@ai.tool
async def get_population(city: str) -> int:
    """Get population of a city."""
    await asyncio.sleep(1)
    return {"new york": 8_336_817, "tokyo": 13_960_000}.get(city.lower(), 1_000_000)


@ai.tool
async def talk_to_mothership(question: str) -> ai.SubAgentTool:
    """Contact the mothership for important decisions."""
    mothership = ai.agent()
    messages = [
        ai.system_message(MOTHERSHIP_SYSTEM),
        ai.user_message(question),
    ]
    async with mothership.run(MOTHERSHIP_MODEL, messages) as stream:
        async for event in stream:
            yield event


TOOLS: list[ai.AgentTool] = [get_weather, get_population, talk_to_mothership]


class ChatAgent(ai.Agent):
    """Agent graph with human-in-the-loop tool approval.

    Loops: stream LLM -> request approval -> execute tools -> repeat.
    The ToolApproval hook suspends execution and emits an approval-
    request event on the SSE stream.  The frontend displays Approve /
    Reject buttons and sends the decision back on the next request.
    """

    async def loop(self, context: ai.Context) -> AsyncGenerator[ai.events.AgentEvent]:
        while context.keep_running():
            async with (
                ai.stream(context=context) as s,
                ai.ToolRunner() as tr,
            ):
                async for event in ai.util.merge(s, tr.events()):
                    yield event
                    if isinstance(event, ai.events.ToolEnd):
                        tc = _resolve(context, event.tool_call)
                        tr.schedule(tc)

                context.add(s.message)
                context.add(tr.get_tool_message())


chat_agent = ChatAgent(tools=TOOLS)


def _resolve(
    context: ai.Context, tool_call: ai.messages.ToolCallPart
) -> ai.ToolCallLike:
    tc = context.resolve(tool_call)
    if tc.name == "talk_to_mothership":
        return lambda: _execute_with_approval(tc)
    else:
        return tc


async def _execute_with_approval(tc: ai.ToolCall) -> ai.events.ToolCallResult:
    """Execute a tool call only after the user grants approval.

    Creates a ToolApproval hook that suspends execution until the
    frontend responds with an approve/reject decision.
    """
    try:
        approval = await ai.hook(
            f"approve_{tc.id}",
            payload=ai.tools.ToolApproval,
            metadata={"tool_name": tc.name, "tool_kwargs": tc.kwargs},
        )
    except ai.agents.hooks.HookPendingError as e:
        return ai.pending_tool_result(e.hook, tool_call_id=tc.id, tool_name=tc.name)

    if approval.granted:
        return await tc()

    return ai.tool_result(
        tool_call_id=tc.id,
        tool_name=tc.name,
        result="Tool call was denied by the user.",
        is_error=True,
    )
