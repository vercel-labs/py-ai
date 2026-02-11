"""Multi-agent example with parallel execution, hooks, and live streaming display.

Two sub-agents run in parallel — one contacts the mothership, the other contacts
the data centers. Each branch requires approval via a hook before its tool can
execute. A third agent summarizes the combined results.

Run:
    AI_GATEWAY_API_KEY=... python examples/run_multiagent_hooks.py
"""

import asyncio
import os
from collections import defaultdict

import pydantic
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

import vercel_ai_sdk as ai

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
# Live display
# ---------------------------------------------------------------------------

LABELS = ["mothership", "data_centers", "summary"]
COLORS = {"mothership": "cyan", "data_centers": "magenta", "summary": "green"}
TITLES = {
    "mothership": "Agent 1 (mothership)",
    "data_centers": "Agent 2 (data centers)",
    "summary": "Agent 3 (summary)",
}


class MultiAgentHookDisplay:
    """Rich live display for multiple parallel agent streams with hook events."""

    def __init__(self):
        self.streams: dict[str, Text] = defaultdict(Text)

    def update(self, msg: ai.Message) -> None:
        label = msg.label or "unknown"
        color = COLORS.get(label, "white")

        if msg.text_delta:
            self.streams[label].append(msg.text_delta, style=color)
        if msg.reasoning_delta:
            self.streams[label].append(msg.reasoning_delta, style="dim")

        for delta in msg.tool_deltas:
            self.streams[label].append(f"{delta.args_delta}", style="yellow")

        if msg.is_done:
            for part in msg.parts:
                match part:
                    case ai.ToolPart(status="pending", tool_name=name, tool_args=args):
                        self.streams[label].append(
                            f"\n-> {name}({args})", style="yellow"
                        )
                    case ai.ToolPart(status="result", tool_name=name, result=result):
                        self.streams[label].append(
                            f"\n= {name} = {result}", style="green"
                        )
            self.streams[label].append("\n")

    def add_hook_event(self, label: str, status: str, detail: str) -> None:
        color = "bold yellow" if status == "pending" else "bold green"
        self.streams[label].append(f"\n[hook {status}] {detail}\n", style=color)

    def render(self) -> Group:
        panels = []
        for label in LABELS:
            if label in self.streams:
                panels.append(
                    Panel(
                        self.streams[label],
                        title=TITLES.get(label, label),
                        border_style=COLORS.get(label, "white"),
                    )
                )
        return Group(*panels)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_hook_part(msg: ai.Message) -> ai.HookPart | None:
    for part in msg.parts:
        if isinstance(part, ai.HookPart):
            return part
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    llm = ai.anthropic.AnthropicModel(
        model="anthropic/claude-haiku-4.5",
        base_url="https://ai-gateway.vercel.sh",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    display = MultiAgentHookDisplay()

    with Live(display.render(), refresh_per_second=15) as live:
        async for msg in ai.run(multiagent, llm, "When will the robots take over?"):
            hook_part = get_hook_part(msg)

            if hook_part and hook_part.status == "pending":
                branch = hook_part.metadata.get("branch", "unknown")
                display.add_hook_event(branch, "pending", hook_part.hook_id)
                live.update(display.render())

                # Pause live display to collect user input
                live.stop()
                answer = input(
                    f"\n[{branch}] Approve {hook_part.metadata.get('tool', '?')}? (y/n): "
                )
                if answer.strip().lower() == "y":
                    Approval.resolve(
                        hook_part.hook_id,
                        {"granted": True, "reason": "approved by operator"},
                    )
                else:
                    Approval.resolve(
                        hook_part.hook_id,
                        {"granted": False, "reason": "denied by operator"},
                    )
                live.start()

            elif hook_part and hook_part.status == "resolved":
                branch = hook_part.metadata.get("branch", "unknown")
                granted = hook_part.resolution and hook_part.resolution.get("granted")
                tag = "approved" if granted else "denied"
                display.add_hook_event(branch, f"resolved ({tag})", hook_part.hook_id)
                live.update(display.render())

            else:
                display.update(msg)
                live.update(display.render())


if __name__ == "__main__":
    asyncio.run(main())
