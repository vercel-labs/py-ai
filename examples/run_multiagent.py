"""Multi-agent example with parallel execution and live streaming display."""

import asyncio
import os
from collections import defaultdict

import dotenv
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

import vercel_ai_sdk as ai

dotenv.load_dotenv()


@ai.tool
async def add_one(number: int) -> int:
    return number + 1


@ai.tool
async def multiply_by_two(number: int) -> int:
    return number * 2


async def multiagent(llm: ai.LanguageModel, user_query: str):
    """Run two agents in parallel, then combine their results."""

    result1, result2 = await asyncio.gather(
        ai.stream_loop(
            llm,
            messages=ai.make_messages(
                system="You are assistant 1. Use your tool on the number.",
                user=user_query,
            ),
            tools=[add_one],
            label="a1",
        ),
        ai.stream_loop(
            llm,
            messages=ai.make_messages(
                system="You are assistant 2. Use your tool on the number.",
                user=user_query,
            ),
            tools=[multiply_by_two],
            label="a2",
        ),
    )

    combined = f"{result1.text}\n{result2.text}"

    return await ai.stream_step(
        llm,
        messages=ai.make_messages(
            system="You are assistant 3. Summarize the results.",
            user=f"Results from the other assistants: {combined}",
        ),
        label="a3",
    )


class MultiAgentDisplay:
    """Live display for multiple parallel agent streams."""

    COLORS = {"a1": "cyan", "a2": "magenta", "a3": "green"}
    TITLES = {
        "a1": "Agent 1 (add_one)",
        "a2": "Agent 2 (multiply)",
        "a3": "Agent 3 (summary)",
    }

    def __init__(self):
        self.streams: dict[str, Text] = defaultdict(Text)

    def update(self, msg: ai.Message) -> None:
        label = msg.label or "unknown"
        color = self.COLORS.get(label, "white")

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
                            f"\n→ {name}({args})", style="yellow"
                        )
                    case ai.ToolPart(status="result", tool_name=name, result=result):
                        self.streams[label].append(
                            f"\n✓ {name} = {result}", style="green"
                        )
            self.streams[label].append("\n")

    def render(self) -> Group:
        panels = []
        for label in ["a1", "a2", "a3"]:
            if label in self.streams:
                panels.append(
                    Panel(
                        self.streams[label],
                        title=self.TITLES.get(label, label),
                        border_style=self.COLORS.get(label, "white"),
                    )
                )
        return Group(*panels)


async def main():
    llm = ai.anthropic.AnthropicModel(
        model="anthropic/claude-haiku-4.5",
        base_url="https://ai-gateway.vercel.sh",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    display = MultiAgentDisplay()

    with Live(display.render(), refresh_per_second=15) as live:
        async for msg in ai.execute(multiagent, llm, "Process the number 5"):
            display.update(msg)
            live.update(display.render())


if __name__ == "__main__":
    asyncio.run(main())
