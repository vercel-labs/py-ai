"""
New step-based API example with parallel execution.

This demonstrates the composable primitives approach where:
- @ai.stream wires a generator into the Runtime
- ai.stream_llm is the raw LLM streaming primitive
- Users can write their own loops with full control
- Durability can be added by decorating individual functions
"""

import asyncio
import json
import os
from collections import defaultdict

import dotenv
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

import vercel_ai_sdk as ai

dotenv.load_dotenv()


# --- Tools (could be decorated with @step for durability) ---


@ai.tool
async def add_one(number: int) -> int:
    return number + 1


@ai.tool
async def multiply_by_two(number: int) -> int:
    return number * 2


# --- Custom streaming functions using new primitives ---


@ai.stream
async def my_stream_step(
    llm: ai.LanguageModel,
    messages: list[ai.Message],
    tools: list[ai.Tool] | None = None,
    label: str | None = None,
):
    """
    Single LLM call wrapped with @ai.stream.
    
    This submits the work to Runtime, which executes it and returns StepResult.
    You could add @w.step here for durability.
    """
    async for msg in ai.stream_llm(llm, messages, tools):
        msg.label = label
        yield msg


@ai.stream
async def my_stream_loop(
    llm: ai.LanguageModel,
    messages: list[ai.Message],
    tools: list[ai.Tool],
    label: str | None = None,
):
    """
    Full agent loop using the primitive approach.
    
    User has full control over the loop structure.
    Each iteration could be a separate durable step if needed.
    """
    local_messages = list(messages)  # Don't mutate input
    
    while True:
        # Stream LLM response
        assistant_msg = None
        async for msg in ai.stream_llm(llm, local_messages, tools):
            msg.label = label
            yield msg
            if msg.is_done:
                assistant_msg = msg
        
        if assistant_msg is None:
            break
            
        local_messages.append(assistant_msg)
        
        # Extract tool calls
        tool_calls = [
            part for part in assistant_msg.parts
            if isinstance(part, ai.ToolPart)
        ]
        
        if not tool_calls:
            break
        
        # Execute tools (each could be @step decorated for durability)
        for tool_call in tool_calls:
            tool_fn = next(t for t in tools if t.name == tool_call.tool_name)
            args = json.loads(tool_call.tool_args)
            
            result = await ai.execute_tool(tool_fn, args)
            
            # Update tool part with result
            tool_part = assistant_msg.get_tool_part(tool_call.tool_call_id)
            if tool_part:
                tool_part.status = "result"
                tool_part.result = result
            
            yield assistant_msg


# --- Main agent using new API ---


async def multiagent(llm: ai.LanguageModel, user_query: str):
    """Run two agents in parallel, then combine their results."""
    
    # Run two stream loops in parallel
    # Each returns a StepResult when complete
    result1, result2 = await asyncio.gather(
        my_stream_loop(
            llm,
            messages=ai.make_messages(
                system="You are assistant 1. Use your tool on the number.",
                user=user_query,
            ),
            tools=[add_one],
            label="a1",
        ),
        my_stream_loop(
            llm,
            messages=ai.make_messages(
                system="You are assistant 2. Use your tool on the number.",
                user=user_query,
            ),
            tools=[multiply_by_two],
            label="a2",
        ),
    )
    
    # Combine results - StepResult has .text property
    combined = f"{result1.text}\n{result2.text}"
    
    # Final summary step
    return await my_stream_step(
        llm,
        messages=ai.make_messages(
            system="You are assistant 3. Summarize the results.",
            user=f"Results from the other assistants: {combined}",
        ),
        label="a3",
    )


# --- Display (same as original) ---


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
        # Same outer API - execute() yields messages
        async for msg in ai.execute(multiagent, llm, "Process the number 5"):
            display.update(msg)
            live.update(display.render())


if __name__ == "__main__":
    asyncio.run(main())
