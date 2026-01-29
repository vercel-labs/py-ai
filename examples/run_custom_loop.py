"""
Custom agent loop example demonstrating durable-compatible primitives.

This shows how to write your own agent loop using the granular primitives:
- ai.stream_step: single LLM call
- ai.execute_tool: single tool execution

Each primitive can be wrapped with a durability decorator (e.g. @workflow.step)
for use with durable execution backends like Workflow or Temporal.

Example with durability (pseudo-code):
    
    from workflow import step, workflow
    
    # Wrap primitives for durability
    durable_stream_step = step(ai.stream_step)
    durable_execute_tool = step(ai.execute_tool)
    
    @workflow
    async def my_durable_agent(llm, query):
        # Same loop, but each step is now durable
        ...
"""

import asyncio
import os

import dotenv
from rich import print

import vercel_ai_sdk as ai

dotenv.load_dotenv()


# --- Tools ---
# Each tool can be individually wrapped with @workflow.step for durability


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Simulated API call
    await asyncio.sleep(0.3)
    return f"Sunny, 72°F in {city}"


@ai.tool
async def get_population(city: str) -> int:
    """Get population of a city."""
    await asyncio.sleep(0.3)
    populations = {"new york": 8_336_817, "los angeles": 3_979_576, "chicago": 2_693_976}
    return populations.get(city.lower(), 1_000_000)


# --- Custom agent loop ---


async def custom_agent_loop(
    llm: ai.LanguageModel,
    messages: list[ai.Message],
    tools: list[ai.Tool],
    label: str | None = None,
) -> ai.StepResult:
    """
    Custom agent loop with explicit step boundaries.
    
    This is the pattern for durable execution:
    - Each stream_step call is one durable step (LLM call)
    - Each execute_tool call is one durable step (tool execution)
    - The loop itself can be the workflow entry point
    
    For durability, wrap the primitives:
        durable_stream_step = workflow.step(ai.stream_step)
        durable_execute_tool = workflow.step(ai.execute_tool)
    """
    local_messages = list(messages)
    iteration = 0

    while True:
        iteration += 1
        print(f"\n[dim]── Loop iteration {iteration} ──[/dim]")

        # STEP 1: LLM call (can be @workflow.step wrapped)
        result = await ai.stream_step(llm, local_messages, tools, label=label)

        if not result.tool_calls:
            print(f"[dim]No tool calls, finishing.[/dim]")
            return result

        print(f"[yellow]Tool calls: {[tc.tool_name for tc in result.tool_calls]}[/yellow]")
        local_messages.append(result.last_message)

        # STEP 2: Execute tools in parallel (each can be @workflow.step wrapped)
        await asyncio.gather(*(
            ai.execute_tool(tc, tools, result.last_message)
            for tc in result.tool_calls
        ))

        print(f"[green]Tools executed successfully[/green]")


# --- Main agent ---


async def agent(llm: ai.LanguageModel, user_query: str):
    """Agent using custom loop with city information tools."""
    
    return await custom_agent_loop(
        llm,
        messages=ai.make_messages(
            system=(
                "You are a helpful assistant with access to city information tools. "
                "When asked about cities, use the tools to get accurate data. "
                "You can call multiple tools in parallel."
            ),
            user=user_query,
        ),
        tools=[get_weather, get_population],
        label="city_agent",
    )


async def main():
    llm = ai.anthropic.AnthropicModel(
        model="anthropic/claude-sonnet-4",
        base_url="https://ai-gateway.vercel.sh",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    print("[bold]Custom Loop Agent Demo[/bold]")
    print("This demonstrates the durable-compatible agent loop pattern.\n")

    async for msg in ai.run(
        agent, llm, "What's the weather and population of New York and Los Angeles?"
    ):
        # Show streaming text
        if msg.text_delta:
            print(f"[blue]{msg.text_delta}[/blue]", end="", flush=True)

        # Show tool status
        if msg.is_done:
            for part in msg.parts:
                if isinstance(part, ai.ToolPart):
                    if part.status == "pending":
                        print(f"\n[yellow]→ Calling {part.tool_name}({part.tool_args})[/yellow]")
                    elif part.status == "result":
                        print(f"[green]✓ {part.tool_name} = {part.result}[/green]")

    print("\n[bold]Done![/bold]")


if __name__ == "__main__":
    asyncio.run(main())
