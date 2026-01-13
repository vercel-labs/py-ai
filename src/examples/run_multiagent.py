"""Multi-agent interaction with single Collector firehose.

All agents stream through one Collector. Messages are distinguished by labels.
The consumer (main) receives everything and can filter/display by label.

Demonstrates:
1. Fan-out: Two agents running in parallel, both pushing to same runtime
2. Agent-as-tool: Inner agent also pushes to the shared runtime
3. Final synthesis: All messages flow through one stream
"""

import asyncio
import inspect
import json
import os

from proto_sdk import core
from proto_sdk.openai import OpenAIModel
from rich import print

import dotenv

dotenv.load_dotenv()


# -----------------------------------------------------------------------------
# AGENT ABSTRACTION
# -----------------------------------------------------------------------------
class Agent:
    """
    Encapsulates agent configuration and provides methods for running/composing.

    An Agent holds:
    - llm: The language model to use
    - system_prompt: The agent's identity/instructions
    - tools: Optional list of tools the agent can use
    - label: Optional label for multi-agent streaming (distinguishes messages)

    Usage patterns:
    1. Direct run: `await agent.run(runtime, "user message")`
    2. As tool: `agent.as_tool("name", "description")` returns a Tool
    3. Custom loop: Use `agent.create_messages()` and run your own loop
    """

    def __init__(
        self,
        llm: core.LanguageModel,
        system_prompt: str,
        tools: list[core.Tool] | None = None,
        label: str | None = None,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.label = label

    def create_messages(self, user_message: str) -> list[core.Message]:
        """Create initial message list with system prompt and user message."""
        return [
            core.Message(
                role="system",
                parts=[core.TextPart(text=self.system_prompt)],
            ),
            core.Message(
                role="user",
                parts=[core.TextPart(text=user_message)],
            ),
        ]

    async def run(
        self,
        runtime: core.Runtime,
        user_message: str,
        *,
        label: str | None = None,
    ) -> None:
        """
        Run the agent loop, pushing all messages to the runtime.

        Args:
            runtime: The runtime to push messages to
            user_message: The user's input
            label: Override the agent's default label for this run
        """
        effective_label = label or self.label or "agent"
        messages = self.create_messages(user_message)

        while True:
            tool_calls = []
            assistant_msg = None

            # Stream from LLM - each message gets labeled and pushed
            async for message in self.llm.stream(messages=messages, tools=self.tools):
                message.label = effective_label
                await runtime.push(message)

                if message.is_done:
                    assistant_msg = message
                    for part in message.parts:
                        if isinstance(part, core.ToolCallPart):
                            tool_calls.append(part)

            if assistant_msg:
                messages.append(assistant_msg)

            if not tool_calls:
                break

            # Execute tool calls
            for tool_call in tool_calls:
                tool_fn = next(t for t in self.tools if t.name == tool_call.tool_name)
                args = json.loads(tool_call.tool_args)

                # Inject runtime if tool wants it (agent-as-tool pattern)
                sig = inspect.signature(tool_fn.fn)
                if "runtime" in sig.parameters:
                    args["runtime"] = runtime

                result = await tool_fn.fn(**args)

                tool_msg = core.Message(
                    role="tool",
                    label=effective_label,
                    parts=[
                        core.ToolResultPart(
                            tool_call_id=tool_call.tool_call_id,
                            result={"output": result},
                        )
                    ],
                )
                messages.append(tool_msg)
                await runtime.push(tool_msg)

    def as_tool(self, name: str, description: str) -> core.Tool:
        """
        Wrap this agent as a tool that can be used by other agents.

        The inner agent shares the runtime with its caller, pushing
        messages to the same firehose with label "tool:{name}".
        """
        async def inner_agent_fn(query: str, runtime: core.Runtime) -> dict:
            await self.run(runtime, query, label=f"tool:{name}")
            return {"status": "done"}

        return core.Tool(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query for the agent"}
                },
                "required": ["query"],
            },
            fn=inner_agent_fn,
        )

    def with_label(self, label: str) -> "Agent":
        """Return a copy of this agent with a different label."""
        return Agent(
            llm=self.llm,
            system_prompt=self.system_prompt,
            tools=self.tools,
            label=label,
        )


# -----------------------------------------------------------------------------
# CUSTOM MULTI-AGENT LOOP - the root function for Collector.stream()
# -----------------------------------------------------------------------------
def get_multiagent_root(llm: core.LanguageModel, user_query: str):
    """
    Factory that returns the root function for Collector.stream().

    The returned function orchestrates multiple agents, all streaming
    through the single runtime it receives.
    """

    # Define agents upfront - configuration is separate from execution
    technical_agent = Agent(
        llm=llm,
        system_prompt="You are a technical expert. Give a brief (2-3 sentence) technical analysis.",
        label="technical",
    )

    business_agent = Agent(
        llm=llm,
        system_prompt="You are a business analyst. Give a brief (2-3 sentence) business analysis.",
        label="business",
    )

    fact_checker = Agent(
        llm=llm,
        system_prompt="You are a fact-checker. Briefly verify or correct the given claim in 1-2 sentences.",
    )

    orchestrator = Agent(
        llm=llm,
        system_prompt=(
            "You are an orchestrator. You have a fact_checker tool. "
            "Use it to verify one key claim, then summarize."
        ),
        tools=[fact_checker.as_tool("fact_checker", "Verify factual claims. Pass a claim to check.")],
        label="orchestrator",
    )

    synthesis_agent = Agent(
        llm=llm,
        system_prompt="You are a senior analyst. Synthesize into 2-3 sentences.",
        label="synthesis",
    )

    async def multiagent_loop(runtime: core.Runtime) -> None:
        """
        Orchestration logic. All sub-agents push to the same runtime.

        Flow:
        1. Fan-out: parallel agents (technical + business)
        2. Agent-as-tool: orchestrator calls fact_checker
        3. Final synthesis
        """

        # STEP 1: FAN-OUT - Two agents in parallel
        await asyncio.gather(
            technical_agent.run(runtime, f"Technical aspects of: {user_query}"),
            business_agent.run(runtime, f"Business implications of: {user_query}"),
        )

        # STEP 2: AGENT-AS-TOOL - orchestrator can call fact_checker
        await orchestrator.run(runtime, f"Review and verify one claim about: {user_query}")

        # STEP 3: FINAL SYNTHESIS
        await synthesis_agent.run(runtime, f"Summarize: {user_query}")

    return multiagent_loop


# -----------------------------------------------------------------------------
# MAIN - Single Collector, all messages flow through one stream
# -----------------------------------------------------------------------------
async def main():
    llm = OpenAIModel(
        model="anthropic/claude-sonnet-4.5",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
        base_url="https://ai-gateway.vercel.sh/v1",
    )

    root = get_multiagent_root(
        llm,
        user_query="What are the implications of large language models for software development?",
    )

    # Legend
    print("[cyan]■ technical[/cyan]  [magenta]■ business[/magenta]  [yellow]■ orchestrator[/yellow]  [green]■ fact_checker[/green]  [blue]■ synthesis[/blue]\n")

    # Single Collector - ALL agents stream through this one firehose
    colors = {
        "technical": "cyan",
        "business": "magenta",
        "orchestrator": "yellow",
        "tool:fact_checker": "green",
        "synthesis": "blue",
    }
    async for msg in core.Collector().stream(root):
        label = msg.label or "unknown"
        color = colors.get(label, "white")
        print(f"[{color}]■[/{color}]", end="", flush=True)

    print()  # Final newline
            


if __name__ == "__main__":
    asyncio.run(main())
