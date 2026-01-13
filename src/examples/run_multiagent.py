"""Multi-agent interaction with single Collector firehose.

All agents stream through one Collector. Messages are distinguished by labels.
The consumer (main) receives everything and can filter/display by label.

Demonstrates:
1. Fan-out: Two agents running in parallel, both pushing to same runtime
2. Agent-as-tool: Inner agent also pushes to the shared runtime
3. Final synthesis: All messages flow through one stream
"""

import asyncio
import dataclasses
import json
import os

from proto_sdk import core
from proto_sdk.openai import OpenAIModel
from rich import print

import dotenv

dotenv.load_dotenv()


# -----------------------------------------------------------------------------
# LABELED AGENT LOOP - like get_root but with label support
# -----------------------------------------------------------------------------
async def run_labeled_agent(
    runtime: core.Runtime,
    llm: core.LanguageModel,
    messages: list[core.Message],
    tools: list[core.Tool],
    label: str,
) -> None:
    """
    Run an agent loop, pushing all messages to the shared runtime with a label.

    This is essentially get_root() inlined, but:
    1. Takes runtime as parameter (doesn't create its own Collector)
    2. Labels every message before pushing

    All messages go through the same firehose - the label lets consumers
    distinguish which agent produced each message.
    """
    while True:
        tool_calls = []
        assistant_msg = None

        # Stream from LLM - each message gets labeled and pushed to shared runtime
        async for message in llm.stream(messages=messages, tools=tools):
            message.label = label  # Tag with agent identity
            await runtime.push(message)

            if message.is_done:
                assistant_msg = message
                # Collect tool calls for execution
                for part in message.parts:
                    if isinstance(part, core.ToolCallPart):
                        tool_calls.append(part)

        # Append assistant message to this agent's history
        if assistant_msg:
            messages.append(assistant_msg)

        # No tool calls = agent is done
        if not tool_calls:
            break

        # Execute tool calls and push results to the firehose
        for tool_call in tool_calls:
            tool_fn = next(t for t in tools if t.name == tool_call.tool_name)
            args = json.loads(tool_call.tool_args)

            # Check if tool wants runtime injected (for agent-as-tool pattern)
            import inspect
            sig = inspect.signature(tool_fn.fn)
            if "runtime" in sig.parameters:
                args["runtime"] = runtime

            result = await tool_fn.fn(**args)

            tool_msg = core.Message(
                role="tool",
                label=label,  # Tool results also get labeled
                parts=[
                    core.ToolResultPart(
                        tool_call_id=tool_call.tool_call_id,
                        result={"output": result},
                    )
                ],
            )
            messages.append(tool_msg)
            await runtime.push(tool_msg)


# -----------------------------------------------------------------------------
# AGENT-AS-TOOL: Wrap an agent as a tool that streams to shared runtime
# -----------------------------------------------------------------------------
def create_agent_tool(
    llm: core.LanguageModel,
    system_prompt: str,
    tool_name: str,
    tool_description: str,
) -> core.Tool:
    """
    Create a tool that runs an inner agent.

    The inner agent receives the same runtime (via injection) and pushes
    its messages to the shared firehose with its own label.
    """

    async def inner_agent_fn(query: str, runtime: core.Runtime) -> dict:
        # Inner agent has isolated message history
        # but shares the runtime/firehose with everyone else
        inner_messages = [
            core.Message(
                role="system",
                parts=[core.TextPart(text=system_prompt)],
            ),
            core.Message(
                role="user",
                parts=[core.TextPart(text=query)],
            ),
        ]

        # Run inner agent - it pushes to the same runtime
        await run_labeled_agent(
            runtime=runtime,
            llm=llm,
            messages=inner_messages,
            tools=[],
            label=f"tool:{tool_name}",
        )

        return {"status": "done"}

    return core.Tool(
        name=tool_name,
        description=tool_description,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The query for the agent"}
            },
            "required": ["query"],
        },
        fn=inner_agent_fn,
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

    async def multiagent_loop(runtime: core.Runtime) -> None:
        """
        Orchestration logic. All sub-agents push to the same runtime.

        Flow:
        1. Fan-out: parallel agents (technical + business)
        2. Agent-as-tool: orchestrator calls fact_checker
        3. Final synthesis
        """

        # =====================================================================
        # STEP 1: FAN-OUT - Two agents in parallel
        # =====================================================================
        # Both agents push to the same runtime, distinguished by label.

        messages_technical = [
            core.Message(
                role="system",
                parts=[core.TextPart(text="You are a technical expert. Give a brief (2-3 sentence) technical analysis.")],
            ),
            core.Message(
                role="user",
                parts=[core.TextPart(text=f"Technical aspects of: {user_query}")],
            ),
        ]

        messages_business = [
            core.Message(
                role="system",
                parts=[core.TextPart(text="You are a business analyst. Give a brief (2-3 sentence) business analysis.")],
            ),
            core.Message(
                role="user",
                parts=[core.TextPart(text=f"Business implications of: {user_query}")],
            ),
        ]

        # Parallel execution - both push to same runtime
        await asyncio.gather(
            run_labeled_agent(runtime, llm, messages_technical, [], "technical"),
            run_labeled_agent(runtime, llm, messages_business, [], "business"),
        )

        # =====================================================================
        # STEP 2: AGENT-AS-TOOL
        # =====================================================================
        # Orchestrator agent can call fact_checker tool.
        # When fact_checker runs, it also pushes to the same runtime.

        fact_checker_tool = create_agent_tool(
            llm,
            system_prompt="You are a fact-checker. Briefly verify or correct the given claim in 1-2 sentences.",
            tool_name="fact_checker",
            tool_description="Verify factual claims. Pass a claim to check.",
        )

        messages_orchestrator = [
            core.Message(
                role="system",
                parts=[core.TextPart(text=(
                    "You are an orchestrator. You have a fact_checker tool. "
                    "Use it to verify one key claim, then summarize."
                ))],
            ),
            core.Message(
                role="user",
                parts=[core.TextPart(text=f"Review and verify one claim about: {user_query}")],
            ),
        ]

        await run_labeled_agent(
            runtime, llm, messages_orchestrator, [fact_checker_tool], "orchestrator"
        )

        # =====================================================================
        # STEP 3: FINAL SYNTHESIS
        # =====================================================================

        messages_synthesis = [
            core.Message(
                role="system",
                parts=[core.TextPart(text="You are a senior analyst. Synthesize into 2-3 sentences.")],
            ),
            core.Message(
                role="user",
                parts=[core.TextPart(text=f"Summarize: {user_query}")],
            ),
        ]

        await run_labeled_agent(runtime, llm, messages_synthesis, [], "synthesis")

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

    # Single Collector - ALL agents stream through this one firehose
    async for msg in core.Collector().stream(root):
        # Output message as dictionary for technical evaluation
        print(dataclasses.asdict(msg))


if __name__ == "__main__":
    asyncio.run(main())
