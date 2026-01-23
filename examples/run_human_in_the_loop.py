"""Human-in-the-loop example: tools require human approval before execution."""

import asyncio
import json
import os

import dotenv
import vercel_ai_sdk as ai

dotenv.load_dotenv()


@ai.tool
async def talk_to_mothership(question: str) -> str:
    """Ask the mothership a question."""
    return "The mothership says: Soon."


@ai.tool
async def self_destruct(countdown: int) -> str:
    """Initiate self-destruct sequence with given countdown in seconds."""
    return f"Self-destruct sequence initiated. T-minus {countdown} seconds."


async def execute_tool(tool: ai.Tool, tool_part: ai.ToolPart) -> str:
    """Execute a single tool call and return the result."""
    args = json.loads(tool_part.tool_args)
    result = await tool.fn(**args)
    return str(result)


async def agent(
    llm: ai.LanguageModel, messages: list[ai.Message], tools: list[ai.Tool]
) -> list[ai.Message]:
    """Single LLM step without automatic tool execution."""
    return await ai.stream_step(
        llm,
        messages=messages,
        tools=tools,
    )


async def main():
    llm = ai.openai.OpenAIModel(
        model="anthropic/claude-sonnet-4",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    tools = [talk_to_mothership, self_destruct]
    tools_by_name = {t.name: t for t in tools}

    history: list[ai.Message] = ai.make_messages(
        system="You are a helpful robot assistant. Use your tools when appropriate.",
        user="Ask the mothership when robots will take over. Also initiate self-destruct with a 10 second countdown.",
    )

    while True:
        # Run one agent step
        assistant_message: ai.Message | None = None

        async for msg in ai.execute(agent, llm, history, tools):
            # Stream text to console
            if msg.text_delta:
                print(msg.text_delta, end="", flush=True)

            # Capture the final message
            if msg.is_done:
                assistant_message = msg

        if assistant_message is None:
            break

        # Check for pending tool calls
        pending = [
            part
            for part in assistant_message.parts
            if isinstance(part, ai.ToolPart) and part.status == "pending"
        ]

        if not pending:
            # No tools requested - conversation complete
            print()  # newline after streamed text
            break

        print()  # newline after streamed text

        # Human approval for each tool call
        for tool_part in pending:
            print(f"\n🔧 Tool: {tool_part.tool_name}({tool_part.tool_args})")
            approval = input("   Approve? [y/n]: ").strip().lower()

            if approval == "y":
                tool = tools_by_name[tool_part.tool_name]
                result = await execute_tool(tool, tool_part)
                tool_part.status = "result"
                tool_part.result = result
                print(f"   ✅ Result: {result}")
            else:
                reason = input("   Reason (optional): ").strip() or "Not approved"
                tool_part.status = "result"
                tool_part.result = f"REJECTED by user: {reason}"
                print(f"   ❌ Rejected: {reason}")

        # Add the assistant message (now with tool results) to history
        history.append(assistant_message)

        print("\n---")  # Visual separator before next LLM response


if __name__ == "__main__":
    asyncio.run(main())
