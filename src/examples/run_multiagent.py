import asyncio
import os

import dotenv
import rich

import py_ai as ai

dotenv.load_dotenv()


def get_text(messages: list[ai.Message]) -> str:
    # this could be a method on the Message class
    # something lile message.get_text() or message.text
    for msg in reversed(messages):
        if msg.role == "assistant":
            for part in msg.parts:
                if isinstance(part, ai.TextPart):
                    return part.text
    return ""


def make_messages(system_prompt: str, user_query: str) -> list[ai.Message]:
    # Create initial messages for an agent.

    # This is boilerplate for every agent call. Could be:
    # - A helper like core.messages(system="...", user="...")
    # - Or just let stream_loop accept (system_prompt, query) directly

    return [
        ai.Message(role="system", parts=[ai.TextPart(text=system_prompt)]),
        ai.Message(role="user", parts=[ai.TextPart(text=user_query)]),
    ]


@ai.tool
async def add_one(number: int) -> int:
    return number + 1


@ai.tool
async def multiply_by_two(number: int) -> int:
    return number * 2


# --- Context7 MCP Integration ---
# Context7 provides up-to-date documentation for any library.
# Add "use context7" to prompts or let the agent auto-invoke.
# API key should be in .env as CONTEXT7_API_KEY


async def context7_example_http(llm: ai.openai.OpenAIModel, user_query: str):
    """Context7 via HTTP transport."""
    context7_tools = await ai.mcp.get_http_tools(
        "https://mcp.context7.com/mcp",
        headers={"CONTEXT7_API_KEY": os.environ.get("CONTEXT7_API_KEY", "")},
        tool_prefix="context7",
    )

    return await ai.stream_loop(
        llm,
        messages=make_messages(
            "Test run. Please reason about this test run briefly, then call resolve-library-id for 'next.js' and stop.",
            user_query,
        ),
        tools=context7_tools,
        label="context7",
    )


async def context7_example_stdio(llm: ai.openai.OpenAIModel, user_query: str):
    """Context7 via stdio transport (npx)."""
    context7_tools = await ai.mcp.get_stdio_tools(
        "npx", "-y", "@upstash/context7-mcp",
        "--api-key", os.environ.get("CONTEXT7_API_KEY", ""),
        tool_prefix="context7",
    )

    return await ai.stream_loop(
        llm,
        messages=make_messages(
            "Test run. Call resolve-library-id for 'next.js' and stop.",
            user_query,
        ),
        tools=context7_tools,
        label="context7",
    )


# Clean up connections when done (optional - happens automatically on exit)
# await ai.mcp.close_connections()


async def thinking_example(llm: ai.LanguageModel, user_query: str):
    """Example using reasoning/thinking models (GPT 5.2, o-series, or Claude with thinking)."""
    return await ai.stream_text(
        llm,
        messages=make_messages(
            "You are a helpful assistant that thinks step by step.",
            user_query,
        ),
        label="thinking",
    )


async def multiagent(llm: ai.openai.OpenAIModel, user_query: str) -> list[ai.Message]:
    stream1, stream2 = await asyncio.gather(
        ai.stream_loop(
            llm,
            messages=make_messages(
                "You are the test assistant 1.",
                f"Use your tool on the user query: {user_query}",
            ),
            tools=[add_one],
            label="a1",
        ),
        ai.stream_loop(
            llm,
            messages=make_messages(
                "You are the test assistant 2.",
                f"Use your tool on the user query: {user_query}",
            ),
            tools=[multiply_by_two],
            label="a2",
        ),
    )

    combined = "\n".join(msg.text for msg in stream1[-1:]) + "\n".join(
        msg.text for msg in stream2[-1:]
    )

    return await ai.stream_text(
        llm,
        messages=make_messages(
            "You are the test assistant 3.",
            f"Add the results of the previous two assistants: {combined}",
        ),
        label="a3",
    )


async def main():
    llm = ai.openai.OpenAIModel(
        model="openai/gpt-5.2",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
        base_url="https://ai-gateway.vercel.sh/v1",
    )

    # LLM with extended thinking using OpenAI's GPT 5.2 reasoning model via Vercel AI Gateway
    # Uses budget_tokens to control reasoning depth (or use reasoning_effort="medium")
    # See: https://vercel.com/docs/ai-gateway/openai-compat/advanced
    thinking_llm = ai.openai.OpenAIModel(
        model="openai/gpt-5.2",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
        thinking=True,
        budget_tokens=10000,
    )

    colors = {
        "a1": "cyan",
        "a2": "magenta",
        "a3": "green",
        "context7": "yellow",
        "thinking": "blue",
    }

    # regular streaming example
    # async for msg in ai.execute(multiagent, llm, user_query):
    #     label = msg.label or "unknown"
    #     color = colors.get(label, "white")
    #     rich.print(f"[{color}]■[/{color}]", end=" ", flush=True)
    #     rich.print(msg)

    # AI SDK UI-formatted SSE streaming
    # async for msg in ai.ui.ai_sdk.to_sse_stream(
    #     ai.execute(multiagent, llm, user_query)
    # ):
    #     rich.print(msg)

    # --- Context7 Example ---
    # Toggle between HTTP and stdio transport:
    context7_query = "next.js middleware"

    # HTTP transport
    async for msg in ai.execute(context7_example_http, thinking_llm, context7_query):
        label = msg.label or "unknown"
        color = colors.get(label, "white")
        rich.print(f"[{color}]■[/{color}]", end=" ", flush=True)
        rich.print(msg)

    # stdio transport
    # async for msg in ai.execute(context7_example_stdio, llm, context7_query):
    #     label = msg.label or "unknown"
    #     color = colors.get(label, "white")
    #     rich.print(f"[{color}]■[/{color}]", end=" ", flush=True)
    #     rich.print(msg)


if __name__ == "__main__":
    asyncio.run(main())
