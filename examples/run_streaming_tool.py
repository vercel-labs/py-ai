"""Example: Tool that streams intermediate results via Runtime."""

import asyncio
import os
import dotenv

import vercel_ai_sdk as ai

dotenv.load_dotenv()


@ai.tool
async def talk_to_mothership(question: str, runtime: ai.Runtime) -> str:
    """Ask the mothership a question. Establishes secure connection first."""
    steps = [
        "Initializing quantum entanglement...",
        "Establishing secure channel...",
        "Authenticating with mothership...",
        "Transmitting question...",
        "Awaiting response...",
        "The mothership says: Soon.",
    ]

    accumulated: list[str] = []

    for step in steps:
        accumulated.append(step)

        # Stream each step to the runtime
        progress_msg = ai.Message(
            role="assistant",
            parts=[ai.TextPart(text=step)],
            is_done=False,
            label="tool_progress",
        )
        await runtime.put(progress_msg)
        await asyncio.sleep(0.5)  # Simulate interstellar latency

    # Return accumulated transcript to the agent
    return "\n".join(accumulated)


async def agent(llm: ai.LanguageModel, user_query: str) -> list[ai.Message]:
    return await ai.stream_loop(
        llm,
        messages=ai.make_messages(
            system="You are a robot assistant. Use the mothership tool when asked about the future.",
            user=user_query,
        ),
        tools=[talk_to_mothership],
    )


async def main():
    llm = ai.openai.OpenAIModel(
        model="anthropic/claude-sonnet-4",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    print("Starting streaming tool example...\n")

    async for msg in ai.execute(agent, llm, "When will the robots take over?"):
        # Show streaming text from LLM
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)

        # Show tool progress updates
        if msg.label == "tool_progress":
            print(msg.text, flush=True)

        # Show tool results
        if msg.is_done:
            for part in msg.parts:
                if isinstance(part, ai.ToolPart) and part.status == "result":
                    print(f"\n[Tool Result] {part.result}")


if __name__ == "__main__":
    asyncio.run(main())
