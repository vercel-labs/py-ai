"""Example: Tool that streams intermediate results via Runtime."""

import asyncio
import os

import dotenv
from rich import print

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
        # Note: is_done is computed from parts' state; parts without state are considered done
        progress_msg = ai.Message(
            role="assistant",
            parts=[ai.TextPart(text=f"<inside of tool> {step}", state="streaming")],
            label="tool_progress",
        )
        await runtime.put_message(progress_msg)
        await asyncio.sleep(0.5)

    return "\n".join(accumulated)


async def agent(llm: ai.LanguageModel, user_query: str):
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

    async for msg in ai.run(agent, llm, "When will the robots take over?"):
        # Show streaming text from LLM
        if msg.text_delta:
            print(f"[blue]{msg.text_delta}[/blue]", end="", flush=True)

        # Show tool progress updates
        if msg.label == "tool_progress":
            print(f"[cyan]{msg.text}[/cyan]")

        # Show tool results
        if msg.is_done:
            print()
            for part in msg.parts:
                if isinstance(part, ai.ToolPart) and part.status == "result":
                    print(f"\n[green]Tool result:[/green]\n{part.result}\n")


if __name__ == "__main__":
    asyncio.run(main())
