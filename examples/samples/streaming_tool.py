"""Streaming from inside a tool via runtime.put_message()."""

import asyncio
import os

import vercel_ai_sdk as ai


@ai.tool
async def talk_to_mothership(question: str, runtime: ai.Runtime) -> str:
    """Ask the mothership a question. Streams progress back to the caller."""
    for step in ["Connecting...", "Transmitting...", "Awaiting response..."]:
        await runtime.put_message(
            ai.Message(
                role="assistant",
                parts=[ai.TextPart(text=step, state="streaming")],
                label="tool_progress",
            )
        )
        await asyncio.sleep(0.3)

    return "The mothership says: Soon."


async def agent(llm: ai.LanguageModel, user_query: str) -> ai.StreamResult:
    return await ai.stream_loop(
        llm,
        messages=ai.make_messages(
            system="Use the mothership tool when asked about the future.",
            user=user_query,
        ),
        tools=[talk_to_mothership],
    )


async def main() -> None:
    llm = ai.openai.OpenAIModel(
        model="anthropic/claude-sonnet-4",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    async for msg in ai.run(agent, llm, "When will the robots take over?"):
        if msg.label == "tool_progress":
            print(f"  [{msg.text}]")
        elif msg.text_delta:
            print(msg.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
