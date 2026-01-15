import asyncio
import os

import dotenv
import rich

import proto_sdk as ai

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


async def multiagent(
    llm: ai.openai.OpenAIModel, user_query: str
) -> list[ai.Message]:
    tech_msgs, biz_msgs = await asyncio.gather(
        ai.buffer(
            ai.stream_loop(
                llm,
                messages=make_messages(
                    "You are the test assistant 1.",
                    f"Add one to user query: {user_query}",
                ),
                tools=[],
            )
        ),
        ai.buffer(
            ai.stream_loop(
                llm,
                messages=make_messages(
                    "You are the test assistant 2.",
                    f"Multiply user query by 2: {user_query}",
                ),
                tools=[],
            )
        ),
    )

    combined = get_text(tech_msgs[-1:]) + get_text(biz_msgs[-1:])

    return await ai.buffer(
        ai.stream_text(
            llm,
            messages=make_messages(
                "You are the test assistant 3.",
                f"Add the results of the previous two assistants: {combined}",
            ),
        ),
    )


async def main():
    llm = ai.openai.OpenAIModel(
        model="anthropic/claude-sonnet-4.5",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
        base_url="https://ai-gateway.vercel.sh/v1",
    )

    user_query = (
        "Ten"
    )

    # Legend
    rich.print(
        "[cyan]■ technical[/cyan]  [magenta]■ business[/magenta]  [yellow]■ orchestrator[/yellow]  [green]■ fact_checker[/green]  [blue]■ synthesis[/blue]\n"
    )

    colors = {
        "technical": "cyan",
        "business": "magenta",
        "orchestrator": "yellow",
        "tool:fact_checker": "green",
        "synthesis": "blue",
    }

    # stream() sets up the runtime context; our flow function receives it
    async for msg in ai.execute(multiagent, llm, user_query):
        rich.print(msg)
        # label = msg.label or "unknown"
        # color = colors.get(label, "white")
        # rich.print(f"[{color}]■[/{color}]", end="", flush=True)

    # rich.print()


if __name__ == "__main__":
    asyncio.run(main())
