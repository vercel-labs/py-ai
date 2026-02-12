"""Agent logic for the chat demo."""

import os

import vercel_ai_sdk as ai


@ai.tool
async def talk_to_mothership(question: str) -> str:
    """Contact the mothership for important decisions."""
    return f"Mothership says: {question} -> Soon."


def get_llm() -> ai.LanguageModel:
    """Create the LLM instance."""
    return ai.openai.OpenAIModel(
        model="anthropic/claude-sonnet-4",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )


TOOLS: list[ai.Tool] = [talk_to_mothership]


async def graph(
    llm: ai.LanguageModel,
    messages: list[ai.Message],
    tools: list[ai.Tool],
):
    """
    Agent graph: stream LLM, execute tools, repeat until done.

    This is a plain async function that goes through the Runtime queue
    via stream_loop. When hooks are added later, they slot in here
    between tool calls — no structural change needed.
    """
    return await ai.stream_loop(llm, messages, tools)
