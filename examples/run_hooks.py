import asyncio
import os

import pydantic
import rich

import vercel_ai_sdk as ai


@ai.tool
async def contact_mothership(query: str) -> str:
    """Contact the mothership for important decisions."""
    return "Soon."


@ai.hook
class CommunicationApproval(pydantic.BaseModel):
    granted: bool
    reason: str


async def graph(llm: ai.LanguageModel, query: str):
    messages = ai.make_messages(
        system="You are a robot assistant. Use the contact_mothership tool when asked about the future.",
        user=query,
    )

    tools = [contact_mothership]

    while True:
        result = await ai.stream_step(llm, messages, tools)

        if not result.tool_calls:
            break

        for tc in result.tool_calls:
            if tc.tool_name == "contact_mothership":
                # Create hook - blocks until resolved (long-running)
                # or cancelled (serverless)
                approval = await CommunicationApproval.create(
                    f"approve_{tc.tool_call_id}",
                    metadata={"tool": tc.tool_name},
                )

                if approval.granted:
                    await ai.execute_tool(tc, message=result.last_message)
                else:
                    tc.set_result({"error": f"Rejected: {approval.reason}"})
            else:
                await ai.execute_tool(tc, message=result.last_message)

        messages.append(result.last_message)

    return result


def get_hook_part(msg: ai.Message) -> ai.HookPart | None:
    for part in msg.parts:
        if isinstance(part, ai.HookPart):
            return part
    return None


async def main():
    llm = ai.openai.OpenAIModel(
        model="anthropic/claude-sonnet-4-20250514",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    # Run the graph and handle hook messages
    result = ai.run(graph, llm, "When will the robots take over?")
    async for msg in result:
        hook_part = get_hook_part(msg)

        if hook_part:
            rich.print(
                f"[bold yellow]Hook:[/] {hook_part.status} - {hook_part.hook_type}"
            )
            rich.print(f"  ID: {hook_part.hook_id}")
            rich.print(f"  Metadata: {hook_part.metadata}")

            if hook_part.status == "pending":
                # In a real app, this would come from user input via API/websocket
                rich.print("[bold green]Auto-approving in 1 second...[/]")
                await asyncio.sleep(1)

                # Resolve the hook
                CommunicationApproval.resolve(
                    hook_part.hook_id,
                    {"granted": True, "reason": "Auto-approved for demo"},
                )
        else:
            if msg.text_delta:
                print(msg.text_delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
