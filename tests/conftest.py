from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence

import pydantic

import vercel_ai_sdk as ai
from vercel_ai_sdk.models.core import llm as llm_
from vercel_ai_sdk.types import messages as messages_


class MockLLM(ai.LanguageModel):
    """LLM that yields pre-configured response sequences, one per call.

    Converts pre-configured ``Message`` objects into ``StreamEvent`` sequences
    so the base-class ``stream()`` (which uses ``StreamHandler``) can
    reconstruct them.
    """

    def __init__(self, responses: list[list[messages_.Message]]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.call_count = 0

    async def stream_events(
        self,
        messages: list[messages_.Message],
        tools: Sequence[ai.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
    ) -> AsyncGenerator[llm_.StreamEvent]:
        if self._call_index >= len(self._responses):
            raise RuntimeError("MockLLM: no more responses configured")
        self.call_count += 1
        seq = self._responses[self._call_index]
        self._call_index += 1

        for msg in seq:
            for i, part in enumerate(msg.parts):
                if isinstance(part, messages_.TextPart):
                    bid = f"text-{i}"
                    yield llm_.TextStart(block_id=bid)
                    if part.text:
                        yield llm_.TextDelta(block_id=bid, delta=part.text)
                    yield llm_.TextEnd(block_id=bid)

                elif isinstance(part, messages_.ReasoningPart):
                    bid = f"reasoning-{i}"
                    yield llm_.ReasoningStart(block_id=bid)
                    if part.text:
                        yield llm_.ReasoningDelta(block_id=bid, delta=part.text)
                    yield llm_.ReasoningEnd(block_id=bid, signature=part.signature)

                elif isinstance(part, messages_.ToolPart):
                    yield llm_.ToolStart(
                        tool_call_id=part.tool_call_id,
                        tool_name=part.tool_name,
                    )
                    if part.tool_args:
                        yield llm_.ToolArgsDelta(
                            tool_call_id=part.tool_call_id,
                            delta=part.tool_args,
                        )
                    yield llm_.ToolEnd(tool_call_id=part.tool_call_id)

        yield llm_.MessageDone()


def text_msg(
    text: str, *, id: str = "msg-1", state: str = "done", delta: str | None = None
) -> messages_.Message:
    return messages_.Message(
        id=id,
        role="assistant",
        parts=[messages_.TextPart(text=text, state=state, delta=delta)],
    )


def tool_msg(
    *,
    id: str = "msg-1",
    tc_id: str = "tc-1",
    name: str = "test_tool",
    args: str = "{}",
    status: str = "pending",
    result: dict[str, object] | None = None,
) -> messages_.Message:
    return messages_.Message(
        id=id,
        role="assistant",
        parts=[
            messages_.ToolPart(
                tool_call_id=tc_id,
                tool_name=name,
                tool_args=args,
                status=status,
                result=result,
                state="done",
            )
        ],
    )
