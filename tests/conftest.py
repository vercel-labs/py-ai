from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence

import vercel_ai_sdk as ai
from vercel_ai_sdk.core import messages


class MockLLM(ai.LanguageModel):
    """LLM that yields pre-configured response sequences, one per call."""

    def __init__(self, responses: list[list[messages.Message]]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.call_count = 0

    async def stream(
        self,
        messages: list[messages.Message],
        tools: Sequence[ai.ToolLike] | None = None,
    ) -> AsyncGenerator[messages.Message, None]:
        if self._call_index >= len(self._responses):
            raise RuntimeError("MockLLM: no more responses configured")
        self.call_count += 1
        seq = self._responses[self._call_index]
        self._call_index += 1
        for msg in seq:
            yield msg


def text_msg(
    text: str, *, id: str = "msg-1", state: str = "done", delta: str | None = None
) -> messages.Message:
    return messages.Message(
        id=id,
        role="assistant",
        parts=[messages.TextPart(text=text, state=state, delta=delta)],
    )


def tool_msg(
    *,
    id: str = "msg-1",
    tc_id: str = "tc-1",
    name: str = "test_tool",
    args: str = "{}",
    status: str = "pending",
    result: dict[str, object] | None = None,
) -> messages.Message:
    return messages.Message(
        id=id,
        role="assistant",
        parts=[
            messages.ToolPart(
                tool_call_id=tc_id,
                tool_name=name,
                tool_args=args,
                status=status,
                result=result,
                state="done",
            )
        ],
    )
