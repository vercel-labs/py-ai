from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Sequence

import pydantic

import vercel_ai_sdk as ai
from vercel_ai_sdk.core import messages
from vercel_ai_sdk.core.messages import StructuredOutputPart


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
        output_type: type[pydantic.BaseModel] | None = None,
    ) -> AsyncGenerator[messages.Message]:
        if self._call_index >= len(self._responses):
            raise RuntimeError("MockLLM: no more responses configured")
        self.call_count += 1
        seq = self._responses[self._call_index]
        self._call_index += 1
        msg = None
        for msg in seq:
            yield msg

        # Simulate structured output validation (matching real provider behavior)
        if output_type is not None and msg is not None and msg.text:
            data = json.loads(msg.text)
            output_type.model_validate(data)  # fail fast on bad data
            part = StructuredOutputPart(
                data=data,
                output_type_name=f"{output_type.__module__}.{output_type.__qualname__}",
            )
            msg = msg.model_copy()
            msg.parts = [*msg.parts, part]
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
