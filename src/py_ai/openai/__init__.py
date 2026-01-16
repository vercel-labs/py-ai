from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import Any, override

import openai

from ..core import runtime as core


def _tools_to_openai(tools: list[core.Tool]) -> list[dict[str, Any]]:
    """Convert internal Tool objects to OpenAI tool schema format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def _messages_to_openai(messages: list[core.Message]) -> list[dict[str, Any]]:
    """Convert internal messages to OpenAI API format.
    
    The Vercel AI Gateway preserves reasoning details across interactions,
    normalizing formats from different providers. This is useful for tool
    calling workflows where the model needs to resume its thought process.
    
    See: https://vercel.com/docs/ai-gateway/openai-compat/advanced
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "tool":
            for part in msg.parts:
                if isinstance(part, core.ToolResultPart):
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": part.tool_call_id,
                            "content": str(part.result),
                        }
                    )
        elif msg.role == "assistant":
            content = ""
            reasoning = ""
            tool_calls = []
            for part in msg.parts:
                if isinstance(part, core.ReasoningPart):
                    reasoning += part.reasoning
                elif isinstance(part, core.TextPart):
                    content += part.text
                elif isinstance(part, core.ToolCallPart):
                    tool_calls.append(
                        {
                            "id": part.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": part.tool_name,
                                "arguments": part.tool_args,
                            },
                        }
                    )
            entry: dict[str, Any] = {"role": "assistant"}
            if content:
                entry["content"] = content
            # Include reasoning for multi-turn context (gateway preserves this)
            if reasoning:
                entry["reasoning"] = reasoning
            if tool_calls:
                entry["tool_calls"] = tool_calls
            result.append(entry)
        else:
            # User/system messages
            content = "".join(p.text for p in msg.parts if isinstance(p, core.TextPart))
            result.append({"role": msg.role, "content": content})
    return result


class OpenAIModel(core.LanguageModel):
    """OpenAI adapter with reasoning/thinking support via Vercel AI Gateway.
    
    Supports reasoning for models like GPT 5.x, o-series, and Claude via gateway.
    Uses the Vercel AI Gateway's unified reasoning API format.
    
    See: https://vercel.com/docs/ai-gateway/openai-compat/advanced
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool = False,
        budget_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        """Initialize OpenAI model adapter.
        
        Args:
            model: Model identifier (e.g., 'openai/gpt-5.2', 'anthropic/claude-sonnet-4.5')
            base_url: API base URL (e.g., 'https://ai-gateway.vercel.sh/v1')
            api_key: API key for authentication
            thinking: Enable reasoning/thinking output
            budget_tokens: Max tokens for reasoning (mutually exclusive with reasoning_effort)
            reasoning_effort: Effort level - 'none', 'minimal', 'low', 'medium', 'high', 'xhigh'
                             (mutually exclusive with budget_tokens)
        """
        self._model = model
        self._thinking = thinking
        self._budget_tokens = budget_tokens
        self._reasoning_effort = reasoning_effort
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key=resolved_key)

    @override
    async def stream(
        self, messages: list[core.Message], tools: list[core.Tool] | None = None
    ) -> AsyncGenerator[core.Message, None]:
        openai_messages = _messages_to_openai(messages)
        openai_tools = _tools_to_openai(tools) if tools else None

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": openai_messages,
            "stream": True,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        # Enable reasoning/thinking via Vercel AI Gateway's unified format
        # See: https://vercel.com/docs/ai-gateway/openai-compat/advanced
        if self._thinking:
            reasoning_config: dict[str, Any] = {"enabled": True}
            # Use budget_tokens OR reasoning_effort (mutually exclusive per docs)
            if self._budget_tokens is not None:
                reasoning_config["max_tokens"] = self._budget_tokens
            elif self._reasoning_effort is not None:
                reasoning_config["effort"] = self._reasoning_effort
            kwargs["extra_body"] = {"reasoning": reasoning_config}

        stream = await self._client.chat.completions.create(**kwargs)

        text_content = ""
        reasoning_content = ""
        tool_calls: dict[int, dict] = {}  # index -> {id, name, args}
        message_id = core._gen_id()

        async for chunk in stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            text_delta = ""
            reasoning_delta = ""

            # Handle reasoning/thinking content via Vercel AI Gateway
            # The gateway may return reasoning in different ways:
            # 1. As a direct attribute (if SDK supports it)
            # 2. In model_extra (Pydantic v2 extra fields)
            # 3. In the raw response data
            reasoning_value = None
            if hasattr(delta, "reasoning") and delta.reasoning:
                reasoning_value = delta.reasoning
            elif hasattr(delta, "model_extra") and delta.model_extra:
                reasoning_value = delta.model_extra.get("reasoning")
            
            if reasoning_value:
                reasoning_delta = reasoning_value
                reasoning_content += reasoning_value

            if delta.content:
                text_delta = delta.content
                text_content += delta.content

            tool_call_deltas: list[core.ToolCallDelta] = []
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {"id": tc.id, "name": None, "args": ""}
                    if tc.id:
                        tool_calls[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls[idx]["args"] += tc.function.arguments
                            if tool_calls[idx]["id"]:
                                tool_call_deltas.append(
                                    core.ToolCallDelta(
                                        tool_call_id=tool_calls[idx]["id"],
                                        tool_name=tool_calls[idx]["name"] or "",
                                        args_delta=tc.function.arguments,
                                    )
                                )

            parts: list[core.Part] = []
            # Reasoning part comes first (like Anthropic's thinking blocks)
            if reasoning_content:
                parts.append(core.ReasoningPart(
                    reasoning=reasoning_content,
                    signature=None,  # OpenAI doesn't use signatures
                ))
            if text_content:
                parts.append(core.TextPart(text=text_content))
            for tc in tool_calls.values():
                if tc["id"]:
                    parts.append(
                        core.ToolCallPart(
                            tool_call_id=tc["id"],
                            tool_name=tc["name"] or "",
                            tool_args=tc["args"],
                        )
                    )

            is_done = choice.finish_reason is not None

            yield core.Message(
                role="assistant",
                parts=parts,
                id=message_id,
                is_done=is_done,
                text_delta=text_delta,
                reasoning_delta=reasoning_delta,
                tool_call_deltas=tool_call_deltas,
            )

            if is_done:
                return
