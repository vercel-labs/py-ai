from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from typing import Any, override

import anthropic

from .. import core


def _tools_to_anthropic(tools: list[core.tools.Tool]) -> list[dict[str, Any]]:
    """Convert internal Tool objects to Anthropic tool schema format."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.schema,
        }
        for tool in tools
    ]


def _messages_to_anthropic(
    messages: list[core.messages.Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert internal messages to Anthropic API format.

    Returns (system_prompt, messages) tuple since Anthropic handles system differently.

    Handles the unified ToolPart model where tool calls and results are in the same
    assistant message. Converts back to Anthropic's expected format:
    - tool_use blocks in assistant messages
    - tool_result blocks in user messages (after the assistant message)
    """
    system_prompt: str | None = None
    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "system":
            system_prompt = "".join(
                p.text for p in msg.parts if isinstance(p, core.messages.TextPart)
            )
        elif msg.role == "assistant":
            content: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []

            for part in msg.parts:
                if isinstance(part, core.messages.ReasoningPart):
                    # Only include thinking blocks if we have the signature
                    # (required by Anthropic API for multi-turn conversations)
                    if part.signature:
                        content.append(
                            {
                                "type": "thinking",
                                "thinking": part.text,
                                "signature": part.signature,
                            }
                        )
                elif isinstance(part, core.messages.TextPart):
                    content.append({"type": "text", "text": part.text})
                elif isinstance(part, core.messages.ToolPart):
                    # tool_args is a JSON string, but Anthropic expects input as a dict
                    tool_input = json.loads(part.tool_args) if part.tool_args else {}
                    content.append(
                        {
                            "type": "tool_use",
                            "id": part.tool_call_id,
                            "name": part.tool_name,
                            "input": tool_input,
                        }
                    )
                    # If tool has a result, collect it for a separate user message
                    if part.status == "result" and part.result is not None:
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": part.tool_call_id,
                                "content": str(part.result),
                            }
                        )

            if content:
                result.append({"role": "assistant", "content": content})

            # Emit tool results as a separate user message (Anthropic API format)
            if tool_results:
                result.append({"role": "user", "content": tool_results})
        else:
            # User messages
            content_text = "".join(
                p.text for p in msg.parts if isinstance(p, core.messages.TextPart)
            )
            result.append({"role": "user", "content": content_text})

    return system_prompt, result


class AnthropicModel(core.runtime.LanguageModel):
    """Anthropic adapter with native extended thinking support."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool = False,
        budget_tokens: int = 10000,
    ) -> None:
        self._model = model
        self._thinking = thinking
        self._budget_tokens = budget_tokens
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
        self._client = anthropic.AsyncAnthropic(base_url=base_url, api_key=resolved_key)

    async def stream_events(
        self,
        messages: list[core.messages.Message],
        tools: list[core.tools.Tool] | None = None,
    ) -> AsyncGenerator[core.llm.StreamEvent, None]:
        """Yield raw stream events from Anthropic API."""
        system_prompt, anthropic_messages = _messages_to_anthropic(messages)
        anthropic_tools = _tools_to_anthropic(tools) if tools else None

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": anthropic_messages,
            "max_tokens": 8192,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        if self._thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._budget_tokens,
            }

        # Track block types by index to know what End event to emit
        block_types: dict[int, str] = {}  # index -> "text" | "thinking" | "tool_use"
        tool_ids: dict[int, str] = {}  # index -> tool_call_id
        signature_buffer: dict[int, str] = {}  # index -> accumulated signature

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    idx = event.index
                    block_types[idx] = block.type

                    if block.type == "text":
                        yield core.llm.TextStart(block_id=str(idx))
                    elif block.type == "thinking":
                        yield core.llm.ReasoningStart(block_id=str(idx))
                    elif block.type == "tool_use":
                        tool_ids[idx] = block.id
                        yield core.llm.ToolStart(
                            tool_call_id=block.id, tool_name=block.name
                        )

                elif event.type == "content_block_delta":
                    delta = event.delta
                    idx = event.index

                    if delta.type == "text_delta":
                        yield core.llm.TextDelta(block_id=str(idx), delta=delta.text)
                    elif delta.type == "thinking_delta":
                        yield core.llm.ReasoningDelta(
                            block_id=str(idx), delta=delta.thinking
                        )
                    elif delta.type == "signature_delta":
                        # Accumulate signature for ReasoningEnd
                        signature_buffer[idx] = (
                            signature_buffer.get(idx, "") + delta.signature
                        )
                    elif delta.type == "input_json_delta":
                        tool_id = tool_ids.get(idx)
                        if tool_id:
                            yield core.llm.ToolArgsDelta(
                                tool_call_id=tool_id, delta=delta.partial_json
                            )

                elif event.type == "content_block_stop":
                    idx = event.index
                    block_type = block_types.get(idx)

                    if block_type == "text":
                        yield core.llm.TextEnd(block_id=str(idx))
                    elif block_type == "thinking":
                        yield core.llm.ReasoningEnd(
                            block_id=str(idx),
                            signature=signature_buffer.get(idx),
                        )
                    elif block_type == "tool_use":
                        tool_id = tool_ids.get(idx)
                        if tool_id:
                            yield core.llm.ToolEnd(tool_call_id=tool_id)

                elif event.type == "message_stop":
                    yield core.llm.MessageDone()

    @override
    async def stream(
        self,
        messages: list[core.messages.Message],
        tools: list[core.tools.Tool] | None = None,
    ) -> AsyncGenerator[core.messages.Message, None]:
        """Stream Messages (uses StreamProcessor internally)."""
        handler = core.llm.StreamHandler()
        async for event in self.stream_events(messages, tools):
            yield handler.handle_event(event)
