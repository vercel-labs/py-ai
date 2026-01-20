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
            "input_schema": tool.parameters,
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
                                "thinking": part.reasoning,
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

    @override
    async def stream(
        self,
        messages: list[core.messages.Message],
        tools: list[core.tools.Tool] | None = None,
    ) -> AsyncGenerator[core.messages.Message, None]:
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

        text_content = ""
        thinking_content = ""
        thinking_signature = ""
        tool_calls: dict[str, dict] = {}
        message_id = core.messages._gen_id()
        current_tool_id: str | None = None

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                text_delta = ""
                thinking_delta = ""
                tool_call_deltas: list[core.messages.ToolDelta] = []

                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool_id = block.id
                        tool_calls[block.id] = {"name": block.name, "args": ""}

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        text_delta = delta.text
                        text_content += delta.text
                    elif delta.type == "thinking_delta":
                        thinking_delta = delta.thinking
                        thinking_content += delta.thinking
                    elif delta.type == "signature_delta":
                        # Capture the signature for thinking blocks
                        thinking_signature += delta.signature
                    elif delta.type == "input_json_delta":
                        if current_tool_id and current_tool_id in tool_calls:
                            tool_calls[current_tool_id]["args"] += delta.partial_json
                            tool_call_deltas.append(
                                core.messages.ToolDelta(
                                    tool_call_id=current_tool_id,
                                    tool_name=tool_calls[current_tool_id]["name"],
                                    args_delta=delta.partial_json,
                                )
                            )

                elif event.type == "content_block_stop":
                    current_tool_id = None

                elif event.type == "message_stop":
                    final_parts: list[core.messages.Part] = []
                    if thinking_content:
                        final_parts.append(
                            core.messages.ReasoningPart(
                                reasoning=thinking_content,
                                signature=thinking_signature or None,
                            )
                        )
                    if text_content:
                        final_parts.append(core.messages.TextPart(text=text_content))
                    for tc_id, tc in tool_calls.items():
                        final_parts.append(
                            core.messages.ToolPart(
                                tool_call_id=tc_id,
                                tool_name=tc["name"],
                                tool_args=tc["args"],
                            )
                        )

                    yield core.messages.Message(
                        role="assistant",
                        parts=final_parts,
                        id=message_id,
                        is_done=True,
                        text_delta="",
                        reasoning_delta="",
                        tool_deltas=[],
                    )
                    return

                current_parts: list[core.messages.Part] = []
                if thinking_content:
                    current_parts.append(
                        core.messages.ReasoningPart(
                            reasoning=thinking_content,
                            signature=thinking_signature or None,
                        )
                    )
                if text_content:
                    current_parts.append(core.messages.TextPart(text=text_content))
                for tc_id, tc in tool_calls.items():
                    current_parts.append(
                        core.messages.ToolPart(
                            tool_call_id=tc_id,
                            tool_name=tc["name"],
                            tool_args=tc["args"],
                        )
                    )

                yield core.messages.Message(
                    role="assistant",
                    parts=current_parts,
                    id=message_id,
                    is_done=False,
                    text_delta=text_delta,
                    reasoning_delta=thinking_delta,
                    tool_deltas=tool_call_deltas,
                )
