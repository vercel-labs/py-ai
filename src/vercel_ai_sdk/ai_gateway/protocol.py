"""Vercel AI Gateway v3 protocol: serialization and deserialization.

Converts between the Python SDK's internal ``Message`` / ``StreamEvent``
types and the LanguageModelV3 wire format used by the gateway at
``/v3/ai/language-model``.

Wire format reference (from ``@ai-sdk/provider``):

* **Request body** -- ``LanguageModelV3CallOptions`` (prompt + tools +
  provider options, sent as JSON).
* **Stream response** -- Server-Sent Events where each ``data:`` line is
  a JSON ``LanguageModelV3StreamPart`` (discriminated on ``type``).
* **Non-stream response** -- JSON ``LanguageModelV3GenerateResult``.
"""

import json
from collections.abc import Sequence
from typing import Any

from .. import core

# ---------------------------------------------------------------------------
# Internal messages  ->  v3 prompt format  (outgoing request body)
# ---------------------------------------------------------------------------


async def _file_part_to_v3(part: core.messages.FilePart) -> dict[str, Any]:
    """Convert an internal :class:`FilePart` to a v3 ``file`` content part.

    Binary data is converted to a ``data:`` URL for JSON transport (matching
    the JS SDK gateway's ``maybeEncodeFileParts``).  HTTP(S) URLs are
    downloaded and converted to ``data:`` URLs because the gateway wire
    format does not accept raw HTTP URLs for file content.
    """
    data = part.data
    if isinstance(data, str) and core.media.data.is_downloadable_url(data):
        downloaded, _ = await core.media.download.download(data)
        data = downloaded

    entry: dict[str, Any] = {
        "type": "file",
        "mediaType": part.media_type,
        "data": core.media.data.data_to_data_url(data, part.media_type),
    }
    if part.filename is not None:
        entry["filename"] = part.filename
    return entry


async def messages_to_v3_prompt(
    messages: list[core.messages.Message],
) -> list[dict[str, Any]]:
    """Convert internal ``Message`` list to ``LanguageModelV3Prompt``.

    The v3 prompt format is an array of messages, each with a ``role`` and
    typed ``content`` parts::

        [
          {"role": "system", "content": "You are helpful."},
          {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
          {"role": "assistant", "content": [
            {"type": "text", "text": "Hello!"},
            {"type": "reasoning", "text": "..."},
            {"type": "tool-call", "toolCallId": "tc-1", ...},
          ]},
          {"role": "tool", "content": [
            {"type": "tool-result", "toolCallId": "tc-1", ...},
          ]},
        ]
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        match msg.role:
            case "system":
                text = "".join(
                    p.text for p in msg.parts if isinstance(p, core.messages.TextPart)
                )
                result.append({"role": "system", "content": text})

            case "user":
                content: list[dict[str, Any]] = []
                for p in msg.parts:
                    if isinstance(p, core.messages.TextPart):
                        content.append({"type": "text", "text": p.text})
                    elif isinstance(p, core.messages.FilePart):
                        content.append(await _file_part_to_v3(p))
                result.append({"role": "user", "content": content})

            case "assistant":
                assistant_content: list[dict[str, Any]] = []
                tool_results: list[dict[str, Any]] = []

                for part in msg.parts:
                    match part:
                        case core.messages.ReasoningPart(text=text):
                            assistant_content.append(
                                {"type": "reasoning", "text": text}
                            )

                        case core.messages.TextPart(text=text):
                            assistant_content.append({"type": "text", "text": text})

                        case core.messages.ToolPart() as tp:
                            tool_input: Any = (
                                json.loads(tp.tool_args) if tp.tool_args else {}
                            )
                            assistant_content.append(
                                {
                                    "type": "tool-call",
                                    "toolCallId": tp.tool_call_id,
                                    "toolName": tp.tool_name,
                                    "input": tool_input,
                                }
                            )
                            if tp.status in ("result", "error"):
                                output = (
                                    {
                                        "type": "error-text",
                                        "value": (
                                            str(tp.result)
                                            if tp.result is not None
                                            else ""
                                        ),
                                    }
                                    if tp.status == "error"
                                    else {
                                        "type": "json",
                                        "value": tp.result,
                                    }
                                )
                                tool_results.append(
                                    {
                                        "type": "tool-result",
                                        "toolCallId": tp.tool_call_id,
                                        "toolName": tp.tool_name,
                                        "output": output,
                                    }
                                )

                result.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                    }
                )
                if tool_results:
                    result.append(
                        {
                            "role": "tool",
                            "content": tool_results,
                        }
                    )

    return result


# ---------------------------------------------------------------------------
# Request body serialization
# ---------------------------------------------------------------------------


async def build_request_body(
    messages: list[core.messages.Message],
    tools: Sequence[core.tools.ToolLike] | None = None,
    output_type: type[Any] | None = None,
    provider_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full ``LanguageModelV3CallOptions`` request body."""
    body: dict[str, Any] = {
        "prompt": await messages_to_v3_prompt(messages),
    }
    if tools:
        body["tools"] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.param_schema,
            }
            for tool in tools
        ]
    if output_type is not None:
        import pydantic

        if issubclass(output_type, pydantic.BaseModel):
            body["responseFormat"] = {
                "type": "json",
                "schema": output_type.model_json_schema(),
                "name": output_type.__name__,
            }
    if provider_options:
        body["providerOptions"] = provider_options
    return body


# ---------------------------------------------------------------------------
# v3 stream parts  ->  internal StreamEvent  (incoming SSE response)
# ---------------------------------------------------------------------------


def parse_stream_part(
    data: dict[str, Any],
) -> list[core.llm.StreamEvent]:
    """Convert a ``LanguageModelV3StreamPart`` to internal events.

    Most parts map 1:1.  A ``tool-call`` part (complete, non-streaming)
    expands to Start + ArgsDelta + End.  Lifecycle events
    (``stream-start``, ``response-metadata``, ``raw``) are silently
    dropped.
    """
    match data.get("type", ""):
        case "text-start":
            return [
                core.llm.TextStart(
                    block_id=data.get("id", "text"),
                )
            ]

        case "text-delta":
            return [
                core.llm.TextDelta(
                    block_id=data.get("id", "text"),
                    delta=data.get("textDelta", data.get("delta", "")),
                )
            ]

        case "text-end":
            return [
                core.llm.TextEnd(
                    block_id=data.get("id", "text"),
                )
            ]

        case "reasoning-start":
            return [
                core.llm.ReasoningStart(
                    block_id=data.get("id", "reasoning"),
                )
            ]

        case "reasoning-delta":
            return [
                core.llm.ReasoningDelta(
                    block_id=data.get("id", "reasoning"),
                    delta=data.get("delta", ""),
                )
            ]

        case "reasoning-end":
            return [
                core.llm.ReasoningEnd(
                    block_id=data.get("id", "reasoning"),
                )
            ]

        case "tool-input-start":
            return [
                core.llm.ToolStart(
                    tool_call_id=data.get("id", ""),
                    tool_name=data.get("toolName", ""),
                )
            ]

        case "tool-input-delta":
            return [
                core.llm.ToolArgsDelta(
                    tool_call_id=data.get("id", ""),
                    delta=data.get("delta", ""),
                )
            ]

        case "tool-input-end":
            return [
                core.llm.ToolEnd(
                    tool_call_id=data.get("id", ""),
                )
            ]

        case "tool-call":
            return _expand_tool_call(data)

        case "file":
            return [
                core.llm.FileEvent(
                    block_id=data.get("id", f"file-{len(data)}"),
                    media_type=data.get("mediaType", "application/octet-stream"),
                    data=data.get("data", ""),
                )
            ]

        case "finish":
            return [_parse_finish(data)]

        case _:
            return []


# ---------------------------------------------------------------------------
# Non-streaming response  ->  internal StreamEvents
# ---------------------------------------------------------------------------


def parse_generate_result(
    data: dict[str, Any],
) -> list[core.llm.StreamEvent]:
    """Convert a ``LanguageModelV3GenerateResult`` into events.

    Synthesises Start/Delta/End events from the content, then a final
    ``MessageDone``.
    """
    events: list[core.llm.StreamEvent] = []

    def _expand_content_item(item: dict[str, Any]) -> None:
        match item.get("type", ""):
            case "text":
                bid = item.get("id", "text")
                text = item.get("text", "")
                events.append(core.llm.TextStart(block_id=bid))
                events.append(core.llm.TextDelta(block_id=bid, delta=text))
                events.append(core.llm.TextEnd(block_id=bid))

            case "reasoning":
                bid = item.get("id", "reasoning")
                text = item.get("text", "")
                events.append(core.llm.ReasoningStart(block_id=bid))
                events.append(core.llm.ReasoningDelta(block_id=bid, delta=text))
                events.append(core.llm.ReasoningEnd(block_id=bid))

            case "tool-call":
                events.extend(_expand_tool_call(item))

            case "file":
                events.append(
                    core.llm.FileEvent(
                        block_id=item.get("id", f"file-{len(events)}"),
                        media_type=item.get("mediaType", "application/octet-stream"),
                        data=item.get("data", ""),
                    )
                )

    match data.get("content"):
        case list() as items:
            for item in items:
                _expand_content_item(item)
        case dict() as item:
            _expand_content_item(item)

    events.append(_parse_finish(data))
    return events


# ---------------------------------------------------------------------------
# Shared helpers (called from multiple sites)
# ---------------------------------------------------------------------------


def _expand_tool_call(
    data: dict[str, Any],
) -> list[core.llm.StreamEvent]:
    """Expand a complete ``tool-call`` part into three events."""
    tc_id = data.get("toolCallId", "")
    tool_name = data.get("toolName", "")
    tool_input = data.get("input", "")
    args_str = tool_input if isinstance(tool_input, str) else json.dumps(tool_input)
    return [
        core.llm.ToolStart(tool_call_id=tc_id, tool_name=tool_name),
        core.llm.ToolArgsDelta(tool_call_id=tc_id, delta=args_str),
        core.llm.ToolEnd(tool_call_id=tc_id),
    ]


def _parse_finish(data: dict[str, Any]) -> core.llm.MessageDone:
    """Parse a ``finish`` stream part into a ``MessageDone`` event."""
    usage_data = data.get("usage")
    usage = _parse_usage(usage_data) if usage_data else None

    match data.get("finishReason"):
        case dict() as d:
            finish_reason = d.get("unified", "stop")
        case str() as s:
            finish_reason = s
        case _:
            finish_reason = "stop"

    return core.llm.MessageDone(finish_reason=finish_reason, usage=usage)


def _parse_usage(data: Any) -> core.messages.Usage:
    """Parse a v3 ``LanguageModelV3Usage`` into an internal ``Usage``.

    Supports both the v3 nested format::

        {"inputTokens": {"total": 10, ...}, "outputTokens": {...}}

    and the flat OpenAI-style format::

        {"prompt_tokens": 10, "completion_tokens": 20}
    """
    if not isinstance(data, dict):
        return core.messages.Usage()

    input_tokens_obj = data.get("inputTokens")
    output_tokens_obj = data.get("outputTokens")

    if isinstance(input_tokens_obj, dict) or isinstance(output_tokens_obj, dict):
        inp = input_tokens_obj if isinstance(input_tokens_obj, dict) else {}
        out = output_tokens_obj if isinstance(output_tokens_obj, dict) else {}
        return core.messages.Usage(
            input_tokens=inp.get("total") or 0,
            output_tokens=out.get("total") or 0,
            reasoning_tokens=out.get("reasoning"),
            cache_read_tokens=inp.get("cacheRead"),
            cache_write_tokens=inp.get("cacheWrite"),
            raw=data,
        )

    return core.messages.Usage(
        input_tokens=(data.get("prompt_tokens") or data.get("inputTokens") or 0),
        output_tokens=(data.get("completion_tokens") or data.get("outputTokens") or 0),
        raw=data,
    )
