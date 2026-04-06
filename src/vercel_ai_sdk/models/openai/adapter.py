"""OpenAI adapter — chat completions API.

Message/tool conversion and streaming via the official ``openai`` SDK.
The SDK client is constructed from :class:`Client` params on each call.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Any

import openai
import pydantic

from ...types import messages as messages_
from ...types import tools as tools_
from ..core import client as client_
from ..core import model as model_
from ..core.helpers import media as media_
from ..core.helpers import streaming as streaming_

# ---------------------------------------------------------------------------
# Message / tool conversion — internal types → OpenAI wire format
# ---------------------------------------------------------------------------


def _tools_to_openai(
    tools: Sequence[tools_.ToolLike],
) -> list[dict[str, Any]]:
    """Convert internal Tool objects to OpenAI tool schema format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.param_schema,
            },
        }
        for tool in tools
    ]


async def _file_part_to_openai(
    part: messages_.FilePart,
) -> dict[str, Any]:
    """Convert a :class:`FilePart` to an OpenAI content-array element.

    * ``image/*`` -> ``image_url`` (URL or ``data:`` URL)
    * ``audio/*`` -> ``input_audio`` (base-64 only; URLs auto-downloaded)
    * ``application/pdf`` -> ``file`` (base-64 only; URLs auto-downloaded)
    * ``text/*`` -> ``text`` (decoded to string)
    * anything else -> ``ValueError``
    """
    mt = part.media_type
    data = part.data

    if mt.startswith("image/"):
        media_type = "image/jpeg" if mt == "image/*" else mt
        url = media_.data_to_data_url(data, media_type)
        return {"type": "image_url", "image_url": {"url": url}}

    if mt.startswith("audio/"):
        if isinstance(data, str) and media_.is_downloadable_url(data):
            downloaded, _ = await media_.download(data)
            data = downloaded
        fmt = mt.split("/", 1)[1] if "/" in mt else mt
        b64 = media_.data_to_base64(data)
        return {
            "type": "input_audio",
            "input_audio": {"data": b64, "format": fmt},
        }

    if mt == "application/pdf":
        if isinstance(data, str) and media_.is_downloadable_url(data):
            downloaded, _ = await media_.download(data)
            data = downloaded
        data_url = media_.data_to_data_url(data, mt)
        filename = part.filename or "document.pdf"
        return {
            "type": "file",
            "file": {"filename": filename, "file_data": data_url},
        }

    if mt.startswith("text/"):
        if isinstance(data, bytes):
            text_content = data.decode("utf-8")
        elif media_.is_url(data):
            text_content = data
        else:
            import base64 as _b64

            text_content = _b64.b64decode(data).decode("utf-8")
        return {"type": "text", "text": text_content}

    raise ValueError(f"Unsupported media type for OpenAI: {mt}")


async def _messages_to_openai(
    messages: list[messages_.Message],
) -> list[dict[str, Any]]:
    """Convert internal messages to OpenAI API format.

    * ``tool_calls`` on assistant messages
    * tool results as separate ``role: "tool"`` messages
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        match msg.role:
            case "assistant":
                content = ""
                reasoning = ""
                tool_calls: list[dict[str, Any]] = []
                tool_results: list[dict[str, Any]] = []

                for part in msg.parts:
                    match part:
                        case messages_.ReasoningPart(text=text):
                            reasoning += text
                        case messages_.TextPart(text=text):
                            content += text
                        case messages_.ToolPart():
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
                            if part.status in ("result", "error"):
                                tool_results.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": part.tool_call_id,
                                        "content": str(part.result)
                                        if part.result is not None
                                        else "",
                                    }
                                )

                entry: dict[str, Any] = {"role": "assistant"}
                if content:
                    entry["content"] = content
                if reasoning:
                    entry["reasoning"] = reasoning
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                result.append(entry)
                result.extend(tool_results)

            case "system":
                content_text = "".join(
                    p.text for p in msg.parts if isinstance(p, messages_.TextPart)
                )
                result.append({"role": "system", "content": content_text})

            case "user":
                has_files = any(isinstance(p, messages_.FilePart) for p in msg.parts)
                if not has_files:
                    text = "".join(
                        p.text for p in msg.parts if isinstance(p, messages_.TextPart)
                    )
                    result.append({"role": "user", "content": text})
                else:
                    parts: list[dict[str, Any]] = []
                    for p in msg.parts:
                        match p:
                            case messages_.TextPart(text=text):
                                parts.append({"type": "text", "text": text})
                            case messages_.FilePart():
                                parts.append(await _file_part_to_openai(p))
                    result.append({"role": "user", "content": parts})
    return result


# ---------------------------------------------------------------------------
# SDK client factory
# ---------------------------------------------------------------------------


def _make_client(client: client_.Client) -> openai.AsyncOpenAI:
    """Construct an ``AsyncOpenAI`` from our generic ``Client``."""
    return openai.AsyncOpenAI(
        base_url=client.base_url,
        api_key=client.api_key or "",
    )


# ---------------------------------------------------------------------------
# Public adapter function
# ---------------------------------------------------------------------------


async def stream(
    client: client_.Client,
    model: model_.Model,
    messages: list[messages_.Message],
    *,
    tools: Sequence[tools_.ToolLike] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    thinking: bool = False,
    budget_tokens: int | None = None,
    reasoning_effort: str | None = None,
    **kwargs: Any,
) -> AsyncGenerator[messages_.Message]:
    """Stream an LLM response via the OpenAI chat completions API.

    Yields ``Message`` snapshots as the response streams in.

    Extra keyword arguments beyond the ``StreamFn`` protocol:

    * ``thinking`` — enable reasoning/thinking output.
    * ``budget_tokens`` — max tokens for reasoning (mutually exclusive
      with ``reasoning_effort``).
    * ``reasoning_effort`` — effort level: ``"none"``, ``"minimal"``,
      ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"``
      (mutually exclusive with ``budget_tokens``).
    """
    sdk_client = _make_client(client)
    openai_messages = await _messages_to_openai(messages)
    openai_tools = _tools_to_openai(tools) if tools else None

    api_kwargs: dict[str, Any] = {
        "model": model.id,
        "messages": openai_messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if openai_tools:
        api_kwargs["tools"] = openai_tools

    if output_type is not None:
        from openai.lib._pydantic import to_strict_json_schema

        api_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": output_type.__name__,
                "schema": to_strict_json_schema(output_type),
                "strict": True,
            },
        }

    # Enable reasoning/thinking via Vercel AI Gateway's unified format
    if thinking:
        reasoning_config: dict[str, Any] = {"enabled": True}
        if budget_tokens is not None:
            reasoning_config["max_tokens"] = budget_tokens
        elif reasoning_effort is not None:
            reasoning_config["effort"] = reasoning_effort
        api_kwargs["extra_body"] = {"reasoning": reasoning_config}

    handler = streaming_.StreamHandler()

    try:
        sdk_stream = await sdk_client.chat.completions.create(**api_kwargs)

        text_started = False
        reasoning_started = False
        tc_state: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: messages_.Usage | None = None

        async for chunk in sdk_stream:
            if chunk.usage is not None:
                raw = chunk.usage.model_dump(exclude_none=True)
                reasoning_tokens: int | None = None
                cache_read: int | None = None
                cd = getattr(
                    chunk.usage,
                    "completion_tokens_details",
                    None,
                )
                if cd:
                    reasoning_tokens = getattr(cd, "reasoning_tokens", None)
                pd = getattr(
                    chunk.usage,
                    "prompt_tokens_details",
                    None,
                )
                if pd:
                    cache_read = getattr(pd, "cached_tokens", None)
                usage = messages_.Usage(
                    input_tokens=chunk.usage.prompt_tokens or 0,
                    output_tokens=chunk.usage.completion_tokens or 0,
                    reasoning_tokens=reasoning_tokens,
                    cache_read_tokens=cache_read,
                    raw=raw,
                )

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            # Reasoning / thinking content
            reasoning_value = None
            if hasattr(delta, "reasoning") and delta.reasoning:
                reasoning_value = delta.reasoning
            elif hasattr(delta, "model_extra") and delta.model_extra:
                reasoning_value = delta.model_extra.get("reasoning")

            if reasoning_value:
                if not reasoning_started:
                    reasoning_started = True
                    yield handler.handle_event(
                        streaming_.ReasoningStart(block_id="reasoning")
                    )
                yield handler.handle_event(
                    streaming_.ReasoningDelta(
                        block_id="reasoning", delta=reasoning_value
                    )
                )

            if delta.content:
                if reasoning_started:
                    yield handler.handle_event(
                        streaming_.ReasoningEnd(block_id="reasoning")
                    )
                    reasoning_started = False

                if not text_started:
                    text_started = True
                    yield handler.handle_event(streaming_.TextStart(block_id="text"))
                yield handler.handle_event(
                    streaming_.TextDelta(block_id="text", delta=delta.content)
                )

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tc_state:
                        tc_state[idx] = {
                            "id": tc.id,
                            "name": None,
                            "started": False,
                        }
                    if tc.id:
                        tc_state[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tc_state[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tid = tc_state[idx]["id"]
                            tname = tc_state[idx]["name"] or ""

                            if not tc_state[idx]["started"] and tid:
                                tc_state[idx]["started"] = True
                                yield handler.handle_event(
                                    streaming_.ToolStart(
                                        tool_call_id=tid,
                                        tool_name=tname,
                                    )
                                )

                            if tid:
                                yield handler.handle_event(
                                    streaming_.ToolArgsDelta(
                                        tool_call_id=tid,
                                        delta=tc.function.arguments,
                                    )
                                )

            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason
                if reasoning_started:
                    yield handler.handle_event(
                        streaming_.ReasoningEnd(block_id="reasoning")
                    )
                if text_started:
                    yield handler.handle_event(streaming_.TextEnd(block_id="text"))
                for tc in tc_state.values():
                    if tc["started"] and tc["id"]:
                        yield handler.handle_event(
                            streaming_.ToolEnd(tool_call_id=tc["id"])
                        )

        yield handler.handle_event(
            streaming_.MessageDone(finish_reason=finish_reason, usage=usage)
        )
    finally:
        await sdk_client.close()
