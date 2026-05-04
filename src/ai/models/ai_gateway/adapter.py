"""AI Gateway v3 adapter.

Converts internal messages to AI Gateway wire payloads and maps gateway
responses back to public event/message types."""

import base64
import json
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import httpx
import pydantic

from ... import types
from ...types import tools as tools_
from .. import core
from ..anthropic.params import AnthropicParams
from ..anthropic.tools import _AnthropicBuiltin
from ..openai.params import OpenAIChatParams, OpenAIResponsesParams
from ..openai.tools import _OpenAIBuiltin
from . import errors, sdk
from .params import GATEWAY_STREAM_PARAMS_TYPES, GatewayParams
from .tools import _GatewayBuiltin

# ---------------------------------------------------------------------------
# Shared request helpers
# ---------------------------------------------------------------------------


def _extract_prompt(messages: list[types.Message]) -> str:
    """Concatenate all text from user/system messages into one prompt."""
    parts: list[str] = []
    for msg in messages:
        if msg.role in ("user", "system"):
            for p in msg.parts:
                if isinstance(p, types.TextPart):
                    parts.append(p.text)
    return " ".join(parts)


def _extract_input_files(
    messages: list[types.Message],
) -> list[types.FilePart]:
    """Collect all file parts from user messages."""
    files_: list[types.FilePart] = []
    for msg in messages:
        if msg.role == "user":
            for p in msg.parts:
                if isinstance(p, types.FilePart):
                    files_.append(p)
    return files_


def _file_part_to_wire(part: types.FilePart) -> dict[str, Any]:
    """Convert a :class:`FilePart` to the gateway wire format for input files."""
    data = part.data
    if isinstance(data, str) and types.media.is_url(data):
        return {"type": "url", "url": data}
    if isinstance(data, bytes):
        b64 = base64.b64encode(data).decode("ascii")
    elif isinstance(data, str):
        b64 = data
    else:
        b64 = str(data)
    return {"type": "file", "data": b64, "mediaType": part.media_type}


# ---------------------------------------------------------------------------
# Streaming request building — Message list → v3 prompt
# ---------------------------------------------------------------------------


async def _file_part_to_v3(part: types.FilePart) -> dict[str, Any]:
    """Convert a :class:`FilePart` to a v3 ``file`` content part."""
    data = part.data
    if isinstance(data, str) and types.media.is_downloadable_url(data):
        downloaded, _ = await core.helpers.files.download(data)
        data = downloaded

    entry: dict[str, Any] = {
        "type": "file",
        "mediaType": part.media_type,
        "data": types.media.data_to_data_url(data, part.media_type),
    }
    if part.filename is not None:
        entry["filename"] = part.filename
    return entry


async def _messages_to_prompt(
    messages: list[types.Message],
) -> list[dict[str, Any]]:
    """Convert ``Message`` list to the v3 prompt wire format."""
    result: list[dict[str, Any]] = []

    for msg in messages:
        match msg.role:
            case "system":
                text = "".join(
                    p.text for p in msg.parts if isinstance(p, types.TextPart)
                )
                result.append({"role": "system", "content": text})

            case "user":
                content: list[dict[str, Any]] = []
                for p in msg.parts:
                    if isinstance(p, types.TextPart):
                        content.append({"type": "text", "text": p.text})
                    elif isinstance(p, types.FilePart):
                        content.append(await _file_part_to_v3(p))
                result.append({"role": "user", "content": content})

            case "assistant":
                assistant_content: list[dict[str, Any]] = []
                for part in msg.parts:
                    match part:
                        case types.ReasoningPart(text=text):
                            assistant_content.append(
                                {"type": "reasoning", "text": text}
                            )
                        case types.TextPart(text=text):
                            assistant_content.append({"type": "text", "text": text})
                        case types.ToolCallPart() as tp:
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
                        case types.BuiltinToolCallPart() as btp:
                            btp_input: Any = (
                                json.loads(btp.tool_args) if btp.tool_args else {}
                            )
                            assistant_content.append(
                                {
                                    "type": "tool-call",
                                    "toolCallId": btp.tool_call_id,
                                    "toolName": btp.tool_name,
                                    "input": btp_input,
                                    "providerExecuted": True,
                                }
                            )
                        case types.BuiltinToolReturnPart() as brp:
                            assistant_content.append(
                                {
                                    "type": "tool-result",
                                    "toolCallId": brp.tool_call_id,
                                    "toolName": brp.tool_name,
                                    "output": {
                                        "type": "json",
                                        "value": brp.result,
                                    },
                                    "providerExecuted": True,
                                }
                            )
                result.append({"role": "assistant", "content": assistant_content})

            case "tool":
                tool_results: list[dict[str, Any]] = []
                for part in msg.parts:
                    if isinstance(part, types.ToolResultPart):
                        output = (
                            {
                                "type": "error-text",
                                "value": (
                                    str(part.result) if part.result is not None else ""
                                ),
                            }
                            if part.is_error
                            else {
                                "type": "json",
                                "value": part.result,
                            }
                        )
                        tool_results.append(
                            {
                                "type": "tool-result",
                                "toolCallId": part.tool_call_id,
                                "toolName": part.tool_name,
                                "output": output,
                            }
                        )
                if tool_results:
                    result.append({"role": "tool", "content": tool_results})

    return result


def _tool_to_v3(tool: types.ToolLike) -> dict[str, Any]:
    """Convert a tool-like object to the v3 wire format.

    Built-in tools use ``model_dump(by_alias=True)`` to emit camelCase keys,
    which is what the gateway's JS-derived wire schema expects.  The
    ``alias_generator=to_camel`` on the ``BuiltinTool`` base class handles
    the conversion, including for nested config models.
    """
    if isinstance(tool, _AnthropicBuiltin):
        return {
            "type": "provider",
            "id": f"anthropic.{tool.wire_type}",
            "name": tool.wire_name,
            "args": tool.model_dump(mode="json", by_alias=True, exclude_none=True),
        }
    if isinstance(tool, _OpenAIBuiltin):
        return {
            "type": "provider",
            "id": f"openai.{tool.wire_type}",
            "name": tool.wire_type,
            "args": tool.model_dump(mode="json", by_alias=True, exclude_none=True),
        }
    if isinstance(tool, _GatewayBuiltin):
        return {
            "type": "provider",
            "id": tool.wire_id,
            "name": tool.wire_name,
            "args": tool.model_dump(mode="json", by_alias=True, exclude_none=True),
        }
    if isinstance(tool, tools_.BuiltinTool):
        raise TypeError(
            f"AI Gateway does not support built-in tool "
            f"{type(tool).__name__}; use anthropic.tools.*, "
            f"openai.tools.*, or ai_gateway.tools.* helpers."
        )
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.param_schema,
    }


async def _build_request_body(
    messages: list[types.Message],
    tools: Sequence[types.proto.ToolLike] | None = None,
    output_type: type[Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the ``LanguageModelV3CallOptions`` request body."""
    stream_params = _normalize_gateway_params(kwargs.get("params"))
    body: dict[str, Any] = {
        "prompt": await _messages_to_prompt(messages),
    }
    if tools:
        body["tools"] = [_tool_to_v3(tool) for tool in tools]
    if output_type is not None and issubclass(output_type, pydantic.BaseModel):
        body["responseFormat"] = {
            "type": "json",
            "schema": output_type.model_json_schema(),
            "name": output_type.__name__,
        }
    provider_options = _merge_provider_options(
        kwargs.get("provider_options"),
        _provider_options_from_params(stream_params),
    )
    if provider_options:
        body["providerOptions"] = provider_options
    extra_body = _gateway_body_from_params(stream_params)
    if extra_body:
        body.update(extra_body)
    return body


def _provider_options_from_params(
    stream_params: Sequence[pydantic.BaseModel],
) -> dict[str, Any] | None:
    """Build Gateway providerOptions from typed params and provider raw body."""
    provider_options: dict[str, Any] = {}
    for param in stream_params:
        key = _gateway_provider_options_key(param)
        payload = _provider_options_payload(param)
        if not payload:
            continue
        if key in provider_options:
            raise ValueError(f"duplicate provider params for {key!r}")
        provider_options[key] = payload
    return provider_options or None


def _normalize_gateway_params(value: Any) -> list[pydantic.BaseModel]:
    """Accept one params object or a sequence and enforce Gateway param types."""
    if value is None:
        return []
    if isinstance(value, pydantic.BaseModel):
        raw_items: list[Any] = [value]
    else:
        raw_items = list(value)

    result: list[pydantic.BaseModel] = []
    for item in raw_items:
        if not isinstance(item, GATEWAY_STREAM_PARAMS_TYPES):
            raise TypeError(
                "ai-gateway streams accept GatewayParams, OpenAIChatParams, "
                "OpenAIResponsesParams, or AnthropicParams"
            )
        result.append(item)
    return result


def _gateway_provider_options_key(param: pydantic.BaseModel) -> str:
    """Return the providerOptions bucket name for one Gateway param wrapper."""
    if isinstance(param, GatewayParams):
        return "gateway"
    if isinstance(param, OpenAIChatParams | OpenAIResponsesParams):
        return "openai"
    if isinstance(param, AnthropicParams):
        return "anthropic"
    raise TypeError(f"unsupported ai-gateway params type {type(param).__name__}")


def _provider_options_payload(param: pydantic.BaseModel) -> dict[str, Any]:
    """Dump typed providerOptions and merge forwarded provider raw body fields."""
    payload = param.model_dump(by_alias=True, exclude_none=True)
    if not isinstance(param, GatewayParams):
        extra_body = getattr(param, "extra_body", None)
        if extra_body:
            payload.update(extra_body)
    return payload


def _gateway_body_from_params(
    stream_params: Sequence[pydantic.BaseModel],
) -> dict[str, Any] | None:
    """Collect raw top-level Gateway body fields from GatewayParams only."""
    extra_body: dict[str, Any] = {}
    for param in stream_params:
        if isinstance(param, GatewayParams) and param.extra_body:
            extra_body.update(param.extra_body)
    return extra_body or None


def _extra_headers_from_params(
    stream_params: Sequence[pydantic.BaseModel],
) -> dict[str, str] | None:
    """Collect raw per-request HTTP headers from any Gateway params object."""
    headers: dict[str, str] = {}
    for param in stream_params:
        extra_headers = getattr(param, "extra_headers", None)
        if extra_headers:
            headers.update(extra_headers)
    return headers or None


def _merge_provider_options(
    existing: dict[str, Any] | None,
    generated: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Combine legacy provider_options with typed params without overwriting."""
    if not existing:
        return generated
    if not generated:
        return dict(existing)

    result = dict(existing)
    for key, value in generated.items():
        if key in result:
            raise ValueError(f"duplicate provider params for {key!r}")
        result[key] = value
    return result


# ---------------------------------------------------------------------------
# Streaming response parsing — v3 stream parts → public Event
# ---------------------------------------------------------------------------


def _is_provider_executed(data: dict[str, Any]) -> bool:
    """Whether a v3 tool part marks itself as provider-executed."""
    return bool(data.get("providerExecuted") or data.get("provider_executed"))


def _expand_tool_call(
    data: dict[str, Any],
    streamed_tool_ids: set[str],
    provider_executed_ids: set[str] | None = None,
) -> list[types.events.Event]:
    """Expand a complete ``tool-call`` part into Start + Delta + End.

    Returns empty when the tool was already streamed via ``tool-input-*``.
    """
    tc_id = data.get("toolCallId", "")
    if tc_id in streamed_tool_ids:
        return []
    if provider_executed_ids is None:
        provider_executed_ids = set()
    tool_name = data.get("toolName", "")
    tool_input = data.get("input", "")
    args_str = tool_input if isinstance(tool_input, str) else json.dumps(tool_input)
    if _is_provider_executed(data) or tc_id in provider_executed_ids:
        provider_executed_ids.add(tc_id)
        return [
            types.events.BuiltinToolStart(tool_call_id=tc_id, tool_name=tool_name),
            types.events.BuiltinToolDelta(tool_call_id=tc_id, chunk=args_str),
            types.events.BuiltinToolEnd(
                tool_call_id=tc_id,
                tool_call=types.messages.BuiltinToolCallPart(
                    tool_call_id=tc_id,
                    tool_name=tool_name,
                    tool_args=args_str,
                ),
            ),
        ]
    return [
        types.events.ToolStart(tool_call_id=tc_id, tool_name=tool_name),
        types.events.ToolDelta(tool_call_id=tc_id, chunk=args_str),
        types.events.ToolEnd(
            tool_call_id=tc_id, tool_call=types.messages.DUMMY_TOOL_CALL
        ),
    ]


def _parse_usage(data: Any) -> types.Usage:
    """Parse v3 usage data into an internal ``Usage``."""
    if not isinstance(data, dict):
        return types.Usage()

    input_tokens_obj = data.get("inputTokens")
    output_tokens_obj = data.get("outputTokens")

    if isinstance(input_tokens_obj, dict) or isinstance(output_tokens_obj, dict):
        inp = input_tokens_obj if isinstance(input_tokens_obj, dict) else {}
        out = output_tokens_obj if isinstance(output_tokens_obj, dict) else {}
        return types.Usage(
            input_tokens=inp.get("total") or 0,
            output_tokens=out.get("total") or 0,
            reasoning_tokens=out.get("reasoning"),
            cache_read_tokens=inp.get("cacheRead"),
            cache_write_tokens=inp.get("cacheWrite"),
            raw=data,
        )

    return types.Usage(
        input_tokens=data.get("prompt_tokens") or data.get("inputTokens") or 0,
        output_tokens=(data.get("completion_tokens") or data.get("outputTokens") or 0),
        raw=data,
    )


def _parse_stream_part(
    data: dict[str, Any],
    streamed_tool_ids: set[str],
    provider_executed_ids: set[str] | None = None,
) -> list[types.events.Event]:
    """Convert a ``LanguageModelV3StreamPart`` to public events."""
    if provider_executed_ids is None:
        provider_executed_ids = set()
    match data.get("type", ""):
        case "text-start":
            return [types.events.TextStart(block_id=data.get("id", "text"))]

        case "text-delta":
            return [
                types.events.TextDelta(
                    block_id=data.get("id", "text"),
                    chunk=data.get("textDelta", data.get("delta", "")),
                )
            ]

        case "text-end":
            return [types.events.TextEnd(block_id=data.get("id", "text"))]

        case "reasoning-start":
            return [types.events.ReasoningStart(block_id=data.get("id", "reasoning"))]

        case "reasoning-delta":
            return [
                types.events.ReasoningDelta(
                    block_id=data.get("id", "reasoning"),
                    chunk=data.get("delta", ""),
                )
            ]

        case "reasoning-end":
            return [types.events.ReasoningEnd(block_id=data.get("id", "reasoning"))]

        case "tool-input-start":
            tcid = data.get("id", "")
            streamed_tool_ids.add(tcid)
            if _is_provider_executed(data):
                provider_executed_ids.add(tcid)
                return [
                    types.events.BuiltinToolStart(
                        tool_call_id=tcid,
                        tool_name=data.get("toolName", ""),
                    )
                ]
            return [
                types.events.ToolStart(
                    tool_call_id=tcid,
                    tool_name=data.get("toolName", ""),
                )
            ]

        case "tool-input-delta":
            tcid = data.get("id", "")
            if tcid in provider_executed_ids:
                return [
                    types.events.BuiltinToolDelta(
                        tool_call_id=tcid,
                        chunk=data.get("delta", ""),
                    )
                ]
            return [
                types.events.ToolDelta(
                    tool_call_id=tcid,
                    chunk=data.get("delta", ""),
                )
            ]

        case "tool-input-end":
            tcid = data.get("id", "")
            if tcid in provider_executed_ids:
                return [
                    types.events.BuiltinToolEnd(
                        tool_call_id=tcid,
                        tool_call=types.messages.BuiltinToolCallPart(
                            tool_call_id=tcid,
                            tool_name="",
                        ),
                    )
                ]
            return [
                types.events.ToolEnd(
                    tool_call_id=tcid,
                    tool_call=types.messages.DUMMY_TOOL_CALL,
                )
            ]

        case "tool-call":
            return _expand_tool_call(data, streamed_tool_ids, provider_executed_ids)

        case "tool-result":
            tcid = data.get("toolCallId", "")
            tool_name = data.get("toolName", "")
            output = data.get("output") or data.get("result")
            is_error = bool(data.get("isError"))
            if _is_provider_executed(data) or tcid in provider_executed_ids:
                provider_executed_ids.add(tcid)
                return [
                    types.events.BuiltinToolResult(
                        tool_call_id=tcid,
                        result=types.messages.BuiltinToolReturnPart(
                            tool_call_id=tcid,
                            tool_name=tool_name,
                            result=output,
                            is_error=is_error,
                        ),
                    )
                ]
            return []

        case "file":
            return [
                types.events.FileEvent(
                    block_id=data.get("id", ""),
                    media_type=data.get("mediaType", "application/octet-stream"),
                    data=data.get("data", ""),
                )
            ]

        case "finish":
            usage_data = data.get("usage")
            usage = _parse_usage(usage_data) if usage_data else None
            return [types.events.StreamEnd(usage=usage)]

        case _:
            return []


async def stream(
    client: core.client.Client,
    model: core.model.Model[Any],
    messages: list[types.Message],
    *,
    tools: Sequence[types.proto.ToolLike] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    **kwargs: Any,
) -> AsyncGenerator[types.events.Event]:
    """Stream an LLM response through the AI Gateway v3 protocol."""
    stream_params = _normalize_gateway_params(kwargs.get("params"))
    body = await _build_request_body(
        messages,
        tools=tools,
        output_type=output_type,
        **{**kwargs, "params": stream_params},
    )
    extra_headers = _extra_headers_from_params(stream_params)
    gateway = sdk.GatewayClient(client, model)

    try:
        async with gateway.stream(
            "language-model",
            body,
            model_type="language",
            streaming=True,
            headers=extra_headers,
        ) as response:
            yield types.events.StreamStart()
            streamed_tool_ids: set[str] = set()
            provider_executed_ids: set[str] = set()
            async for data in gateway.iter_sse(response):
                for event in _parse_stream_part(
                    data, streamed_tool_ids, provider_executed_ids
                ):
                    yield event
    except errors.GatewayError:
        raise
    except httpx.TimeoutException as exc:
        raise errors.GatewayTimeoutError(cause=exc) from exc
    except Exception as exc:
        raise errors.GatewayResponseError(
            message=f"Unexpected error during streaming: {exc}",
            cause=exc,
        ) from exc


# ---------------------------------------------------------------------------
# Media generation
# ---------------------------------------------------------------------------


async def _generate_image(
    client: core.client.Client,
    model: core.model.Model[Any],
    messages: list[types.Message],
    params: core.ImageParams,
) -> types.Message:
    """Hit ``/image-model`` and return a Message with FileParts."""
    prompt = _extract_prompt(messages)
    input_files = _extract_input_files(messages)

    body: dict[str, Any] = {
        "prompt": prompt,
        **params.model_dump(by_alias=True, exclude_none=True),
    }
    if input_files:
        body["files"] = [_file_part_to_wire(f) for f in input_files]

    gateway = sdk.GatewayClient(client, model)
    response = await gateway.post_json("image-model", body, model_type="image")

    data = response.json()
    raw_images: list[str] = data.get("images", [])
    usage_data = data.get("usage")
    usage = None
    if usage_data:
        usage = types.Usage(
            input_tokens=usage_data.get("inputTokens") or 0,
            output_tokens=usage_data.get("outputTokens") or 0,
        )

    parts: list[types.Part] = []
    for img_b64 in raw_images:
        media_type = types.media.detect_image_media_type(img_b64) or "image/png"
        parts.append(types.FilePart(data=img_b64, media_type=media_type))

    return types.Message(role="assistant", parts=parts, usage=usage)


async def _generate_video(
    client: core.client.Client,
    model: core.model.Model[Any],
    messages: list[types.Message],
    params: core.VideoParams,
) -> types.Message:
    """Hit ``/video-model`` (SSE) and return a Message with FileParts."""
    prompt = _extract_prompt(messages)
    input_files = _extract_input_files(messages)

    body: dict[str, Any] = {
        "prompt": prompt,
        **params.model_dump(by_alias=True, exclude_none=True),
    }
    if input_files:
        body["image"] = _file_part_to_wire(input_files[0])

    gateway = sdk.GatewayClient(client, model)

    async with gateway.stream(
        "video-model",
        body,
        model_type="video",
        accept="text/event-stream",
        timeout=httpx.Timeout(timeout=600.0, connect=10.0),
    ) as response:
        event_data: dict[str, Any] = {}
        async for parsed in gateway.iter_sse(response):
            event_data = parsed
            break

    if not event_data:
        raise errors.GatewayResponseError(
            "SSE stream ended without any data events",
        )

    if event_data.get("type") == "error":
        raise errors.GatewayInvalidRequestError(
            message=event_data.get("message", "unknown error"),
            status_code=event_data.get("statusCode", 400),
        )

    raw_videos: list[dict[str, Any]] = event_data.get("videos", [])
    parts: list[types.Part] = []
    for video_data in raw_videos:
        vtype = video_data.get("type", "base64")
        media_type = video_data.get("mediaType", "video/mp4")

        if vtype == "url":
            downloaded_bytes, content_type = await core.helpers.files.download(
                video_data["url"]
            )
            if content_type:
                media_type = content_type
            parts.append(types.FilePart(data=downloaded_bytes, media_type=media_type))
        else:
            raw_data = video_data.get("data", "")
            parts.append(types.FilePart(data=raw_data, media_type=media_type))

    return types.Message(role="assistant", parts=parts)


async def generate(
    client: core.client.Client,
    model: core.model.Model[Any],
    messages: list[types.Message],
    params: core.GenerateParams,
) -> types.Message:
    """Generate media through the AI Gateway."""
    if isinstance(params, core.VideoParams):
        return await _generate_video(client, model, messages, params)
    return await _generate_image(client, model, messages, params)
