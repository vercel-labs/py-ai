"""Google adapter - Gemini Developer API.

Message/tool conversion and streaming via the first-party ``google-genai`` SDK.
The SDK client is constructed from :class:`Client` params on each call.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any

import pydantic

from ... import types
from ...types import events
from .. import core
from . import tools as google_tools

PROVIDER_NAME = "google"
_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _provider_metadata(**values: Any) -> dict[str, Any]:
    return {"provider": PROVIDER_NAME, **values}


def _get_field(value: Any, *names: str, default: Any = None) -> Any:
    if value is None:
        return default
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _dump_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, pydantic.BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {str(k): _dump_jsonable(v) for k, v in value.items() if v is not None}
    if isinstance(value, list | tuple):
        return [_dump_jsonable(v) for v in value]
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return enum_value
    if hasattr(value, "__dict__"):
        return {
            k: _dump_jsonable(v)
            for k, v in vars(value).items()
            if not k.startswith("_") and v is not None
        }
    return value


def _wire_value(value: Any) -> Any:
    enum_value = getattr(value, "value", None)
    return enum_value if enum_value is not None else value


def _json_dumps(value: Any) -> str:
    return json.dumps(_dump_jsonable(value), separators=(",", ":"))


def _parse_json_object(value: str) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


def _thought_signature_metadata(part: Any) -> dict[str, Any] | None:
    signature = _get_field(part, "thought_signature", "thoughtSignature")
    if signature is None:
        return None
    if isinstance(signature, bytes):
        return _provider_metadata(
            thoughtSignature=base64.b64encode(signature).decode("ascii"),
            thoughtSignatureEncoding="base64",
        )
    return _provider_metadata(thoughtSignature=str(signature))


def _thought_signature_from_metadata(
    provider_metadata: dict[str, Any] | None,
) -> str | bytes | None:
    if not provider_metadata:
        return None
    signature = provider_metadata.get("thoughtSignature") or provider_metadata.get(
        "thought_signature"
    )
    if not isinstance(signature, str):
        return signature
    if provider_metadata.get("thoughtSignatureEncoding") == "base64":
        return base64.b64decode(signature)
    return signature


def _file_part_to_google(part: types.messages.FilePart) -> dict[str, Any]:
    media_type = "image/jpeg" if part.media_type == "image/*" else part.media_type
    data = part.data
    if isinstance(data, str) and data.startswith(("http://", "https://", "gs://")):
        file_data: dict[str, Any] = {
            "mime_type": media_type,
            "file_uri": data,
        }
        if part.filename:
            file_data["display_name"] = part.filename
        return {"file_data": file_data}
    inline_data: dict[str, Any] = {
        "mime_type": media_type,
        "data": types.media.data_to_base64(data),
    }
    if part.filename:
        inline_data["display_name"] = part.filename
    return {"inline_data": inline_data}


def _tool_result_payload(part: types.messages.ToolResultPart) -> dict[str, Any]:
    if part.is_error:
        return {"error": str(part.result) if part.result is not None else ""}
    return {"output": part.result}


def _server_tool_metadata(
    part: types.messages.BuiltinToolCallPart | types.messages.BuiltinToolReturnPart,
) -> tuple[str | None, str | None]:
    provider_metadata = part.provider_metadata or {}
    server_tool_id = provider_metadata.get("serverToolCallId") or provider_metadata.get(
        "server_tool_call_id"
    )
    server_tool_type = provider_metadata.get("serverToolType") or provider_metadata.get(
        "server_tool_type"
    )
    if server_tool_type is None and part.tool_name.startswith("server:"):
        server_tool_type = part.tool_name.removeprefix("server:")
    return (
        str(server_tool_id) if server_tool_id is not None else None,
        str(server_tool_type) if server_tool_type is not None else None,
    )


async def _messages_to_google(
    messages: list[types.messages.Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert internal messages to google-genai content dictionaries."""
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for msg in messages:
        match msg.role:
            case "system":
                text = "".join(
                    p.text for p in msg.parts if isinstance(p, types.messages.TextPart)
                )
                if text:
                    system_parts.append(text)

            case "user":
                parts: list[dict[str, Any]] = []
                for part in msg.parts:
                    match part:
                        case types.messages.TextPart(text=text):
                            parts.append({"text": text})
                        case types.messages.FilePart():
                            parts.append(_file_part_to_google(part))
                if parts:
                    contents.append({"role": "user", "parts": parts})

            case "assistant":
                parts = []
                for part in msg.parts:
                    match part:
                        case types.messages.TextPart(text=text):
                            if text:
                                wire: dict[str, Any] = {"text": text}
                                signature = _thought_signature_from_metadata(
                                    part.provider_metadata
                                )
                                if signature is not None:
                                    wire["thought_signature"] = signature
                                parts.append(wire)
                        case types.messages.ReasoningPart(text=text):
                            if text:
                                wire = {"text": text, "thought": True}
                                signature = _thought_signature_from_metadata(
                                    part.provider_metadata
                                )
                                if signature is not None:
                                    wire["thought_signature"] = signature
                                parts.append(wire)
                        case types.messages.FilePart():
                            parts.append(_file_part_to_google(part))
                        case types.messages.ToolCallPart():
                            parts.append(
                                {
                                    "function_call": {
                                        "id": part.tool_call_id,
                                        "name": part.tool_name,
                                        "args": _parse_json_object(part.tool_args),
                                    }
                                }
                            )
                        case types.messages.BuiltinToolCallPart():
                            tool_args = _parse_json_object(part.tool_args)
                            if part.tool_name == "code_execution":
                                parts.append({"executable_code": tool_args})
                            else:
                                server_tool_id, server_tool_type = (
                                    _server_tool_metadata(part)
                                )
                                if server_tool_type is None:
                                    raise NotImplementedError(
                                        "Google BuiltinToolCallPart requires "
                                        "server tool metadata unless it is "
                                        "code_execution"
                                    )
                                parts.append(
                                    {
                                        "tool_call": {
                                            "id": server_tool_id or part.tool_call_id,
                                            "tool_type": server_tool_type,
                                            "args": tool_args,
                                        }
                                    }
                                )
                        case types.messages.BuiltinToolReturnPart():
                            if part.tool_name == "code_execution":
                                parts.append(
                                    {
                                        "code_execution_result": part.result
                                        if isinstance(part.result, dict)
                                        else {"output": part.result}
                                    }
                                )
                            else:
                                server_tool_id, server_tool_type = (
                                    _server_tool_metadata(part)
                                )
                                if server_tool_type is None:
                                    raise NotImplementedError(
                                        "Google BuiltinToolReturnPart requires "
                                        "server tool metadata unless it is "
                                        "code_execution"
                                    )
                                parts.append(
                                    {
                                        "tool_response": {
                                            "id": server_tool_id or part.tool_call_id,
                                            "tool_type": server_tool_type,
                                            "response": part.result
                                            if isinstance(part.result, dict)
                                            else {"output": part.result},
                                        }
                                    }
                                )
                if parts:
                    contents.append({"role": "model", "parts": parts})

            case "tool":
                parts = []
                for part in msg.parts:
                    if isinstance(part, types.messages.ToolResultPart):
                        parts.append(
                            {
                                "function_response": {
                                    "id": part.tool_call_id,
                                    "name": part.tool_name,
                                    "response": _tool_result_payload(part),
                                }
                            }
                        )
                if parts:
                    contents.append({"role": "user", "parts": parts})

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def _split_tools(
    tools: Sequence[types.tools.Tool],
) -> tuple[list[types.tools.Tool], list[types.tools.Tool]]:
    custom: list[types.tools.Tool] = []
    builtin: list[types.tools.Tool] = []
    for tool in tools:
        if tool.kind == "provider":
            builtin.append(tool)
        else:
            custom.append(tool)
    return custom, builtin


def _custom_tools_to_google(
    tools: Sequence[types.tools.Tool],
) -> dict[str, Any] | None:
    function_declarations: list[dict[str, Any]] = []
    for tool in tools:
        args = tool.args
        if not isinstance(args, types.tools.FunctionToolArgs):
            raise TypeError(f"function tool {tool.name!r} has invalid args")
        function_declarations.append(
            {
                "name": tool.name,
                "description": args.description or "",
                "parameters_json_schema": args.params,
            }
        )
    if not function_declarations:
        return None
    return {"function_declarations": function_declarations}


def _builtin_tools_to_google(
    tools: Sequence[types.tools.Tool],
) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    for tool in tools:
        args_model = tool.args
        if not isinstance(args_model, google_tools.GoogleProviderArgs):
            raise ValueError(
                "GoogleModel does not support provider args "
                f"{type(args_model).__name__}"
            )
        args = args_model.model_dump(mode="json", exclude_none=True)
        wire.append({args_model.google_tool_name: args})
    return wire


def _prepare_tools(
    tools: Sequence[types.tools.Tool] | None,
) -> tuple[list[dict[str, Any]] | None, bool]:
    if not tools:
        return None, False

    custom_tools, builtin_tools = _split_tools(tools)
    wire: list[dict[str, Any]] = []

    custom_wire = _custom_tools_to_google(custom_tools)
    if custom_wire is not None:
        wire.append(custom_wire)

    if builtin_tools:
        wire.extend(_builtin_tools_to_google(builtin_tools))

    return wire or None, bool(builtin_tools)


def _coerce_params(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("google stream params must be a dict")


def _make_client(client: core.client.Client) -> Any:
    """Construct a ``google-genai`` async client from our generic ``Client``."""
    from google import genai

    base_url = client.base_url.rstrip("/")
    if base_url == _DEFAULT_BASE_URL:
        return genai.Client(api_key=client.api_key or "").aio
    return genai.Client(
        api_key=client.api_key or "",
        http_options={"base_url": client.base_url},
    ).aio


def _convert_usage(usage_metadata: Any) -> types.usage.Usage | None:
    if usage_metadata is None:
        return None
    prompt_tokens = _get_field(
        usage_metadata, "prompt_token_count", "promptTokenCount", default=0
    )
    candidates_tokens = _get_field(
        usage_metadata, "candidates_token_count", "candidatesTokenCount", default=0
    )
    thoughts_tokens = _get_field(
        usage_metadata, "thoughts_token_count", "thoughtsTokenCount"
    )
    cached_tokens = _get_field(
        usage_metadata, "cached_content_token_count", "cachedContentTokenCount"
    )
    reasoning_tokens = int(thoughts_tokens) if thoughts_tokens is not None else None
    return types.usage.Usage(
        input_tokens=int(prompt_tokens or 0),
        output_tokens=int(candidates_tokens or 0) + (reasoning_tokens or 0),
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=int(cached_tokens) if cached_tokens is not None else None,
        raw=_dump_jsonable(usage_metadata),
    )


def _first_candidate(chunk: Any) -> Any:
    candidates = _get_field(chunk, "candidates", default=None)
    if not candidates:
        return None
    return candidates[0]


def _chunk_parts(chunk: Any, candidate: Any) -> list[Any]:
    content = _get_field(candidate, "content", default=None)
    parts = _get_field(content, "parts", default=None)
    if parts is not None:
        return list(parts)
    direct_parts = _get_field(chunk, "parts", default=None)
    return list(direct_parts or [])


def _tool_call_id(prefix: str = "google_tool") -> str:
    return types.messages.generate_id(prefix)


async def stream(
    client: core.client.Client,
    model: core.model.Model,
    messages: list[types.messages.Message],
    *,
    tools: Sequence[types.tools.Tool] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    **kwargs: Any,
) -> AsyncGenerator[events.Event]:
    """Stream an LLM response via the Gemini Developer API.

    ``params`` may be a raw dict of ``GenerateContentConfig`` fields.
    Provider-specific request options are forwarded without local validation.
    """
    sdk_client = _make_client(client)
    stream_params = _coerce_params(kwargs.get("params"))
    system_instruction, google_contents = await _messages_to_google(messages)
    google_tools, has_builtin_tools = _prepare_tools(tools)

    config: dict[str, Any] = dict(stream_params)
    if system_instruction:
        config["system_instruction"] = system_instruction
    if google_tools:
        config["tools"] = google_tools
    if has_builtin_tools:
        tool_config = dict(config.get("tool_config") or {})
        tool_config.setdefault("include_server_side_tool_invocations", True)
        config["tool_config"] = tool_config
    if output_type is not None:
        config["response_mime_type"] = "application/json"
        config["response_json_schema"] = output_type.model_json_schema()

    current_text_block_id: str | None = None
    current_reasoning_block_id: str | None = None
    block_counter = 0
    last_code_execution_tool_call_id: str | None = None
    last_server_tool_call_id: str | None = None
    usage: types.usage.Usage | None = None
    final_provider_metadata: dict[str, Any] | None = None

    def _close_text() -> list[events.Event]:
        nonlocal current_text_block_id
        if current_text_block_id is None:
            return []
        ev = events.TextEnd(block_id=current_text_block_id)
        current_text_block_id = None
        return [ev]

    def _close_reasoning() -> list[events.Event]:
        nonlocal current_reasoning_block_id
        if current_reasoning_block_id is None:
            return []
        ev = events.ReasoningEnd(block_id=current_reasoning_block_id)
        current_reasoning_block_id = None
        return [ev]

    def _close_language_blocks() -> list[events.Event]:
        return [*_close_text(), *_close_reasoning()]

    def _start_text(provider_metadata: dict[str, Any] | None) -> list[events.Event]:
        nonlocal block_counter, current_text_block_id
        if current_text_block_id is not None:
            return []
        current_text_block_id = str(block_counter)
        block_counter += 1
        return [
            events.TextStart(
                block_id=current_text_block_id,
                provider_metadata=provider_metadata,
            )
        ]

    def _start_reasoning(
        provider_metadata: dict[str, Any] | None,
    ) -> list[events.Event]:
        nonlocal block_counter, current_reasoning_block_id
        if current_reasoning_block_id is not None:
            return []
        current_reasoning_block_id = str(block_counter)
        block_counter += 1
        return [
            events.ReasoningStart(
                block_id=current_reasoning_block_id,
                provider_metadata=provider_metadata,
            )
        ]

    try:
        yield events.StreamStart()

        sdk_stream = sdk_client.models.generate_content_stream(
            model=model.id,
            contents=google_contents,
            config=config or None,
        )

        async for chunk in sdk_stream:
            chunk_usage = _convert_usage(
                _get_field(chunk, "usage_metadata", "usageMetadata")
            )
            if chunk_usage is not None:
                usage = chunk_usage

            candidate = _first_candidate(chunk)
            if candidate is None:
                continue

            parts = _chunk_parts(chunk, candidate)
            for part in parts:
                text = _get_field(part, "text")
                if text is not None:
                    provider_metadata = _thought_signature_metadata(part)
                    if text == "":
                        continue
                    if _get_field(part, "thought", default=False) is True:
                        for ev in _close_text():
                            yield ev
                        for ev in _start_reasoning(provider_metadata):
                            yield ev
                        if current_reasoning_block_id is not None:
                            yield events.ReasoningDelta(
                                block_id=current_reasoning_block_id,
                                chunk=str(text),
                                provider_metadata=provider_metadata,
                            )
                    else:
                        for ev in _close_reasoning():
                            yield ev
                        for ev in _start_text(provider_metadata):
                            yield ev
                        if current_text_block_id is not None:
                            yield events.TextDelta(
                                block_id=current_text_block_id,
                                chunk=str(text),
                                provider_metadata=provider_metadata,
                            )
                    continue

                inline_data = _get_field(part, "inline_data", "inlineData")
                if inline_data is not None:
                    for ev in _close_language_blocks():
                        yield ev
                    data = _get_field(inline_data, "data")
                    media_type = _get_field(
                        inline_data, "mime_type", "mimeType", default=""
                    )
                    if isinstance(data, bytes):
                        data = base64.b64encode(data).decode("ascii")
                    yield events.FileEvent(
                        block_id=types.messages.generate_id("file"),
                        media_type=str(media_type),
                        data=data or "",
                        provider_metadata=_thought_signature_metadata(part),
                    )
                    continue

                executable_code = _get_field(part, "executable_code", "executableCode")
                if executable_code is not None:
                    for ev in _close_language_blocks():
                        yield ev
                    tool_call_id = _get_field(executable_code, "id") or _tool_call_id(
                        "google_builtin"
                    )
                    last_code_execution_tool_call_id = str(tool_call_id)
                    payload = _dump_jsonable(executable_code)
                    payload_json = _json_dumps(payload)
                    provider_metadata = _provider_metadata()
                    yield events.BuiltinToolStart(
                        tool_call_id=last_code_execution_tool_call_id,
                        tool_name="code_execution",
                        provider_metadata=provider_metadata,
                    )
                    yield events.BuiltinToolDelta(
                        tool_call_id=last_code_execution_tool_call_id,
                        chunk=payload_json,
                        provider_metadata=provider_metadata,
                    )
                    yield events.BuiltinToolEnd(
                        tool_call_id=last_code_execution_tool_call_id,
                        tool_call=types.messages.BuiltinToolCallPart(
                            tool_call_id=last_code_execution_tool_call_id,
                            tool_name="code_execution",
                            provider_metadata=provider_metadata,
                        ),
                        provider_metadata=provider_metadata,
                    )
                    continue

                code_execution_result = _get_field(
                    part, "code_execution_result", "codeExecutionResult"
                )
                if code_execution_result is not None:
                    tool_call_id = _get_field(code_execution_result, "id") or (
                        last_code_execution_tool_call_id
                        or _tool_call_id("google_builtin")
                    )
                    result = _dump_jsonable(code_execution_result)
                    provider_metadata = _provider_metadata()
                    yield events.BuiltinToolResult(
                        tool_call_id=str(tool_call_id),
                        result=types.messages.BuiltinToolReturnPart(
                            tool_call_id=str(tool_call_id),
                            tool_name="code_execution",
                            result=result,
                            provider_metadata=provider_metadata,
                        ),
                        provider_metadata=provider_metadata,
                    )
                    last_code_execution_tool_call_id = None
                    continue

                tool_call = _get_field(part, "tool_call", "toolCall")
                if tool_call is not None:
                    for ev in _close_language_blocks():
                        yield ev
                    tool_call_id = _get_field(tool_call, "id") or _tool_call_id(
                        "google_builtin"
                    )
                    tool_type = str(
                        _wire_value(_get_field(tool_call, "tool_type", "toolType"))
                    )
                    last_server_tool_call_id = str(tool_call_id)
                    provider_metadata = _provider_metadata(
                        serverToolCallId=str(tool_call_id),
                        serverToolType=tool_type,
                    )
                    args = _get_field(tool_call, "args", default={}) or {}
                    yield events.BuiltinToolStart(
                        tool_call_id=str(tool_call_id),
                        tool_name=f"server:{tool_type}",
                        provider_metadata=provider_metadata,
                    )
                    yield events.BuiltinToolDelta(
                        tool_call_id=str(tool_call_id),
                        chunk=_json_dumps(args),
                        provider_metadata=provider_metadata,
                    )
                    yield events.BuiltinToolEnd(
                        tool_call_id=str(tool_call_id),
                        tool_call=types.messages.BuiltinToolCallPart(
                            tool_call_id=str(tool_call_id),
                            tool_name=f"server:{tool_type}",
                            provider_metadata=provider_metadata,
                        ),
                        provider_metadata=provider_metadata,
                    )
                    continue

                tool_response = _get_field(part, "tool_response", "toolResponse")
                if tool_response is not None:
                    tool_call_id = _get_field(tool_response, "id") or (
                        last_server_tool_call_id or _tool_call_id("google_builtin")
                    )
                    tool_type = str(
                        _wire_value(_get_field(tool_response, "tool_type", "toolType"))
                    )
                    provider_metadata = _provider_metadata(
                        serverToolCallId=str(tool_call_id),
                        serverToolType=tool_type,
                    )
                    yield events.BuiltinToolResult(
                        tool_call_id=str(tool_call_id),
                        result=types.messages.BuiltinToolReturnPart(
                            tool_call_id=str(tool_call_id),
                            tool_name=f"server:{tool_type}",
                            result=_dump_jsonable(
                                _get_field(tool_response, "response", default={})
                            ),
                            provider_metadata=provider_metadata,
                        ),
                        provider_metadata=provider_metadata,
                    )
                    last_server_tool_call_id = None
                    continue

                function_call = _get_field(part, "function_call", "functionCall")
                if function_call is not None:
                    for ev in _close_language_blocks():
                        yield ev
                    tool_name = _get_field(function_call, "name")
                    if not tool_name:
                        continue
                    tool_call_id = _get_field(function_call, "id") or _tool_call_id()
                    args = _get_field(function_call, "args", default={}) or {}
                    args_json = _json_dumps(args)
                    provider_metadata = _thought_signature_metadata(part)
                    yield events.ToolStart(
                        tool_call_id=str(tool_call_id),
                        tool_name=str(tool_name),
                        provider_metadata=provider_metadata,
                    )
                    yield events.ToolDelta(
                        tool_call_id=str(tool_call_id),
                        chunk=args_json,
                        provider_metadata=provider_metadata,
                    )
                    yield events.ToolEnd(
                        tool_call_id=str(tool_call_id),
                        tool_call=types.messages.DUMMY_TOOL_CALL,
                        provider_metadata=provider_metadata,
                    )
                    continue

            finish_reason = _get_field(candidate, "finish_reason", "finishReason")
            if finish_reason is not None:
                raw_usage = _get_field(chunk, "usage_metadata", "usageMetadata")
                final_provider_metadata = _provider_metadata(
                    promptFeedback=_dump_jsonable(
                        _get_field(chunk, "prompt_feedback", "promptFeedback")
                    ),
                    groundingMetadata=_dump_jsonable(
                        _get_field(candidate, "grounding_metadata", "groundingMetadata")
                    ),
                    urlContextMetadata=_dump_jsonable(
                        _get_field(
                            candidate,
                            "url_context_metadata",
                            "urlContextMetadata",
                        )
                    ),
                    safetyRatings=_dump_jsonable(
                        _get_field(candidate, "safety_ratings", "safetyRatings")
                    ),
                    usageMetadata=_dump_jsonable(raw_usage),
                    finishReason=_dump_jsonable(finish_reason),
                    finishMessage=_dump_jsonable(
                        _get_field(candidate, "finish_message", "finishMessage")
                    ),
                )

        for ev in _close_language_blocks():
            yield ev
        yield events.StreamEnd(usage=usage, provider_metadata=final_provider_metadata)
    finally:
        close = getattr(sdk_client, "aclose", None)
        if close is not None:
            await close()
