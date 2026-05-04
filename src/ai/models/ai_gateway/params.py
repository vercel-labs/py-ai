from typing import Annotated, Any, Literal

import pydantic

_PARAMS_CONFIG = pydantic.ConfigDict(frozen=True, populate_by_name=True)

type GatewaySort = Literal["cost", "ttft", "tps"]
type GatewayCaching = Literal["auto"]
type GatewayCredential = dict[str, Any]
type GatewayByok = dict[str, list[GatewayCredential]]
type GatewayTimeoutMs = Annotated[int, pydantic.Field(ge=1000)]
type GatewayOpenAIReasoningEffort = Literal[
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
]
type GatewayOpenAIServiceTier = Literal["auto", "default", "flex", "priority"]
type GatewayOpenAIPromptCacheRetention = Literal["in_memory", "24h"]
type GatewayOpenAITextVerbosity = Literal["low", "medium", "high"]
type GatewayOpenAISystemMessageMode = Literal["system", "developer", "remove"]
type GatewayOpenAIResponsesTruncation = Literal["auto", "disabled"]
type GatewayOpenAIReasoningSummary = Literal["auto", "concise", "detailed"]
type GatewayOpenAIResponsesInclude = Literal[
    "web_search_call.action.sources",
    "code_interpreter_call.outputs",
    "computer_call_output.output.image_url",
    "file_search_call.results",
    "message.input_image.image_url",
    "message.output_text.logprobs",
    "reasoning.encrypted_content",
]
type GatewayAnthropicEffort = Literal["low", "medium", "high", "xhigh", "max"]
type GatewayAnthropicSpeed = Literal["fast", "standard"]
type GatewayAnthropicInferenceGeo = Literal["us", "global"]
type GatewayAnthropicServiceTier = Literal["auto", "standard_only"]
type GatewayAnthropicStructuredOutputMode = Literal[
    "output_format",
    "json_tool",
    "auto",
]


class GatewayProviderTimeouts(pydantic.BaseModel):
    """Per-provider BYOK timeouts in milliseconds."""

    model_config = _PARAMS_CONFIG

    # Per-provider BYOK startup timeouts in milliseconds.
    byok: dict[str, GatewayTimeoutMs] | None = None


class _GatewayEscapeHatchParams(pydantic.BaseModel):
    """Raw request hooks consumed by the Gateway adapter, not dumped directly."""

    model_config = _PARAMS_CONFIG

    # Raw body fields for request options this wrapper has not typed yet.
    extra_body: dict[str, Any] | None = pydantic.Field(
        default=None,
        validation_alias="extraBody",
        exclude=True,
    )
    # Raw HTTP headers to send on this Gateway request.
    extra_headers: dict[str, str] | None = pydantic.Field(
        default=None,
        validation_alias="extraHeaders",
        exclude=True,
    )


class GatewayParams(_GatewayEscapeHatchParams):
    """AI Gateway routing and request options for language-model streams."""

    # Restricts routing to this allowlist of provider slugs.
    only: list[str] | None = None
    # Tries provider slugs in this explicit fallback order.
    order: list[str] | None = None
    # Sorts candidate providers by cost or performance before routing.
    sort: GatewaySort | None = None
    # Tries these model slugs as request-scoped model fallbacks.
    models: list[str] | None = None
    # Request-scoped provider credentials overriding cached BYOK.
    byok: GatewayByok | None = None
    # Enables automatic prompt caching when supported.
    caching: GatewayCaching | None = None
    # Filters system providers to those with zero data retention.
    zero_data_retention: bool | None = pydantic.Field(
        default=None,
        validation_alias="zeroDataRetention",
        serialization_alias="zeroDataRetention",
    )
    # Filters system providers to those that do not train on prompts.
    disallow_prompt_training: bool | None = pydantic.Field(
        default=None,
        validation_alias="disallowPromptTraining",
        serialization_alias="disallowPromptTraining",
    )
    # Filters system providers to HIPAA-compliant gateway providers.
    hipaa_compliant: bool | None = pydantic.Field(
        default=None,
        validation_alias="hipaaCompliant",
        serialization_alias="hipaaCompliant",
    )
    # Usage-reporting tags for attribution and filtering.
    tags: list[str] | None = None
    # End-user identifier for spend tracking and attribution.
    user: str | None = None
    # Entity id against which quota is tracked and enforced.
    quota_entity_id: str | None = pydantic.Field(
        default=None,
        validation_alias="quotaEntityId",
        serialization_alias="quotaEntityId",
    )
    # Per-provider BYOK startup timeouts before trying fallbacks.
    provider_timeouts: GatewayProviderTimeouts | None = pydantic.Field(
        default=None,
        validation_alias="providerTimeouts",
        serialization_alias="providerTimeouts",
    )


class GatewayOpenAIResponsesContextManagement(pydantic.BaseModel):
    """Server-side Responses API context compaction settings via Gateway."""

    model_config = _PARAMS_CONFIG

    type: Literal["compaction"]
    compact_threshold: int = pydantic.Field(
        validation_alias="compactThreshold",
        serialization_alias="compactThreshold",
    )


class _GatewayOpenAISharedParams(_GatewayEscapeHatchParams):
    """OpenAI options forwarded through AI Gateway."""

    service_tier: GatewayOpenAIServiceTier | None = pydantic.Field(
        default=None,
        validation_alias="serviceTier",
        serialization_alias="serviceTier",
    )
    logprobs: bool | int | None = None
    metadata: dict[str, Any] | None = None
    user: str | None = None
    prompt_cache_key: str | None = pydantic.Field(
        default=None,
        validation_alias="promptCacheKey",
        serialization_alias="promptCacheKey",
    )
    prompt_cache_retention: GatewayOpenAIPromptCacheRetention | None = pydantic.Field(
        default=None,
        validation_alias="promptCacheRetention",
        serialization_alias="promptCacheRetention",
    )
    parallel_tool_calls: bool | None = pydantic.Field(
        default=None,
        validation_alias="parallelToolCalls",
        serialization_alias="parallelToolCalls",
    )
    reasoning_effort: GatewayOpenAIReasoningEffort | None = pydantic.Field(
        default=None,
        validation_alias="reasoningEffort",
        serialization_alias="reasoningEffort",
    )
    text_verbosity: GatewayOpenAITextVerbosity | None = pydantic.Field(
        default=None,
        validation_alias="textVerbosity",
        serialization_alias="textVerbosity",
    )
    safety_identifier: str | None = pydantic.Field(
        default=None,
        validation_alias="safetyIdentifier",
        serialization_alias="safetyIdentifier",
    )
    system_message_mode: GatewayOpenAISystemMessageMode | None = pydantic.Field(
        default=None,
        validation_alias="systemMessageMode",
        serialization_alias="systemMessageMode",
    )
    force_reasoning: bool | None = pydantic.Field(
        default=None,
        validation_alias="forceReasoning",
        serialization_alias="forceReasoning",
    )
    strict_json_schema: bool | None = pydantic.Field(
        default=None,
        validation_alias="strictJsonSchema",
        serialization_alias="strictJsonSchema",
    )


class GatewayOpenAIChatParams(_GatewayOpenAISharedParams):
    """OpenAI Chat Completions provider options forwarded through AI Gateway."""

    logit_bias: dict[int, float] | None = pydantic.Field(
        default=None,
        validation_alias="logitBias",
        serialization_alias="logitBias",
    )
    top_logprobs: int | None = pydantic.Field(
        default=None,
        validation_alias="topLogprobs",
        serialization_alias="topLogprobs",
    )
    store: bool | None = None
    prediction: dict[str, Any] | None = None
    max_completion_tokens: int | None = pydantic.Field(
        default=None,
        validation_alias="maxCompletionTokens",
        serialization_alias="maxCompletionTokens",
    )


class GatewayOpenAIResponsesParams(_GatewayOpenAISharedParams):
    """OpenAI Responses provider options forwarded through AI Gateway."""

    previous_response_id: str | None = pydantic.Field(
        default=None,
        validation_alias="previousResponseId",
        serialization_alias="previousResponseId",
    )
    conversation: str | dict[str, Any] | None = None
    instructions: str | None = None
    truncation: GatewayOpenAIResponsesTruncation | None = None
    context_management: list[GatewayOpenAIResponsesContextManagement] | None = (
        pydantic.Field(
            default=None,
            validation_alias="contextManagement",
            serialization_alias="contextManagement",
        )
    )
    reasoning_summary: GatewayOpenAIReasoningSummary | None = pydantic.Field(
        default=None,
        validation_alias="reasoningSummary",
        serialization_alias="reasoningSummary",
    )
    include: list[GatewayOpenAIResponsesInclude] | None = None
    max_tool_calls: int | None = pydantic.Field(
        default=None,
        validation_alias="maxToolCalls",
        serialization_alias="maxToolCalls",
    )
    max_output_tokens: int | None = pydantic.Field(
        default=None,
        validation_alias="maxOutputTokens",
        serialization_alias="maxOutputTokens",
    )
    builtin_tools: list[dict[str, Any]] | None = pydantic.Field(
        default=None,
        validation_alias="builtinTools",
        serialization_alias="builtinTools",
    )


class GatewayAnthropicAdaptiveThinking(pydantic.BaseModel):
    """Adaptive thinking configuration forwarded through AI Gateway."""

    model_config = _PARAMS_CONFIG

    type: Literal["adaptive"]
    display: Literal["omitted", "summarized"] | None = None


class GatewayAnthropicEnabledThinking(pydantic.BaseModel):
    """Fixed-budget extended thinking forwarded through AI Gateway."""

    model_config = _PARAMS_CONFIG

    type: Literal["enabled"]
    budget_tokens: int | None = pydantic.Field(
        default=None,
        validation_alias="budgetTokens",
        serialization_alias="budgetTokens",
    )


class GatewayAnthropicDisabledThinking(pydantic.BaseModel):
    """Disables Anthropic extended thinking via AI Gateway."""

    model_config = _PARAMS_CONFIG

    type: Literal["disabled"]


type GatewayAnthropicThinking = Annotated[
    GatewayAnthropicAdaptiveThinking
    | GatewayAnthropicEnabledThinking
    | GatewayAnthropicDisabledThinking,
    pydantic.Field(discriminator="type"),
]


class GatewayAnthropicTaskBudget(pydantic.BaseModel):
    """Advisory token budget forwarded through AI Gateway."""

    model_config = _PARAMS_CONFIG

    type: Literal["tokens"]
    total: int
    remaining: int | None = None


class GatewayAnthropicMetadata(pydantic.BaseModel):
    """Anthropic request metadata forwarded through AI Gateway."""

    model_config = _PARAMS_CONFIG

    user_id: str | None = pydantic.Field(
        default=None,
        validation_alias="userId",
        serialization_alias="userId",
    )


class GatewayAnthropicCacheControl(pydantic.BaseModel):
    """Anthropic prompt-cache control forwarded through AI Gateway."""

    model_config = _PARAMS_CONFIG

    type: Literal["ephemeral"]
    ttl: Literal["5m", "1h"] | None = None


class GatewayAnthropicMCPToolConfiguration(pydantic.BaseModel):
    """Deprecated Anthropic MCP tool filter block forwarded through AI Gateway."""

    model_config = _PARAMS_CONFIG

    enabled: bool | None = None
    allowed_tools: list[str] | None = pydantic.Field(
        default=None,
        validation_alias="allowedTools",
        serialization_alias="allowedTools",
    )


class GatewayAnthropicMCPServer(pydantic.BaseModel):
    """Remote MCP server option forwarded through AI Gateway."""

    model_config = _PARAMS_CONFIG

    type: Literal["url"]
    name: str
    url: str
    authorization_token: str | None = pydantic.Field(
        default=None,
        validation_alias="authorizationToken",
        serialization_alias="authorizationToken",
    )
    tool_configuration: GatewayAnthropicMCPToolConfiguration | None = pydantic.Field(
        default=None,
        validation_alias="toolConfiguration",
        serialization_alias="toolConfiguration",
    )


class GatewayAnthropicProviderSkill(pydantic.BaseModel):
    """Anthropic-hosted skill option forwarded through AI Gateway."""

    model_config = _PARAMS_CONFIG

    type: Literal["anthropic"]
    skill_id: str = pydantic.Field(
        validation_alias="skillId",
        serialization_alias="skillId",
    )
    version: str | None = None


class GatewayAnthropicCustomSkill(pydantic.BaseModel):
    """Custom skill option forwarded through AI Gateway."""

    model_config = _PARAMS_CONFIG

    type: Literal["custom"]
    provider_reference: dict[str, str] = pydantic.Field(
        validation_alias="providerReference",
        serialization_alias="providerReference",
    )
    version: str | None = None


type GatewayAnthropicSkill = Annotated[
    GatewayAnthropicProviderSkill | GatewayAnthropicCustomSkill,
    pydantic.Field(discriminator="type"),
]


class GatewayAnthropicContainer(pydantic.BaseModel):
    """Container/session option forwarded through AI Gateway."""

    model_config = _PARAMS_CONFIG

    id: str | None = None
    skills: list[GatewayAnthropicSkill] | None = None


class GatewayAnthropicParams(_GatewayEscapeHatchParams):
    """Anthropic Messages provider options forwarded through AI Gateway."""

    thinking: GatewayAnthropicThinking | None = None
    effort: GatewayAnthropicEffort | None = None
    task_budget: GatewayAnthropicTaskBudget | None = pydantic.Field(
        default=None,
        validation_alias="taskBudget",
        serialization_alias="taskBudget",
    )
    speed: GatewayAnthropicSpeed | None = None
    inference_geo: GatewayAnthropicInferenceGeo | None = pydantic.Field(
        default=None,
        validation_alias="inferenceGeo",
        serialization_alias="inferenceGeo",
    )
    metadata: GatewayAnthropicMetadata | None = None
    context_management: dict[str, Any] | None = pydantic.Field(
        default=None,
        validation_alias="contextManagement",
        serialization_alias="contextManagement",
    )
    container: GatewayAnthropicContainer | None = None
    service_tier: GatewayAnthropicServiceTier | None = pydantic.Field(
        default=None,
        validation_alias="serviceTier",
        serialization_alias="serviceTier",
    )
    cache_control: GatewayAnthropicCacheControl | None = pydantic.Field(
        default=None,
        validation_alias="cacheControl",
        serialization_alias="cacheControl",
    )
    mcp_servers: list[GatewayAnthropicMCPServer] | None = pydantic.Field(
        default=None,
        validation_alias="mcpServers",
        serialization_alias="mcpServers",
    )
    betas: list[str] | None = pydantic.Field(
        default=None,
        validation_alias="anthropicBeta",
        serialization_alias="anthropicBeta",
    )
    send_reasoning: bool | None = pydantic.Field(
        default=None,
        validation_alias="sendReasoning",
        serialization_alias="sendReasoning",
    )
    tool_streaming: bool | None = pydantic.Field(
        default=None,
        validation_alias="toolStreaming",
        serialization_alias="toolStreaming",
    )
    structured_output_mode: GatewayAnthropicStructuredOutputMode | None = (
        pydantic.Field(
            default=None,
            validation_alias="structuredOutputMode",
            serialization_alias="structuredOutputMode",
        )
    )
    disable_parallel_tool_use: bool | None = pydantic.Field(
        default=None,
        validation_alias="disableParallelToolUse",
        serialization_alias="disableParallelToolUse",
    )

    @pydantic.field_validator("structured_output_mode", mode="before")
    @classmethod
    def _validate_structured_output_mode(cls, value: Any) -> Any:
        return {
            "outputFormat": "output_format",
            "jsonTool": "json_tool",
        }.get(value, value)

    @pydantic.field_serializer("structured_output_mode")
    def _serialize_structured_output_mode(
        self,
        value: GatewayAnthropicStructuredOutputMode | None,
    ) -> str | None:
        if value is None:
            return None
        return {
            "output_format": "outputFormat",
            "json_tool": "jsonTool",
        }.get(value, value)


type GatewayStreamParams = (
    GatewayParams
    | GatewayOpenAIChatParams
    | GatewayOpenAIResponsesParams
    | GatewayAnthropicParams
)

GATEWAY_STREAM_PARAMS_TYPES = (
    GatewayParams,
    GatewayOpenAIChatParams,
    GatewayOpenAIResponsesParams,
    GatewayAnthropicParams,
)
