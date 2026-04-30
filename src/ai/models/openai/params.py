from typing import Any, ClassVar, Literal

import pydantic

_PARAMS_CONFIG = pydantic.ConfigDict(frozen=True, populate_by_name=True)

type OpenAIReasoningEffort = Literal[
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
]
type OpenAIServiceTier = Literal["auto", "default", "flex", "priority"]
type OpenAIPromptCacheRetention = Literal["in_memory", "24h"]
type OpenAITextVerbosity = Literal["low", "medium", "high"]
type OpenAISystemMessageMode = Literal["system", "developer", "remove"]
type OpenAIResponsesTruncation = Literal["auto", "disabled"]
type OpenAIReasoningSummary = Literal["auto", "concise", "detailed"]
type OpenAIResponsesInclude = Literal[
    "web_search_call.action.sources",
    "code_interpreter_call.outputs",
    "computer_call_output.output.image_url",
    "file_search_call.results",
    "message.input_image.image_url",
    "message.output_text.logprobs",
    "reasoning.encrypted_content",
]


class OpenAIResponsesContextManagement(pydantic.BaseModel):
    """Server-side Responses API context compaction settings."""

    model_config = _PARAMS_CONFIG

    # Requests server-side context compaction.
    type: Literal["compaction"]
    # Input-token threshold that triggers compaction.
    compact_threshold: int


class _OpenAISharedParams(pydantic.BaseModel):
    """OpenAI options shared by Chat Completions and Responses."""

    model_config = _PARAMS_CONFIG

    # Selects OpenAI processing tier for latency/cost tradeoffs.
    service_tier: OpenAIServiceTier | None = None
    # Requests token log probabilities; an int selects top-token count.
    logprobs: bool | int | None = None
    # Associates provider-side storage metadata with the generation.
    metadata: dict[str, Any] | None = None
    # End-user identifier used by the provider for abuse monitoring.
    user: str | None = None
    # Manual cache key used to improve prompt-cache hit rates.
    prompt_cache_key: str | None = None
    # Controls how long eligible prompt-cache prefixes are retained.
    prompt_cache_retention: OpenAIPromptCacheRetention | None = None
    # Allows the model to call multiple tools in one turn.
    parallel_tool_calls: bool | None = None
    # Tunes reasoning effort for supported reasoning models.
    reasoning_effort: OpenAIReasoningEffort | None = None
    # Controls verbosity for supported GPT-5 text output.
    text_verbosity: OpenAITextVerbosity | None = None
    # Stable safety-monitoring identifier distinct from user identity.
    safety_identifier: str | None = None
    # Selects how system messages are represented for the model.
    system_message_mode: OpenAISystemMessageMode | None = None
    # Treats an otherwise unknown model id as a reasoning model.
    force_reasoning: bool | None = None
    # Enables strict JSON Schema enforcement for structured outputs.
    strict_json_schema: bool | None = None


class OpenAIChatParams(_OpenAISharedParams):
    """OpenAI Chat Completions stream options."""

    provider: ClassVar[str] = "openai"

    # Biases specified token ids up or down during sampling.
    logit_bias: dict[int, float] | None = None
    # Number of likely tokens to include per output token position.
    top_logprobs: int | None = None
    # Persists the Chat Completions response when supported.
    store: bool | None = None
    # Prediction-mode payload used to accelerate expected outputs.
    prediction: dict[str, Any] | None = None
    # Completion token cap used by reasoning chat models.
    max_completion_tokens: int | None = None


class OpenAIResponsesParams(_OpenAISharedParams):
    """OpenAI Responses API stream options."""

    provider: ClassVar[str] = "openai"

    # Previous response id to continue without resending full history.
    previous_response_id: str | None = None
    # Conversation id or object to continue a server-side conversation.
    conversation: str | dict[str, Any] | None = None
    # Overrides instructions when continuing prior context.
    instructions: str | None = None
    # Controls automatic response-input truncation behavior.
    truncation: OpenAIResponsesTruncation | None = None
    # Server-side context compaction settings.
    context_management: list[OpenAIResponsesContextManagement] | None = None
    # Requests a reasoning summary for supported reasoning models.
    reasoning_summary: OpenAIReasoningSummary | None = None
    # Extra response fields to include in the provider response.
    include: list[OpenAIResponsesInclude] | None = None
    # Total cap on built-in tool calls made during one response.
    max_tool_calls: int | None = None
    # Maximum number of output tokens for the Responses API.
    max_output_tokens: int | None = None
    # Raw built-in tool definitions for future Responses API wiring.
    builtin_tools: list[dict[str, Any]] | None = None
