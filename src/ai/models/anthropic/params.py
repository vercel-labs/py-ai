from typing import Annotated, Any, ClassVar, Literal

import pydantic

_PARAMS_CONFIG = pydantic.ConfigDict(frozen=True, populate_by_name=True)

type AnthropicEffort = Literal["low", "medium", "high", "xhigh", "max"]
type AnthropicSpeed = Literal["fast", "standard"]
type AnthropicInferenceGeo = Literal["us", "global"]
type AnthropicServiceTier = Literal["auto", "standard_only"]
type AnthropicStructuredOutputMode = Literal[
    "output_format",
    "json_tool",
    "auto",
]


class AnthropicAdaptiveThinking(pydantic.BaseModel):
    """Adaptive thinking configuration for newer Claude models."""

    model_config = _PARAMS_CONFIG

    # Enables adaptive thinking for supported newer Claude models.
    type: Literal["adaptive"]
    # Controls whether returned thinking text is omitted or summarized.
    display: Literal["omitted", "summarized"] | None = None


class AnthropicEnabledThinking(pydantic.BaseModel):
    """Fixed-budget extended thinking configuration."""

    model_config = _PARAMS_CONFIG

    # Enables extended thinking with an optional token budget.
    type: Literal["enabled"]
    # Maximum tokens available for the thinking block.
    budget_tokens: int | None = None


class AnthropicDisabledThinking(pydantic.BaseModel):
    """Disables extended thinking for the request."""

    model_config = _PARAMS_CONFIG

    # Disables extended thinking.
    type: Literal["disabled"]


type AnthropicThinking = Annotated[
    AnthropicAdaptiveThinking | AnthropicEnabledThinking | AnthropicDisabledThinking,
    pydantic.Field(discriminator="type"),
]


class AnthropicTaskBudget(pydantic.BaseModel):
    """Advisory token budget for agentic turns."""

    model_config = _PARAMS_CONFIG

    # Declares the budget unit as tokens.
    type: Literal["tokens"]
    # Total token budget available for the current task.
    total: int
    # Remaining token budget available for the current task.
    remaining: int | None = None


class AnthropicMetadata(pydantic.BaseModel):
    """Request metadata forwarded to Anthropic."""

    model_config = _PARAMS_CONFIG

    # Opaque external user identifier for the request.
    user_id: str | None = None


class AnthropicCacheControl(pydantic.BaseModel):
    """Top-level automatic prompt-cache policy."""

    model_config = _PARAMS_CONFIG

    # Uses Anthropic ephemeral prompt caching.
    type: Literal["ephemeral"]
    # Prompt-cache time-to-live.
    ttl: Literal["5m", "1h"] | None = None


class AnthropicMCPToolConfiguration(pydantic.BaseModel):
    """Deprecated MCP server tool filter block."""

    model_config = _PARAMS_CONFIG

    # Whether tools from this MCP server are enabled.
    enabled: bool | None = None
    # Optional allowlist of MCP tools from this server.
    allowed_tools: list[str] | None = None


class AnthropicMCPServer(pydantic.BaseModel):
    """Remote MCP server available to Claude for the request."""

    model_config = _PARAMS_CONFIG

    # Declares a remote URL MCP server.
    type: Literal["url"]
    # Name used to identify this MCP server.
    name: str
    # Remote MCP server URL.
    url: str
    # Optional bearer token for the remote MCP server.
    authorization_token: str | None = None
    # Deprecated MCP server tool filter settings.
    tool_configuration: AnthropicMCPToolConfiguration | None = None


class AnthropicProviderSkill(pydantic.BaseModel):
    """Anthropic-hosted agent skill reference."""

    model_config = _PARAMS_CONFIG

    # References an Anthropic-hosted skill.
    type: Literal["anthropic"]
    # Anthropic-hosted skill id.
    skill_id: str
    # Optional skill version.
    version: str | None = None


class AnthropicCustomSkill(pydantic.BaseModel):
    """Custom agent skill referenced by provider-specific ids."""

    model_config = _PARAMS_CONFIG

    # References a custom provider skill.
    type: Literal["custom"]
    # Provider-specific skill reference identifiers.
    provider_reference: dict[str, str]
    # Optional skill version.
    version: str | None = None


type AnthropicSkill = Annotated[
    AnthropicProviderSkill | AnthropicCustomSkill,
    pydantic.Field(discriminator="type"),
]


class AnthropicContainer(pydantic.BaseModel):
    """Container/session and optional agent skills for programmatic tools."""

    model_config = _PARAMS_CONFIG

    # Existing container/session id for programmatic tools.
    id: str | None = None
    # Agent skills available in the container.
    skills: list[AnthropicSkill] | None = None


class AnthropicParams(pydantic.BaseModel):
    """Anthropic Messages stream options."""

    model_config = _PARAMS_CONFIG

    provider: ClassVar[str] = "anthropic"

    # Enables, disables, or adapts Claude extended thinking.
    thinking: AnthropicThinking | None = None
    # Controls output reasoning effort through output_config.
    effort: AnthropicEffort | None = None
    # Advisory total/remaining token budget for agentic tasks.
    task_budget: AnthropicTaskBudget | None = None
    # Selects faster or standard inference when supported.
    speed: AnthropicSpeed | None = None
    # Controls whether inference can run globally or only in the US.
    inference_geo: AnthropicInferenceGeo | None = None
    # Provider request metadata, currently an opaque end-user id.
    metadata: AnthropicMetadata | None = None
    # Advanced Anthropic context clearing/compaction settings.
    context_management: dict[str, Any] | None = None
    # Container id and optional skills for programmatic tool execution.
    container: AnthropicContainer | None = None
    # Selects automatic priority fallback or standard-only capacity.
    service_tier: AnthropicServiceTier | None = None
    # Top-level prompt-cache control policy.
    cache_control: AnthropicCacheControl | None = None
    # Remote MCP servers Claude can call during the request.
    mcp_servers: list[AnthropicMCPServer] | None = None
    # Raw beta feature identifiers to send with the request.
    betas: list[str] | None = None
    # Sends prior reasoning blocks back to models that support them.
    send_reasoning: bool | None = None
    # Enables fine-grained eager streaming for tool inputs by default.
    tool_streaming: bool | None = None
    # Chooses native output_format, JSON-tool fallback, or auto mode.
    structured_output_mode: AnthropicStructuredOutputMode | None = None
    # Disables parallel tool use so Claude uses at most one tool.
    disable_parallel_tool_use: bool | None = None
