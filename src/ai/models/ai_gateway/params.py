"""AI Gateway request-scoped routing options.

Only the gateway-unique routing/BYOK params live here. Typed provider-specific
options (OpenAI, Anthropic) are imported from the native ``openai`` /
``anthropic`` packages and reused as-is. Untyped options for providers without
local params use :class:`ProviderOptions` — see :data:`GatewayStreamParams`.

The native modules import only ``pydantic`` + stdlib + internal types,
so referencing them here does not pull in the ``openai`` / ``anthropic``
SDKs (those are loaded lazily by each package's adapter).
"""

from typing import Annotated, Any, Literal

import pydantic

from ..anthropic.params import AnthropicParams
from ..openai.params import OpenAIChatParams, OpenAIResponsesParams

_PARAMS_CONFIG = pydantic.ConfigDict(frozen=True, populate_by_name=True)

type GatewaySort = Literal["cost", "ttft", "tps"]
type GatewayCaching = Literal["auto"]
type GatewayCredential = dict[str, Any]
type GatewayByok = dict[str, list[GatewayCredential]]
type GatewayTimeoutMs = Annotated[int, pydantic.Field(ge=1000)]


class GatewayProviderTimeouts(pydantic.BaseModel):
    """Per-provider BYOK timeouts in milliseconds."""

    model_config = _PARAMS_CONFIG

    # Per-provider BYOK startup timeouts in milliseconds.
    byok: dict[str, GatewayTimeoutMs] | None = None


class GatewayParams(pydantic.BaseModel):
    """AI Gateway routing and request options for language-model streams."""

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


class ProviderOptions(pydantic.BaseModel):
    """Untyped providerOptions bucket for Gateway providers without typed params."""

    model_config = _PARAMS_CONFIG

    provider: str = pydantic.Field(min_length=1)
    options: dict[str, Any]


type GatewayStreamParams = (
    GatewayParams
    | ProviderOptions
    | OpenAIChatParams
    | OpenAIResponsesParams
    | AnthropicParams
)

GATEWAY_STREAM_PARAMS_TYPES = (
    GatewayParams,
    ProviderOptions,
    OpenAIChatParams,
    OpenAIResponsesParams,
    AnthropicParams,
)
