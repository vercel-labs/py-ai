from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Sequence
from typing import override

import pydantic

from .. import core
from ..anthropic import AnthropicModel
from ..openai import OpenAIModel

_DEFAULT_BASE_URL = "https://ai-gateway.vercel.sh"


class GatewayModel(core.llm.LanguageModel):
    """Vercel AI Gateway provider.

    Pre-configured for the Vercel AI Gateway with automatic routing:
    Anthropic models use the native Anthropic API through the gateway,
    except when structured output is requested (which requires the
    OpenAI-compatible endpoint). All other models use the
    OpenAI-compatible endpoint.

    Usage::

        import vercel_ai_sdk as ai

        llm = ai.ai_gateway.GatewayModel(model="anthropic/claude-sonnet-4")

    Args:
        model: Model identifier in ``provider/model`` format
            (e.g., ``'anthropic/claude-sonnet-4'``, ``'openai/gpt-4.1'``)
        api_key: API key for the gateway. Falls back to the
            ``AI_GATEWAY_API_KEY`` environment variable.
        base_url: Gateway base URL. Defaults to
            ``https://ai-gateway.vercel.sh``.
        thinking: Enable reasoning/thinking output.
        budget_tokens: Max tokens for reasoning
            (mutually exclusive with *reasoning_effort*).
        reasoning_effort: Effort level for reasoning — ``'none'``,
            ``'minimal'``, ``'low'``, ``'medium'``, ``'high'``, ``'xhigh'``
            (mutually exclusive with *budget_tokens*; OpenAI models only).
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4",
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        thinking: bool = False,
        budget_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("AI_GATEWAY_API_KEY") or ""
        self._base_url = base_url.rstrip("/")
        self._thinking = thinking
        self._budget_tokens = budget_tokens
        self._reasoning_effort = reasoning_effort

    def _is_anthropic_model(self) -> bool:
        return self._model.startswith("anthropic/")

    def _make_openai(self) -> OpenAIModel:
        return OpenAIModel(
            model=self._model,
            base_url=f"{self._base_url}/v1",
            api_key=self._api_key,
            thinking=self._thinking,
            budget_tokens=self._budget_tokens,
            reasoning_effort=self._reasoning_effort,
        )

    def _make_anthropic(self) -> AnthropicModel:
        return AnthropicModel(
            model=self._model,
            base_url=self._base_url,
            api_key=self._api_key,
            thinking=self._thinking,
            budget_tokens=self._budget_tokens or 10000,
        )

    def _resolve(
        self, output_type: type[pydantic.BaseModel] | None
    ) -> core.llm.LanguageModel:
        """Pick delegate based on model and feature requirements.

        - Anthropic models without structured output use the native
          Anthropic API (richer reasoning support, native tool format).
        - Anthropic models *with* structured output use OpenAI-compat
          (structured output via the Anthropic-native gateway endpoint
          is not currently supported).
        - All other models use OpenAI-compat.
        """
        if self._is_anthropic_model() and output_type is None:
            return self._make_anthropic()
        return self._make_openai()

    @override
    async def stream(
        self,
        messages: list[core.messages.Message],
        tools: Sequence[core.tools.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
    ) -> AsyncGenerator[core.messages.Message]:
        delegate = self._resolve(output_type)
        async for msg in delegate.stream(messages, tools, output_type):
            yield msg
