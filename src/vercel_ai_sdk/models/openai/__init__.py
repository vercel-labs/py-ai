"""OpenAI provider adapter."""

from .llm import OpenAIModel, _messages_to_openai

__all__ = ["OpenAIModel", "_messages_to_openai"]
