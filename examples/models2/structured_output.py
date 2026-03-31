"""Structured output — get validated JSON from the model."""

import asyncio

import pydantic

from vercel_ai_sdk import models2 as m
from vercel_ai_sdk.types import messages as messages_

model = m.Model(
    id="anthropic/claude-sonnet-4",
    api="ai-gateway",
    provider="ai-gateway",
)


class Recipe(pydantic.BaseModel):
    name: str
    ingredients: list[str]
    steps: list[str]
    prep_time_minutes: int


messages = [
    messages_.Message(
        role="user",
        parts=[messages_.TextPart(text="Give me a simple pancake recipe.")],
    ),
]


async def main() -> None:
    # Stream with structured output — watch JSON arrive, get validated at the end
    async for msg in m.stream(model, messages, output_type=Recipe):
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)
        if msg.output:
            recipe: Recipe = msg.output
            print(f"\n\nParsed recipe: {recipe.name}")
            print(f"  Ingredients: {', '.join(recipe.ingredients)}")
            print(f"  Prep time: {recipe.prep_time_minutes} min")


if __name__ == "__main__":
    asyncio.run(main())
