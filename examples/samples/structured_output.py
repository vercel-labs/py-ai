"""Structured output — get validated JSON from the model."""

import asyncio

import pydantic

import ai

model = ai.ai_gateway("anthropic/claude-sonnet-4")


class Recipe(pydantic.BaseModel):
    name: str
    ingredients: list[str]
    steps: list[str]
    prep_time_minutes: int


messages = [ai.user_message("Give me a simple pancake recipe.")]


async def main() -> None:
    # Broken for now: stream(output_type=...) requests JSON/schema mode, but
    # the stream wrapper does not yet validate final text into s.output.
    raise RuntimeError("structured output aggregation needs to be implemented")


if __name__ == "__main__":
    asyncio.run(main())
