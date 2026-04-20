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
    # Stream with structured output — watch JSON arrive, get validated at the end
    async for msg in await ai.stream(model, messages, output_type=Recipe):
        for ev in msg.deltas:
            if isinstance(ev.part, ai.TextPart):
                print(ev.chunk, end="", flush=True)
        if msg.output:
            recipe: Recipe = msg.output
            print(f"\n\nParsed recipe: {recipe.name}")
            print(f"  Ingredients: {', '.join(recipe.ingredients)}")
            print(f"  Prep time: {recipe.prep_time_minutes} min")


if __name__ == "__main__":
    asyncio.run(main())
