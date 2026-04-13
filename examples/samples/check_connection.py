"""Check connection and list models — verify credentials and model availability."""

import asyncio

import ai

MODELS = [
    ai.ai_gateway("anthropic/claude-sonnet-4"),
    ai.anthropic("claude-sonnet-4-20250514"),
    ai.openai("gpt-5.4-mini"),
]

PROVIDERS = [
    ("ai_gateway", ai.ai_gateway),
    ("anthropic", ai.anthropic),
    ("openai", ai.openai),
]


async def _check(model: ai.Model) -> None:
    try:
        ok = await ai.check_connection(model)
        status = "[OK]  " if ok else "[FAIL]"
    except Exception as exc:
        status = f"[ERR] {exc}"
    print(f"  {status}  {model.provider}/{model.id}")


async def _list_models(name: str, provider: object) -> None:
    try:
        ids: list[str] = await provider.list()  # type: ignore[union-attr]
        print(f"  {name}: {len(ids)} models")
        for mid in ids:
            print(f"    - {mid}")
    except Exception as exc:
        print(f"  {name}: [ERR] {exc}")


async def main() -> None:
    print("Checking connections...\n")
    await asyncio.gather(*[_check(m) for m in MODELS])

    print("\nListing models...\n")
    await asyncio.gather(*[_list_models(n, p) for n, p in PROVIDERS])

    print()


if __name__ == "__main__":
    asyncio.run(main())
