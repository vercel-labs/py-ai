from typing import Protocol


class Filesystem(Protocol):
    """Abstraction over a working environment."""

    async def read(
        self, path: str, *, offset: int | None = None, limit: int | None = None
    ) -> str: ...

    async def write(self, path: str, content: str) -> None: ...

    async def edit(self, path: str, old: str, new: str) -> str:
        """Replace *old* with *new* in file. Returns the updated content."""
        ...

    async def ls(
        self,
        path: str = ".",
        *,
        depth: int | None = None,
        pattern: str | None = None,
        include_hidden: bool = False,
    ) -> str: ...

    async def glob(self, pattern: str, *, path: str | None = None) -> list[str]: ...

    async def grep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        include: str | None = None,
        context_lines: int | None = None,
        max_count: int | None = None,
        case_sensitive: bool = True,
    ) -> str: ...

    async def bash(self, command: str, *, timeout: int | None = None) -> str: ...
