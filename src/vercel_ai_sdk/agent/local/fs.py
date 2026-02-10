from __future__ import annotations

import asyncio
import pathlib
from dataclasses import dataclass, field

from .. import proto


@dataclass
class LocalFilesystem(proto.Filesystem):
    """Filesystem backed by the local disk rooted at *cwd*."""

    cwd: pathlib.Path = field(default_factory=pathlib.Path.cwd)

    def _resolve(self, path: str) -> pathlib.Path:
        p = pathlib.Path(path)
        if p.is_absolute():
            return p
        return self.cwd / p

    # -- read ----------------------------------------------------------------

    async def read(
        self, path: str, *, offset: int | None = None, limit: int | None = None
    ) -> str:
        target = self._resolve(path)
        text = target.read_text()
        lines = text.splitlines(keepends=True)

        total = len(lines)
        start = offset or 0
        end = start + limit if limit is not None else total
        selected = lines[start:end]

        header = f"[{total} lines | showing {start + 1}-{min(end, total)}]\n"
        numbered = "".join(
            f"{start + i + 1:>5}\t{line}" for i, line in enumerate(selected)
        )
        return header + numbered

    # -- write ---------------------------------------------------------------

    async def write(self, path: str, content: str) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    # -- edit ----------------------------------------------------------------

    async def edit(self, path: str, old: str, new: str) -> str:
        target = self._resolve(path)
        content = target.read_text()

        count = content.count(old)
        if count == 0:
            raise ValueError("old_string not found in file")
        if count > 1:
            raise ValueError(
                f"old_string appears {count} times (must be unique). "
                "Include more surrounding context."
            )

        updated = content.replace(old, new, 1)
        target.write_text(updated)
        return updated

    # -- ls ------------------------------------------------------------------

    async def ls(
        self,
        path: str = ".",
        *,
        depth: int | None = None,
        pattern: str | None = None,
        include_hidden: bool = False,
    ) -> str:
        args = ["find", str(self._resolve(path))]
        if depth is not None:
            args += ["-maxdepth", str(depth)]
        if not include_hidden:
            args += ["!", "-path", "*/.*"]
        if pattern:
            args += ["-name", pattern]
        args.append("-print")

        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        return stdout.decode().strip()

    # -- glob ----------------------------------------------------------------

    async def glob(self, pattern: str, *, path: str | None = None) -> list[str]:
        root = self._resolve(path or ".")
        return sorted(str(p.relative_to(root)) for p in root.glob(pattern))

    # -- grep ----------------------------------------------------------------

    async def grep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        include: str | None = None,
        context_lines: int | None = None,
        max_count: int | None = None,
        case_sensitive: bool = True,
    ) -> str:
        args = [
            "rg",
            "--line-number",
            "--heading",
            "--color",
            "never",
        ]
        if not case_sensitive:
            args.append("-i")
        if include:
            args += ["--glob", include]
        if context_lines is not None:
            args += ["-C", str(context_lines)]
        if max_count is not None:
            args += ["--max-count", str(max_count)]
        args += ["--", pattern, str(self._resolve(path or "."))]

        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        result = stdout.decode()

        MAX_OUTPUT = 50_000
        if len(result) > MAX_OUTPUT:
            result = (
                result[:MAX_OUTPUT]
                + "\n\n[Output truncated — use a more specific pattern or path]"
            )
        return result.strip() or "(no matches found)"

    # -- bash ----------------------------------------------------------------

    async def bash(self, command: str, *, timeout: int | None = None) -> str:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self.cwd,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise TimeoutError(
                f"Command timed out after {timeout}s. "
                "Try increasing the timeout or breaking the command into smaller steps."
            )

        output = stdout.decode() if stdout else ""
        if proc.returncode != 0:
            return f"[exit code {proc.returncode}]\n{output}"
        return output
