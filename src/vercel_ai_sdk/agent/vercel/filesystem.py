from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from vercel.sandbox import AsyncSandbox
from vercel.sandbox.models import WriteFile

from .. import proto

logger = logging.getLogger(__name__)

HOME_DIR = "/home/vercel-sandbox"
MAX_TIMEOUT_MS = 5 * 60 * 60 * 1000  # 5 hours
MAX_OUTPUT = 50_000
CWD_MARKER = "___VERCEL_CWD___"


class SandboxError(Exception):
    """Raised when a sandbox operation fails."""


class SandboxGoneError(SandboxError):
    """Raised when the sandbox VM is no longer available (410/422)."""


def _is_gone_error(exc: BaseException) -> bool:
    """
    Detect stale/terminated sandbox errors.
    Mirrors isSandboxGoneError from agent-sdk.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code in (410, 422):
            return True

    msg = str(exc)
    if "Expected a stream of command data" in msg:
        return True
    if "Expected a stream of logs" in msg:
        return True

    return False


@dataclass
class VercelSandbox:
    """
    Filesystem implementation backed by a Vercel remote sandbox VM.

    Implements the proto.Filesystem protocol. All file and shell operations
    are delegated to a remote Vercel sandbox via the vercel SDK.

    Lifecycle mirrors the agent-sdk pattern: lazy creation on first use,
    cached sandbox instance, automatic recovery from stale VMs, and
    snapshot/restore support.
    """

    # --- Config ---
    vcpus: int = 2
    ports: list[int] | None = None
    timeout_ms: int = MAX_TIMEOUT_MS
    snapshot_id: str | None = None
    auto_start: bool = True

    # --- Credentials (None = resolve from VERCEL_TOKEN / OIDC env vars) ---
    token: str | None = None
    project_id: str | None = None
    team_id: str | None = None

    # --- Internal state (not part of the public interface) ---
    _sandbox: AsyncSandbox | None = field(default=None, init=False, repr=False)
    _sandbox_task: asyncio.Task[AsyncSandbox] | None = field(
        default=None, init=False, repr=False
    )
    _cwd: str = field(default=HOME_DIR, init=False, repr=False)
    _recovered_from_stale: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.auto_start:
            self._sandbox_task = asyncio.ensure_future(self._create_sandbox())

    # =========================================================================
    # Sandbox lifecycle
    # =========================================================================

    async def _create_sandbox(self) -> AsyncSandbox:
        """Create or restore a sandbox VM."""
        creds: dict[str, Any] = {}
        if self.token:
            creds["token"] = self.token
        if self.project_id:
            creds["project_id"] = self.project_id
        if self.team_id:
            creds["team_id"] = self.team_id

        create_kwargs: dict[str, Any] = {
            "resources": {"vcpus": self.vcpus},
            "timeout": self.timeout_ms,
            **creds,
        }
        if self.ports:
            create_kwargs["ports"] = self.ports
        if self.snapshot_id and not self._recovered_from_stale:
            create_kwargs["source"] = {
                "type": "snapshot",
                "snapshot_id": self.snapshot_id,
            }

        sandbox = await AsyncSandbox.create(**create_kwargs)
        logger.info(
            "sandbox created: %s (status=%s)", sandbox.sandbox_id, sandbox.status
        )
        return sandbox

    async def _ensure_sandbox(self) -> AsyncSandbox:
        """
        Get the running sandbox instance, creating it lazily if needed.
        Deduplicates concurrent callers — only one creation flight at a time.
        """
        if self._sandbox is not None:
            return self._sandbox

        if self._sandbox_task is None:
            self._sandbox_task = asyncio.ensure_future(self._create_sandbox())

        self._sandbox = await self._sandbox_task
        self._sandbox_task = None
        return self._sandbox

    async def _recover_from_stale(self) -> None:
        """
        Clear the cached sandbox so the next _ensure_sandbox() creates a new one.
        If we had a snapshot, the new sandbox will restore from it.
        """
        logger.warning("sandbox gone — recovering (snapshot_id=%s)", self.snapshot_id)
        self._sandbox = None
        self._sandbox_task = None
        self._recovered_from_stale = True

    async def start(self) -> str:
        """Ensure the sandbox is running. Returns the sandbox status."""
        sb = await self._ensure_sandbox()
        return sb.status

    async def stop(self) -> None:
        """Stop the sandbox VM."""
        if self._sandbox is None:
            return
        try:
            await self._sandbox.stop()
        except Exception:
            pass
        self._sandbox = None
        self._sandbox_task = None

    async def snapshot(self) -> str:
        """
        Snapshot the sandbox. Stops the VM as a side-effect.
        Returns the snapshot ID for later restoration.
        """
        sb = await self._ensure_sandbox()
        snap = await sb.snapshot()
        snap_id = snap.snapshot_id
        self.snapshot_id = snap_id
        self._sandbox = None
        self._sandbox_task = None
        logger.info("sandbox snapshotted: %s", snap_id)
        return snap_id

    async def get_status(self) -> str:
        sb = await self._ensure_sandbox()
        return sb.status

    async def get_domain(self, port: int) -> str:
        sb = await self._ensure_sandbox()
        return sb.domain(port)

    async def extend_timeout(self, duration_ms: int) -> None:
        sb = await self._ensure_sandbox()
        await sb.extend_timeout(duration_ms)

    # =========================================================================
    # Context manager — stops sandbox on exit
    # =========================================================================

    async def __aenter__(self) -> VercelSandbox:
        await self._ensure_sandbox()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.stop()

    # =========================================================================
    # Internal: run commands with stale-sandbox recovery
    # =========================================================================

    async def _run_command(
        self,
        cmd: str,
        args: list[str] | None = None,
        *,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> tuple[str, str, int]:
        """
        Run a command in the sandbox. Returns (stdout, stderr, exit_code).
        Automatically retries once if the sandbox is gone.
        """
        try:
            return await self._do_run_command(cmd, args, cwd=cwd, timeout=timeout)
        except SandboxGoneError:
            await self._recover_from_stale()
            return await self._do_run_command(cmd, args, cwd=cwd, timeout=timeout)

    async def _do_run_command(
        self,
        cmd: str,
        args: list[str] | None = None,
        *,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> tuple[str, str, int]:
        sb = await self._ensure_sandbox()
        effective_cwd = cwd or self._cwd

        try:
            finished = await asyncio.wait_for(
                sb.run_command(cmd, args or [], cwd=effective_cwd),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Command timed out after {timeout}s. "
                "Try increasing the timeout or breaking the command into smaller steps."
            )
        except Exception as exc:
            if _is_gone_error(exc):
                raise SandboxGoneError(str(exc)) from exc
            raise SandboxError(str(exc)) from exc

        stdout = await finished.stdout()
        stderr = await finished.stderr()
        return stdout, stderr, finished.exit_code

    def _resolve_path(self, path: str) -> str:
        """Resolve a relative path against the sandbox CWD."""
        if path.startswith("/"):
            return path
        return f"{self._cwd}/{path}"

    # =========================================================================
    # Filesystem protocol implementation
    # =========================================================================

    async def read(
        self, path: str, *, offset: int | None = None, limit: int | None = None
    ) -> str:
        sb = await self._ensure_sandbox()
        resolved = self._resolve_path(path)

        raw = await sb.read_file(resolved)
        if raw is None:
            raise FileNotFoundError(f"File not found: {path}")

        text = raw.decode()
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

    async def write(self, path: str, content: str) -> None:
        sb = await self._ensure_sandbox()
        resolved = self._resolve_path(path)

        parent = "/".join(resolved.split("/")[:-1])
        if parent:
            await sb.mk_dir(parent)

        files: list[WriteFile] = [{"path": resolved, "content": content.encode()}]
        await sb.write_files(files)

    async def edit(self, path: str, old: str, new: str) -> str:
        sb = await self._ensure_sandbox()
        resolved = self._resolve_path(path)

        raw = await sb.read_file(resolved)
        if raw is None:
            raise FileNotFoundError(f"File not found: {path}")

        content = raw.decode()
        count = content.count(old)
        if count == 0:
            raise ValueError("old_string not found in file")
        if count > 1:
            raise ValueError(
                f"old_string appears {count} times (must be unique). "
                "Include more surrounding context."
            )

        updated = content.replace(old, new, 1)
        files: list[WriteFile] = [{"path": resolved, "content": updated.encode()}]
        await sb.write_files(files)
        return updated

    async def ls(
        self,
        path: str = ".",
        *,
        depth: int | None = None,
        pattern: str | None = None,
        include_hidden: bool = False,
    ) -> str:
        resolved = self._resolve_path(path)
        args = [resolved]
        if depth is not None:
            args += ["-maxdepth", str(depth)]
        if not include_hidden:
            args += ["!", "-path", "*/.*"]
        if pattern:
            args += ["-name", pattern]
        args.append("-print")

        stdout, _, _ = await self._run_command("find", args)
        return stdout.strip()

    async def glob(self, pattern: str, *, path: str | None = None) -> list[str]:
        root = self._resolve_path(path or ".")
        cmd = f"cd {_shell_quote(root)} && find . -path {_shell_quote(f'./{pattern}')} 2>/dev/null | sort"
        stdout, _, _ = await self._run_command("bash", ["-c", cmd])
        if not stdout.strip():
            return []
        return [
            line.removeprefix("./")
            for line in stdout.strip().splitlines()
            if line.strip()
        ]

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
        args = ["--line-number", "--heading", "--color", "never"]
        if not case_sensitive:
            args.append("-i")
        if include:
            args += ["--glob", include]
        if context_lines is not None:
            args += ["-C", str(context_lines)]
        if max_count is not None:
            args += ["--max-count", str(max_count)]
        args += ["--", pattern, self._resolve_path(path or ".")]

        stdout, _, _ = await self._run_command("rg", args)
        result = stdout
        if len(result) > MAX_OUTPUT:
            result = (
                result[:MAX_OUTPUT]
                + "\n\n[Output truncated — use a more specific pattern or path]"
            )
        return result.strip() or "(no matches found)"

    async def bash(self, command: str, *, timeout: int | None = None) -> str:
        """
        Run a bash command with CWD persistence between calls.

        After the user's command finishes, we capture the new working directory
        so that subsequent bash calls start in the right place (e.g. after `cd`).
        """
        wrapped = f"{{ {command} ; }} 2>&1 ; echo '{CWD_MARKER}' ; pwd"

        stdout, _, exit_code = await self._run_command(
            "bash", ["-c", wrapped], cwd=self._cwd, timeout=timeout
        )

        if CWD_MARKER in stdout:
            parts = stdout.rsplit(CWD_MARKER, 1)
            output = parts[0]
            new_cwd = parts[1].strip()
            if new_cwd:
                self._cwd = new_cwd
        else:
            output = stdout

        if exit_code != 0:
            return f"[exit code {exit_code}]\n{output}"
        return output


def _shell_quote(s: str) -> str:
    """Quote a string for safe use in a shell command."""
    return "'" + s.replace("'", "'\\''") + "'"
