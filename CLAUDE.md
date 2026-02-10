## Current task

Implement `agent` module.

Filesystem tools (local and sandbox)

1. read
2. write / search-replace
3. ls / glob
4. grep
5. bash

Durability

1. state tracking
2. durable execution

Agent features

1. skills
2. web tools / browser
3. memory management
4. sub-agents

## References

See `.reference`

### agent-sdk — Vercel Agent SDK (TypeScript, primary inspiration)
Opinionated sandboxed coding agent framework. LLM loop + sandbox + tools + session persistence. Biased towards Next.js/Vercel, but the tool design and sandbox interface are the reference spec.
- 7 built-in tools: Read, Write, Edit, List, Grep, Bash, JavaScript (meta-tool that orchestrates other tools via code)
- Sandbox abstraction with two bindings (local dev, Vercel VM) — exec, writeFiles, lifecycle (start/stop/snapshot)
- Process manager — persistent CWD per session, background process support
- Skills — SKILL.md discovery with YAML frontmatter, progressive disclosure
- Durable sessions with send/stream/interrupt
- Storage backends (local filesystem, Vercel Postgres, custom HTTP)
- Prompt caching (Anthropic/OpenAI automatic cache breakpoints)
- No web/search tools, no sub-agents, no memory, no context management

### deepagents — LangChain Deep Agents (Python, patterns to draw from)
Ready-to-run agent harness built on LangGraph. Middleware stack architecture where each capability is a composable layer. Most feature-complete of the three.
- 7 filesystem tools: ls, read_file, write_file, edit_file, glob, grep, execute
- Backend protocol ABC with pluggable impls (in-memory state, local filesystem, local shell, LangGraph store, composite routing)
- Sub-agent spawning via `task` tool with isolated context windows
- Auto-summarization when context hits 85% of window, evicts history to file
- Large result eviction — results >20k tokens written to file, replaced with preview
- Memory — AGENTS.md files injected into system prompt, self-modifiable by agent
- Skills — SKILL.md progressive disclosure (same pattern as Vercel)
- Patch dangling tool calls from interrupted sessions
- Web tools (CLI only): web_search (Tavily), fetch_url (HTML to markdown), http_request

### pi — Python Intelligence (terminal coding agent)
Terminal agent (Rust PTY + Python). Uses pydantic-ai. Local filesystem tools are pure pathlib, trivially portable. Shell tools depend on Pi's Rust binary (not portable). Key portable code:
- list_files, read_file, read_chunk — pure pathlib
- search_replace — exact match + rapidfuzz fuzzy fallback (threshold 80%)
- rewrite — write + mkdir + difflib diff
- exec (raw) — `asyncio.create_subprocess_exec`
- `@suppress_errors` decorator
- edit_lock (`asyncio.Lock` for concurrent write safety)

### riff — Code Generation Agent (web app)
Same tool patterns as Pi but targeting remote Daytona sandboxes. Uses pydantic-ai. Key portable code beyond what Pi has:
- grep — builds ripgrep command with flags
- tree — directory structure with exclude patterns
- lint() after edit — ruff for Python, biome for TS
- TodoList/Todo/TodoStatus pydantic models
- add_todos, mark_todos, todo_status tools
- `@suppress_errors` with recursive timeout detection (prefer over Pi's)
- repair_stray_tool_calls — patches dangling tool calls

