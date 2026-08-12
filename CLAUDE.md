# Claude Code Project Instructions

@AGENTS.md

## Claude Code-specific guidance

- `AGENTS.md` is the canonical shared instruction file for both Claude Code and Codex. Do not duplicate its repository rules here.
- Load only the playbook triggered by the current task. Keep long procedures in playbooks or Skills rather than expanding this always-loaded file.
- Treat existing user and project Claude configuration as private state. Do not modify `~/.claude.json`, `~/.claude/`, project `.mcp.json`, plugins, hooks, Skills, or permissions unless the task explicitly requests that configuration change.
- For tests of MCP or Skill behavior, use fake CLIs and temporary HOME/config/state roots; never exercise the developer's real Claude installation.
- Use the reviewed Context7 `find-docs` Skill for version-sensitive library documentation when available. Do not send source code, credentials, vulnerability details, private package names, customer data, or other sensitive content to Context7.
- Use Serena only when symbol-level navigation or cross-file reference analysis materially helps. Do not assume it is installed or registered.
- Playwright CLI/Skill and Chrome DevTools MCP are relevant only if the task introduces or validates an actual browser surface; Ops-Tools itself is currently a terminal CLI.
- Do not add or recommend Sequential Thinking, Playwright MCP, or GitHub MCP as Ops-Tools catalog entries unless an explicit product review changes that boundary.
- Claude MCP registration defaults to Local scope. Project-shared `.mcp.json`, User scope, hooks, plugins, network capabilities, and persistent services require explicit intent and review.
- Before claiming completion, report exact commands run and distinguish passed, failed, blocked, not run, and not applicable checks.
