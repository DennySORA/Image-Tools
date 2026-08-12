# Repository Agent Instructions

> Keep this file limited to durable, repository-wide rules. Load a playbook only when its trigger applies.

## 1. Objective and sources of truth

Deliver the smallest coherent change that fully satisfies the request, preserves unrelated behavior, and is verified with the repository's actual tooling.

Use this precedence:

1. The user's explicit request and acceptance criteria.
2. The nearest applicable `AGENTS.md` or `AGENTS.override.md`.
3. The applicable playbook referenced below.
4. Source code, manifests, lockfiles, tests, workflows, and scripts.
5. README files and existing local conventions.

Source and executable checks override stale prose. Never invent a requirement, command result, platform result, security finding, release artifact, screenshot, commit, or compatibility claim.

## 2. Repository map

- `src/main.rs` — CLI entry points, top-level menu, and non-interactive subcommands.
- `src/core/` — shared configuration, errors, and guarded process execution.
- `src/features/` — one directory per product feature.
- `src/features/artifact_versions.rs` — reviewed exact AI-tool versions and immutable Skill source metadata.
- `src/ui/` — terminal output, prompts, and output sanitization.
- `src/i18n/` — translation keys and the four supported locales.
- `.github/workflows/` — CI, security, release, catalog refresh, and platform canaries.
- `.github/scripts/` — executable workflow logic and smoke-test fixtures.
- `docs/` — localized READMEs and scoped engineering playbooks.

## 3. Working method

- Inspect before editing: applicable instructions, `git status`, manifests, lockfiles, tests, workflows, and the actual execution path.
- Preserve unrelated and uncommitted work. Do not reset, checkout, stash, rebase, delete, or broadly rewrite files unless explicitly requested.
- Use a short plan for multi-file, cross-platform, release-sensitive, security-sensitive, or destructive work.
- Prefer repository evidence over memory. For version-sensitive APIs, inspect the reviewed or installed version and then use the official Context7 `find-docs` Skill when available, or official documentation, release notes, and registries.
- Ask only when a missing decision materially changes user-visible behavior, security posture, platform support, persisted state, or an irreversible operation and cannot be inferred safely.
- Edit sources of truth and regenerate derived data with repository scripts. Do not hand-edit generated outputs without a documented reason.
- Do not print, modify, or commit secrets. Never commit `.env*`, tokens, credentials, private keys, scanner raw secret output, or private local paths.
- Complete all unblocked work in the current task. Report exact blockers and residual risk instead of guessing.

## 4. On-demand playbooks

Read only the playbook whose trigger applies:

- `docs/codex/QUALITY_GATES.md` — non-trivial implementation, dependency/catalog changes, platform behavior, security work, release work, or commits.
- `docs/codex/PROCESS_EXECUTION.md` — subprocesses, privilege elevation, downloads, archives, installers, package managers, cancellation, timeouts, or rollback.
- `docs/codex/UI_UX.md` — terminal menus, prompts, progress, status/error output, localization, Unicode width, or accessibility.
- `docs/codex/SOCKETS_LOCAL.md` — only if an explicit request introduces a network listener or socket protocol. It is not part of normal Ops-Tools work.

If a referenced file is missing, continue from repository evidence and report the missing guidance. Never assume that an optional Skill, MCP server, browser, external CLI, or network service is installed.

## 5. Product boundaries

### Local interactive Rust CLI

- Ops-Tools is a Rust 2024 local CLI, not a Nuxt application, Python backend, web server, account system, or database service.
- Do not introduce HTTP, WebSocket, Socket.IO, CORS, browser state, login, OAuth application auth, JWT, sessions, RBAC, or a resident daemon without an explicit product-scope change and threat model.
- External-service OAuth used by a managed developer tool is not Ops-Tools application authentication. Never surface its credentials in generated browser-visible content or commit them.
- Windows can run supported CLI features, but the System Updater must remain capability-gated where host maintenance is unsupported.

### Safe system mutation

- System updates, package changes, reboots, cleanup, installer execution, filesystem publication, and configuration replacement are high-risk operations.
- Default to preview, exact scope, explicit confirmation, bounded execution, fail-closed validation, and actionable rollback information.
- Never execute real package-manager, CUDA/driver, reboot, prune, release-publish, or host-mutating commands in tests. Use fake executables, temporary roots, fixtures, and read-only diagnostics.
- Preserve WSL boundaries: never perform a guest `sudo reboot` or install a Linux display driver in WSL. Windows-side WSL lifecycle commands remain host operations.
- DGX support means DGX Spark/GB10 only unless the product scope and safety model are explicitly expanded.

## 6. Rust architecture and implementation

- Keep feature-specific policy and orchestration inside the owning `src/features/<feature>/` module.
- Put only genuinely shared infrastructure in `src/core/`; avoid a generic utility dumping ground.
- Reuse `src/core/process_runner.rs` for external processes. A direct `std::process::Command` call requires a narrow, documented reason and equivalent cancellation, timeout, output, and process-tree safety where applicable.
- Construct commands as program plus argument vector. Do not interpolate untrusted values into `sh -c`, PowerShell command strings, or other shell-evaluated text.
- Keep platform, package-manager, host-kind, architecture, and hardware capabilities explicit. Unsupported combinations must fail closed rather than silently selecting a nearby path.
- Model operation state and failure policy explicitly. Cleanup, reboot, publication, or dependent steps must not run after a prerequisite failure.
- Preserve error causes internally and return stable, actionable messages. Do not expose secrets or excessive local-path detail.
- Avoid `unsafe`; when unavoidable, state the invariant and cover the boundary with focused tests.
- Do not add a production dependency when the standard library or an existing dependency provides a small, maintainable solution.
- `Cargo.toml` and `Cargo.lock` are authoritative. Use exact or intentionally bounded versions consistent with repository policy and keep lockfile changes in the same change set.

## 7. AI tooling, MCP, and Skill lifecycle

Keep ownership boundaries strict:

- `artifact_versions.rs` owns reviewed exact runtime versions, immutable Git revisions, and reviewed digests.
- **AI Tool Upgrader** owns installation and upgrades of Claude Code, Codex, Context7 CLI, Playwright CLI, and Serena runtimes.
- **MCP Manager** owns MCP registration, scope, drift detection, replacement, and rollback; it does not own runtime upgrades.
- **Skill Installer** owns Skill/plugin installation, opt-in refresh, provenance, capability review, security gates, drift detection, and rollback; it must not silently upgrade an existing runtime.

Current stable integration boundaries:

- MCP catalog: Chrome DevTools and Serena.
- Playwright uses the official CLI plus Skill, not Playwright MCP.
- Context7 uses the CLI plus `find-docs` Skill.
- GitHub MCP and Sequential Thinking are not catalog entries.
- Claude MCP scopes are Local, Project shared, and User; Local is the safe default. Codex registration follows the client capability implemented in source.

Rules:

- Never use mutable `@latest`, branch names, or unverified archives for stable catalog entries.
- Fetch immutable versions/revisions, verify expected digests, extract to private staging, reject unsafe trees, run the Skill security gate, show capability/diff information, and publish atomically.
- Do not collapse runtime, registration, and Skill state into one lifecycle.
- Do not add a new MCP, Skill, plugin, hook, persistent service, network capability, or installer script without documenting trust, permissions, update policy, source immutability, verification, and rollback.
- Context7 queries leave the machine. Never send source code, credentials, vulnerability details, private package names, customer data, or other sensitive material.

## 8. Terminal UI and internationalization

- Route terminal presentation through `src/ui/` and user-visible strings through `src/i18n/`; do not scatter ad hoc styling and prompts.
- Keep English, Traditional Chinese, Simplified Chinese, and Japanese translations synchronized when user-visible behavior changes.
- Use Unicode display width, not byte or scalar count, for aligned terminal output.
- Every risky action must state what changes, where it changes, and whether rollback exists before confirmation.
- Distinguish success, partial success, failure, skipped, cancelled, timed out, drifted, unsupported, and update-available states in both text and color.
- Do not rely on color alone. Preserve readable output in narrow terminals and when ANSI styling is unavailable.
- Treat Ctrl-C, Escape/back, EOF, and prompt cancellation as normal controlled outcomes, not panics.

## 9. Validation

Start with the narrowest relevant checks, then expand according to blast radius. Use repository scripts rather than inventing alternatives.

Baseline:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-targets --all-features --locked
cargo check --all-targets --all-features --locked
```

Read `docs/codex/QUALITY_GATES.md` for subsystem-specific commands and evidence. Classify every relevant check as passed, failed, blocked, or not run, with the reason. Never claim a platform, hardware, security, or release result that was not executed.

## 10. Git, review, and completion

- Do not push, force-push, rewrite history, alter Git identity, or create a release unless explicitly requested.
- Commit only when requested. Use one coherent Conventional Commit per independently valid change set and stage only owned files or hunks.
- Review both the full diff and staged diff before committing.
- A task is complete when requested behavior is implemented, relevant checks have run, the final diff is intentional, and no known critical/high-severity defect remains.

### Code review rules

Prioritize findings that can cause:

- destructive host changes, incorrect privilege use, or unsafe reboot/cleanup sequencing;
- platform misclassification, especially WSL, Windows, Linux ARM64, and DGX Spark/GB10;
- mutable or unverified third-party execution, archive traversal, digest bypass, or secret disclosure;
- lifecycle ownership drift between the AI Tool Upgrader, MCP Manager, and Skill Installer;
- unbounded or uncancellable subprocesses, shell injection, leaked child processes, or unsafe output handling;
- tests that touch the real host, network credentials, package manager, release service, or user configuration;
- documentation that overstates supported platforms, hardware certification, security coverage, or executed validation.
