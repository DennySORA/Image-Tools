# Documentation Instructions

These instructions apply under `docs/`. The repository-root `AGENTS.md` remains authoritative for product, architecture, safety, and validation rules.

## Documentation purpose

Keep documentation exact, scoped, and synchronized with executable source. Do not preserve stale full-stack, browser, Python, Nuxt, authentication, or socket guidance in this Rust CLI repository.

## Sources of truth

For technical claims, use this order:

1. User-requested behavior and acceptance criteria.
2. Source code and tests.
3. `Cargo.toml`, `Cargo.lock`, configuration schemas, and reviewed artifact catalog.
4. GitHub workflow and repository scripts.
5. Existing README/playbook prose.

Do not copy a version, catalog entry, target count, scanner flag, platform boundary, or command from memory. Reference the owning source rather than duplicating volatile constants when possible.

## Localized READMEs

The repository maintains:

- `README.md`;
- `docs/README_zh-TW.md`;
- `docs/README_zh-CN.md`;
- `docs/README_ja.md`.

When user-visible capability, platform support, safety behavior, command syntax, installation, MCP/Skill catalog, or release verification changes, update all applicable localized files in the same change or state why a translation update is intentionally deferred.

Keep package names, CLI commands, paths, version strings, MCP names, and protocol identifiers exact. Translate prose, not machine identifiers.

## Playbooks

- Keep the root instructions concise and move task-specific procedures into `docs/codex/`.
- Every playbook must state when it should be read.
- Remove obsolete guidance rather than retaining contradictory “legacy” sections.
- Do not load or reference `SOCKETS_LOCAL.md` for ordinary work; it is a guard for a future explicit listener proposal.
- Keep terminal UI guidance in `UI_UX.md` and process/installer guidance in `PROCESS_EXECUTION.md`.
- Avoid duplicating the full validation matrix across files.

## Claims and evidence

Distinguish:

- supported by source;
- exercised in generic CI;
- cross-built/emulated;
- tested by fake-backed safety fixtures;
- verified on the current host;
- verified on physical hardware;
- not yet verified.

For example, a Linux ARM64 QEMU smoke test is not physical DGX Spark certification. A fake-backed updater test is not a real package upgrade. An inventory snapshot is not a filesystem restore point.

Never claim a scan is comprehensive, a release is published, or a workflow is green without current evidence.

## Formatting and validation

- Use repository-relative links and paths.
- Keep headings and terminology consistent.
- Use fenced code blocks with an appropriate language when useful.
- Keep commands directly executable; do not include shell prompts.
- Avoid raw HTML unless Markdown cannot express the requirement.
- Run:

```bash
git diff --check
```

- Verify every touched link/path and compare commands against the owning workflow/script.
- Review the final prose for contradictions with root `AGENTS.md`, source, and localized READMEs.

Do not create or commit generated screenshots, logs, scanner reports, archives, or build outputs unless the task explicitly requires them.
