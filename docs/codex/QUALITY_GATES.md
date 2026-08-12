# Engineering and Validation Playbook

Read this file for every non-trivial implementation, dependency/catalog change, platform-behavior change, security change, release-sensitive change, or commit.

## 1. Repository-first preparation

Before editing:

- read the applicable agent instructions and playbooks;
- inspect `git status`, the current branch, manifests, lockfiles, workflows, scripts, tests, and relevant execution paths;
- identify unrelated user changes and preserve them;
- map the affected feature, shared core, UI, i18n, platform, persistence, external-process, and release paths;
- inspect existing patterns before adding abstractions or dependencies;
- identify which claims can be verified on the current host and which require CI, emulation, or physical hardware.

For dependency or catalog changes, verify official release/registry metadata, runtime compatibility, migration notes, immutable source identity, and lockfile integrity. Use the repository's reviewed catalog model rather than introducing floating versions.

## 2. Baseline Rust quality gate

The local baseline mirrors CI:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-targets --all-features --locked
cargo check --all-targets --all-features --locked
```

Run the narrowest focused test first while iterating, then the full applicable baseline before completion. Do not omit `--locked` from validation intended to represent CI or release behavior.

When dependencies change:

```bash
cargo check --all-targets --all-features --locked
cargo test --all-targets --all-features --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
```

Inspect `Cargo.toml` and `Cargo.lock` together and explain every intentional dependency change.

## 3. Risk-based validation matrix

### Documentation-only changes

- Verify paths, command examples, feature names, platform claims, and cross-document links against source/workflows.
- Run:

```bash
git diff --check
```

- If user-visible capabilities changed, update the applicable localized READMEs or report why they are intentionally unchanged.
- Do not claim a command, workflow, release asset, platform test, or hardware canary result that was not executed.

### Terminal UI or i18n changes

- Run focused tests for `src/ui`, prompt selection/cancellation, formatting, sanitization, and affected feature presentation.
- Run the full baseline.
- Exercise applicable menu/prompt paths manually in a real terminal when possible.
- Check English, Traditional Chinese, Simplified Chinese, and Japanese output for missing keys, overflow, ambiguous confirmation, and incorrect Unicode alignment.
- Follow `docs/codex/UI_UX.md`.

### Shared process execution changes

Changes to `src/core/process_runner.rs` or process wrappers require tests for:

- captured and inherited stdio;
- timeout and cooperative cancellation;
- process-tree termination;
- output truncation and incomplete-reader handling;
- child/descendant cleanup;
- platform-specific behavior on Unix and Windows where applicable;
- freshly written executable behavior and error propagation.

Run the full baseline and follow `docs/codex/PROCESS_EXECUTION.md`.

### System Updater changes

Test, as applicable:

- OS, Linux-family, host-kind, architecture, hardware, CUDA, and package-manager detection;
- supported, unsupported, and ambiguous platform combinations;
- dry-run versus apply behavior;
- safe/default/full/aggressive profile boundaries;
- dependency-aware step skipping and `StopGroup`/`StopWorkflow` behavior;
- privilege preflight and no-sudo paths;
- WSL guest safeguards;
- DGX Spark/GB10 package provenance and version pinning;
- cleanup and reboot prerequisites;
- report/lock/config paths;
- failed/partial/cancelled/timed-out summary accuracy.

Use fake-backed platform tests; never mutate the real host from an automated test. When relevant:

```bash
bash .github/scripts/test-platform-safety.sh
cargo run --locked -- system-platform-diagnostic --format json
```

The diagnostic is read-only, but its result validates only the current machine. A physical DGX Spark claim requires a recorded canary run.

### AI Tool Upgrader changes

Verify:

- exact reviewed target version resolution;
- missing/current/upgrade-available/newer-than-catalog/invalid-version states;
- no silent downgrade of a newer runtime;
- npm/uv command construction and post-install version verification;
- separation from MCP registration and Skill installation;
- cancellation, timeout, partial failure, and retry behavior.

Use fake `npm`, `uv`, and tool executables. Do not install or upgrade the developer's real global tools during tests.

### MCP Manager changes

Verify both Claude and Codex paths as applicable:

- effective registration normalization;
- command, argument, environment, and scope drift;
- Chrome headed/headless equivalence rules;
- Claude Local, Project shared, and User scope semantics;
- Codex registration behavior;
- exact reviewed runtime command;
- replacement rollback;
- refusal when an old registration contains unrestorable fields;
- no GitHub, Playwright, or Sequential Thinking MCP regression unless the product decision explicitly changes.

Use temporary fake homes/configs and fake CLIs. Do not modify the developer's real `~/.claude.json`, `.mcp.json`, or `~/.codex/config.toml`.

### Skill Installer changes

Verify:

- catalog ID and install-name uniqueness per client/scope;
- immutable full Git revision and reviewed archive digest;
- staging permissions and structural validation;
- traversal, symlink, special-file, executable, hook, network, subprocess, dependency-change, and persistent-service policy;
- first install, current, update available, skipped, drifted, untracked, refresh, and rollback states;
- provenance lock separation by client, scope, and install root;
- diff bounds and high-risk path classification;
- no silent runtime upgrade;
- atomic publication and previous-version recovery.

Tests must use temporary install/state roots and fixture archives. Do not touch real agent Skill/plugin directories.

### Security Scanner changes

Verify:

- scanner version/capability validation;
- history, working-tree, and bounded ignored-secret snapshots;
- secret output redaction on stdout and stderr;
- timeout, output limits, cancellation, and process-tree termination;
- Trivy `vuln,misconfig` behavior and license scan;
- exact Semgrep rules/version behavior;
- built-in manifest/lockfile analyzer exit-code contract;
- no mutable installer or `@latest` fallback;
- fail-closed behavior when provenance or a required capability is missing.

Run focused fixtures before full scans. Release-sensitive changes should also exercise repository scripts such as:

```bash
bash .github/scripts/run-semgrep.sh .
bash .github/scripts/run-trufflehog.sh .
cargo run --quiet --locked -- security-supply-chain --format json .
```

These may require container/network/tool availability; classify unavailable prerequisites as blocked rather than passing.

### Rust Upgrader changes

Verify all four modes independently:

- Toolchain;
- Cargo-installed tools;
- compatible project dependencies;
- breaking project dependencies.

Cover clean/dirty Git state, optional backup branch/ref behavior, pinned Cargo helper installation with `--locked`, command failure, test failure, cancellation, and rollback guidance. A mode that does not require a backup must not be mistaken for a failed prerequisite.

### Rust Builder changes

Verify:

- target metadata and host/builder capability;
- required target, linker, sysroot, container runtime, or emulator;
- Cargo versus Cross command construction;
- actual emitted artifacts;
- output architecture where verifiable;
- native or emulated `--version` smoke test where supported;
- clear `build-only verified` status when execution cannot be tested.

For the release target helper:

```bash
bash .github/scripts/build-target.sh <target> release
```

Do not generalize one host-target result to every advertised target.

### FFmpeg Installer changes

Verify:

- supported Linux/package-manager boundary;
- immutable or explicitly reviewed source revision policy;
- dependency preflight;
- CPU-only and optional NVENC capability paths;
- staging prefix and install publication;
- `ffmpeg`/`ffprobe`, encoder, and 10-bit format checks;
- symlink replacement and rollback;
- cancellation and failed-build cleanup;
- nonfree/GPL disclosure.

Do not build or install into the developer's real `~/ffbuild` or `~/.local/bin` during automated tests.

### CI, installer, catalog automation, or release changes

Run the applicable repository scripts:

```bash
bash .github/scripts/test-install-platforms.sh
bash .github/scripts/test-publish-release.sh
python3 .github/scripts/check-action-pins.py
```

For catalog automation, test:

- registry-specific version ordering;
- immutable version/revision updates;
- hook command changes, executable content/mode changes, and entry-point target changes;
- sdist and applicable wheel analysis;
- archive entry/count/size/path/special-file limits;
- safe branch update behavior;
- manual-review escalation for new capabilities.

For release changes, verify tag/version/main ancestry, release gates, checksums, archive contents, SBOM, provenance/attestation inputs, and Linux ARM64 build plus QEMU smoke behavior. Never create a tag or publish a release unless explicitly requested.

## 4. Test design

- Test externally observable behavior and critical invariants, not implementation trivia.
- A bug fix should include a deterministic failing regression test first when practical.
- Unit tests must not call real external services or mutate real global/user state.
- Use fake executables, temporary directories, fixture archives, fake HOME/config/state roots, and local protocol doubles at the correct boundary.
- Cover success, expected failure, malformed/empty/boundary input, timeout, cancellation, truncation, rollback, partial completion, idempotency, and cleanup as applicable.
- Keep tests deterministic: no mutable upstream branches, live registries, current clock dependence without injection, or host-specific assumptions without a capability gate.
- Never weaken a test, validation, security control, or error path merely to make CI pass.

## 5. Diff and commit gate

After each coherent change set:

1. Run applicable focused checks.
2. Run the required broader matrix.
3. Inspect runtime/terminal behavior where applicable.
4. Inspect `git diff --check`, the full diff, and the staged diff.
5. Remove debug code, secrets, temporary archives, scanner outputs, caches, target artifacts, and unintended generated files.
6. Stage only owned files or hunks.
7. Commit only after validation passes and only when requested.

Conventional Commit:

```text
<type>(<scope>): <imperative summary>
```

Use an English imperative title, preferably at most 50 and never more than 72 characters, with no trailing period. A non-trivial body should state what changed, why, exact tests, and dependency/version decisions. Use `BREAKING CHANGE:` only for an intentional documented migration.

Do not push, force-push, rewrite history, alter Git identity, tag, or publish unless explicitly requested.

## 6. Evidence classification and final report

Classify every relevant check as:

- **passed** — command executed and met its acceptance criteria;
- **failed** — command executed and did not meet its acceptance criteria;
- **blocked** — an explicit external condition prevented execution;
- **not run** — intentionally omitted, with the reason;
- **not applicable** — outside the change's blast radius.

Never convert a warning, skipped test, missing artifact, unavailable tool, or unexecuted platform into a pass.

The final report should contain only relevant sections:

1. What changed and why.
2. Exact commands and outcomes.
3. Important platform, security, compatibility, or lifecycle decisions.
4. Blockers, checks not run, and residual risk.
5. Commit hash/message, only if a commit was requested and created.

Confidence must be proportional to executed evidence. Do not claim universal or unverified 100% confidence.
