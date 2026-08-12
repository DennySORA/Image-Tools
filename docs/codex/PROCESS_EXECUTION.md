# External Process, Privilege, and Installer Playbook

Read this file before changing subprocess execution, privilege elevation, downloads, archives, installers, package managers, runtime updates, cancellation, timeouts, or rollback.

## 1. Shared execution boundary

`src/core/process_runner.rs` is the repository-wide external-process execution boundary. It provides process-group isolation, optional timeout, cooperative cancellation, bounded output capture, inherited interactive stdio, and process-tree termination.

- Reuse the shared runner instead of introducing another polling, timeout, capture, or kill implementation.
- A direct `std::process::Command` call is acceptable only for a narrow bootstrap/platform primitive that cannot use the shared runner. Document the reason and preserve equivalent safety where relevant.
- Never use shell interpolation for untrusted or externally derived values. Build a program plus explicit argument vector.
- Avoid `sh -c`, `bash -c`, `cmd /C`, or PowerShell command strings. When a shell is genuinely required, keep the script static and pass values through positional arguments or environment variables after validation.

## 2. Process policy

Choose policy intentionally:

- **Captured/non-interactive:** stdin null, bounded stdout/stderr where output volume is not strictly controlled.
- **Interactive:** inherited stdio only when the child must display and receive the exact package-manager/tool confirmation.
- **Timeout:** required for network calls, scanners, installers, package metadata refresh, and commands that may hang.
- **Cancellation:** propagate the operation's shared token and treat Ctrl-C as controlled cancellation.
- **Termination:** timeout, cancellation, guard drop, and output-drain failure must not leave descendants running.
- **Output:** preserve enough stderr/stdout for diagnosis, mark truncation, and redact before persistence or display.

Do not report a timeout/cancellation as a normal non-zero exit. Keep interruption, exit status, truncation, and reader completeness distinguishable.

## 3. Argument, path, and environment safety

- Validate enum-like arguments against an allowlist.
- Treat filesystem paths as data, not command syntax.
- Use canonicalization only when its symlink semantics are correct for the operation.
- Reject traversal, absolute archive entries, unexpected symlinks, device files, FIFOs, sockets, and other special files.
- Apply file-count, total-expanded-size, per-file-size, and output-size limits to untrusted archives/content.
- Pass the minimum environment needed. Do not inherit or print secrets unnecessarily.
- Redact tokens, credentials, auth headers, secret-prone environment variables, and sensitive scanner output on both stdout and stderr.

## 4. Privilege model

Privilege is a per-step requirement, not a workflow-wide assumption.

Model operations as appropriate:

- no elevation;
- `sudo` required;
- root only;
- host-side bridge required;
- unsupported.

Rules:

- Do not call `sudo -v` unless at least one selected step requires it.
- Never concatenate user-controlled text into a privileged shell command.
- Revalidate paths, package names, versions, repositories, and digests before the privileged boundary.
- WSL guest code must not perform Windows-host lifecycle work or guest reboot.
- A container/rootless environment must not be treated as native root merely because a command exists.
- Tests must never invoke the real `sudo`, package manager, reboot, driver installer, or service manager.

## 5. Download and provenance

For executable code, installers, MCP/Skill sources, scanner binaries, and release assets:

1. Use HTTPS and an exact reviewed version/revision.
2. Reject mutable `latest`, branch, or unversioned stable URLs unless the task explicitly changes policy and documents the risk.
3. Obtain a digest or signature from an independently authenticated source.
4. Download to a private temporary location with time and size limits.
5. Verify digest/signature before parsing or executing.
6. Inspect archive structure before extraction.
7. Extract into private staging.
8. Run structural and security validation.
9. Publish atomically.
10. Preserve only the bounded rollback material required by the feature.

A successful HTTP status, plausible filename, shebang, archive type, or minimum size is not integrity verification.

## 6. Package-manager and host mutation

Separate planning from application:

- detect the exact platform/package manager/capability;
- refresh metadata;
- produce or validate a transaction plan where supported;
- block protected removals or unsupported operations;
- show the exact destructive scope;
- require explicit confirmation;
- apply;
- verify;
- perform cleanup only when its prerequisites succeeded;
- decide reboot only after all required update/verification dependencies succeeded.

Do not add non-interactive confirmation flags merely for convenience when the existing safety design requires the native transaction review.

Unsupported platforms must fail closed with a concrete explanation. Do not fall through to the nearest command name.

## 7. MCP, Skill, and developer-tool execution

- AI Tool Upgrader owns runtime installation/upgrades.
- MCP Manager owns registration only.
- Skill Installer owns Skill/plugin content and provenance only.
- Never execute Skill-provided scripts merely to inspect them.
- Treat hooks, executables, network use, subprocess use, dependency changes, and persistent services as explicit capabilities requiring catalog declaration and review.
- Preserve reviewed exact versions and immutable refs from `artifact_versions.rs`.
- Do not modify the user's real global agent configuration or Skill directories in automated tests.

## 8. Tests

Use fake executable fixtures and temporary roots. Cover:

- exact program/argument construction, including spaces and leading hyphens;
- success and non-zero exit;
- timeout;
- cancellation;
- child and descendant termination;
- inherited versus captured stdio;
- stdout/stderr truncation;
- secret redaction;
- invalid path/package/version rejection;
- digest mismatch and missing provenance;
- traversal, unsafe symlink, special file, and archive-bomb limits;
- interrupted publication and rollback;
- no-sudo and unsupported-platform paths.

Keep all fake commands observable through logs or fixture state so tests prove which command would have run without touching the host.

## 9. Completion evidence

Report:

- exact program and argument policy changed;
- timeout, cancellation, output, process-tree, and redaction behavior;
- privilege boundary;
- provenance and archive validation;
- publication/rollback behavior;
- exact focused and full test commands;
- platform paths not executed and their remaining risk.
