# Network Listener Boundary

> This file is intentionally not part of normal Ops-Tools work. Read it only when an explicit request proposes a listening HTTP, WebSocket, Socket.IO, TCP, Unix-domain-socket, or resident service.

## Current product state

Ops-Tools is a local interactive Rust CLI. It does not provide an application web server, browser frontend, account system, raw WebSocket endpoint, Socket.IO endpoint, or background network daemon.

Configured MCP servers are external child processes, normally registered over stdio. Their lifecycle belongs to the MCP Manager and shared process-execution policy; they are not an Ops-Tools socket protocol.

For normal subprocess, MCP, installer, timeout, cancellation, or process-tree changes, read:

```text
docs/codex/PROCESS_EXECUTION.md
```

## Do not add implicitly

Do not introduce any of the following as implementation convenience:

- an HTTP health/config API;
- WebSocket or Socket.IO control channel;
- a listener on `0.0.0.0`;
- LAN/remote management;
- browser-based authentication;
- OAuth/OIDC/JWT/session/RBAC application infrastructure;
- a persistent updater, scanner, or MCP daemon;
- an unauthenticated local API assumed safe merely because it is loopback.

These are product-scope and threat-model changes, not small refactors.

## Requirements for an explicitly approved listener

Before implementation, define and document:

- why a CLI/subprocess/stdio design is insufficient;
- process ownership and shutdown lifecycle;
- loopback-only versus Unix-domain-socket transport;
- exact bind address and port/path behavior;
- authentication/authorization decision, including same-machine untrusted processes;
- TLS decision;
- payload/schema/version contract;
- message and request size limits;
- timeouts, cancellation, retries, queue bounds, and backpressure;
- origin policy if a browser is involved;
- secret storage and logging policy;
- platform support on Windows, macOS, native Linux, WSL, and containers;
- tests for startup collision, malformed/oversized input, disconnect, shutdown, and leaked tasks/processes.

Default to `127.0.0.1` and/or `::1` only after the threat model explicitly accepts loopback. Never use wildcard origins or `0.0.0.0` as convenience defaults.

If a network service becomes an accepted feature, replace this guard document with a protocol-specific playbook and update the root `AGENTS.md`, release/security gates, README, and platform model.
