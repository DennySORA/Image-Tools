# Terminal UI/UX Playbook

Read this file for interactive menus, prompts, progress, terminal layout, status/error presentation, localization, Unicode alignment, or accessibility changes.

## 1. Design objective

Produce a calm, compact, predictable terminal interface for high-impact developer and system operations. Visual polish is subordinate to clarity, safety, accurate state, and recoverability.

The user must be able to answer, before confirming:

- what operation will run;
- which host, client, scope, path, package, runtime, MCP, or Skill it affects;
- whether the action is read-only or mutating;
- what validation will occur;
- whether rollback or recovery exists.

## 2. Pre-change audit

Inspect the actual terminal flow and implementation. Record concrete problems in:

- menu hierarchy and primary task path;
- default selection and first-Enter behavior;
- prompt wording, cancellation, back navigation, and EOF handling;
- destructive-action preview and confirmation;
- status distinctions and partial-failure summary;
- alignment in English, Traditional Chinese, Simplified Chinese, and Japanese;
- narrow terminal behavior, long paths/versions/errors, and Unicode width;
- color contrast and no-color readability;
- duplicate, stale, or contradictory messages;
- interleaving of child-process output and Ops-Tools status;
- Windows/macOS/Linux terminal differences where relevant.

Base changes on observed behavior and source evidence, not web-app design conventions.

## 3. UI ownership

- Use `src/ui/Console` for status, warnings, errors, sections, and consistent styling.
- Use `src/ui/Prompts` and established `dialoguer` patterns for selection and confirmation.
- Use `src/ui/sanitize.rs` before displaying or persisting untrusted external output.
- Use `src/i18n` keys for user-visible interactive text.
- Keep feature policy in the owning feature module; UI helpers must not decide platform, security, or lifecycle policy.
- Avoid raw `println!`/`eprintln!` for interactive feature output unless implementing a deliberate machine-readable or basic CLI subcommand contract.

## 4. State vocabulary

Use distinct text and styling for every applicable state:

- current;
- missing;
- update available;
- newer than reviewed catalog;
- selected;
- disabled/unsupported;
- pending;
- running;
- success;
- partial success;
- failed;
- skipped;
- cancelled;
- timed out;
- drifted;
- untracked;
- rollback available;
- rollback failed;
- not attempted because a prerequisite failed.

Do not use a single generic “done” or “error” state when the operational consequence differs. Never communicate state by color or icon alone.

## 5. Selection and navigation

- A single-select prompt must have an intentional selectable default; pressing Enter first must not be discarded.
- Headers and separators must not become selectable.
- Multi-select defaults must reflect safe policy, not maximum coverage.
- Escape/back/cancel must return to the nearest safe level without running work.
- EOF and Ctrl-C must exit or cancel cleanly without panic or partial implicit confirmation.
- Preserve the user's prior selection when returning to a menu when practical.
- Do not make destructive or broad-scope options the default.

## 6. Confirmation contract

Before a mutation, show the minimum complete preview:

- action;
- target/scope;
- reviewed version/revision when relevant;
- important capability or privilege;
- expected files/configuration/package effects;
- rollback availability;
- irreversible or externally hosted behavior.

Confirmation rules:

- Read-only diagnostics may run without a destructive warning.
- Broad, destructive, privileged, network, persistent-service, project-shared, or rollback-losing actions require explicit confirmation.
- Do not use ambiguous prompts such as “Continue?” without showing the operation immediately above.
- Do not default irreversible confirmation to yes.
- A catalogue pin or checksum does not imply harmless content; still show declared Skill capabilities.
- For project-shared Claude MCP scope, explicitly state that `.mcp.json` may be committed.

## 7. Output and layout

- Use `unicode-width` for terminal display alignment.
- Use `saturating_sub` or equivalent when calculating padding.
- Prefer one item/state per line and bounded path previews.
- Wrap or truncate intentionally; never panic on a narrow terminal.
- Keep long command output separate from the final summary.
- Mark captured output truncation.
- Avoid repaint-heavy animation, spinners that obscure child output, and layout assumptions requiring a specific terminal width.
- Keep machine-readable commands stable and free from decorative ANSI output.

Suggested manual widths:

- 60 columns: degraded but readable;
- 80 columns: minimum normal target;
- 100–120 columns: common development terminal;
- 160 columns: ensure spacing does not become wasteful.

## 8. Internationalization

Supported locales:

- English;
- Traditional Chinese;
- Simplified Chinese;
- Japanese.

Rules:

- Add/update all four locales in the same change when user-visible behavior changes.
- Do not build sentences by concatenating translated fragments whose order differs by language.
- Keep placeholders named or positionally unambiguous and verify their count.
- Avoid unexplained abbreviations and culture-specific idioms.
- Allow translated labels and descriptions to expand substantially.
- Keep package names, command names, paths, versions, and protocol identifiers untranslated.
- Test mixed CJK/ASCII alignment with actual display width.

## 9. Accessibility and terminal compatibility

- Text must carry the full meaning; color and symbols are enhancements.
- Preserve visible keyboard focus/selection in interactive prompts.
- Use clear verbs and recovery instructions.
- Treat expected cancellation and unsupported capability calmly; reserve alarming language for actual risk or failure.
- Avoid rapidly changing/flashing output.
- Preserve readable output when ANSI color is unavailable or disabled.
- Avoid leaking secrets, full auth headers, or unnecessary private paths into terminal history and CI logs.
- Keep Windows console differences in mind for path separators, process interruption, and Unicode.

## 10. Validation matrix

For changed flows, cover applicable cases:

- default selection and first Enter;
- back/Escape;
- Ctrl-C;
- EOF/non-interactive stdin;
- empty catalog/list;
- missing dependency;
- current/update/newer/drifted/untracked states;
- safe versus broad scope;
- confirmation accepted and declined;
- success, partial failure, total failure, timeout, cancellation, rollback;
- long paths, versions, command output, and translated labels;
- 60/80/100/120-column terminals;
- all four locales;
- no-color/plain-text readability;
- Windows, macOS, and Linux behavior when platform-specific.

Automate deterministic formatting, selection, sanitization, and summary logic where practical. Manual terminal inspection complements tests but does not replace them.

## 11. Completion evidence

Report:

- affected menu/prompt/status flow;
- safe default and confirmation behavior;
- states exercised;
- locales checked;
- terminal widths/platforms checked;
- exact tests;
- remaining unverified terminal/platform behavior.

Browser screenshots, responsive web viewports, Playwright, Chrome DevTools, CSS, WCAG web-component checks, and Socket.IO state are not part of this CLI playbook unless the product explicitly gains a browser UI.
