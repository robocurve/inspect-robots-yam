# connect_operator_session: route status and gates through the framework session Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the duck-typed `connect_operator_session(session)` hook (inspect-robots
plan 0048, PR 2 of its 3-PR arc) so the framework's `OperatorSession` becomes the single
owner of the terminal on YAM runs: the status ticker renders through `session.status(...)`,
the readiness gates block through `session.gate(...)`, and this embodiment performs no
stdin reads and no direct status prints for the rest of the run. Runs where the hook is
never called (`--no-prompt`, non-TTY, win32, direct `rollout()`, older cores) are
byte-for-byte unchanged, keeping the legacy keypress poll as the working end-of-episode
path (core plan 0048 decision 5a).

**Why:** With the framework console active, the tty driver echoes typed interjections at
the cursor while this embodiment's `\r`-rewriting ticker repaints over them every second,
shredding the operator's in-progress line. Routing rendering through the session removes
the second stdout writer; PR 3 (core) then gives the session a fixed two-line footer and
the garbling is gone for good.

**Architecture:** One new public method on `YAMEmbodiment`. `connect_operator_session(session)`
stores the session and reuses the existing stand-down machinery: it sets
`self._deferred_operator_end = True` (same flag `defer_operator_end()` sets — every
consumption site already goes quiet on it) and replaces `self._status` with
`session.status` (the constructor's `status_fn` seam, so every existing `_status` call
site routes through the session unchanged). The two `wait_ready` call sites gain a
session branch: when a session is held, they call `session.gate(prompt, hint=...)`
instead of `self._operator.wait_ready(...)` — `gate` fd-flushes stale console input
before prompting and never drains after, which is exactly the deferred-mode discipline
plan 0019 built (`flush_first=True, drain=False`), now owned by the framework. The
in-episode ticker wording becomes session-aware: "Enter ends the episode" instead of
"any key ends the episode", fixing the wording plan 0019 corrected on the "Running:"
line but not on the per-second ticker. `defer_operator_end()` stays as the legacy hook
for cores that predate the session.

**Tech Stack:** Python 3.10+, no new runtime deps; pytest with the repo's injected seams
(`status_fn`, `operator`, `poll_end`, fake driver/cameras).

## Global Constraints

- Gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy` (strict), `uv run pytest --cov -q` at **100% coverage**. D1 docstrings
  state the contract. Line length per repo config.
- Set up the worktree venv with `uv sync --extra dev --python 3.11` before running
  mypy: 3.12+ venvs produce ~118 phantom NumPy-stub errors in this repo.
- **Version floor bump:** `pyproject.toml` dependency becomes `inspect-robots>=0.42`
  (v0.42.0 is tagged and released — core commit 6c022ead ships `OperatorSession` and
  the CLI hook caller). Keep the comment style of the existing pin line. The repo
  tracks `uv.lock`: run `uv lock` and commit it.
- Zero behavior change when the hook is never called: every existing test passes
  untouched. The legacy `default_poll_end` path and `defer_operator_end()` are kept.
- Worktree: `.claude/worktrees/operator-session-hook`; run everything via `uv run ...`
  from there. Reference the tracking issue in commit messages; end each with the
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer. **No git operations
  from Codex** — the orchestrating session commits.
- Public prose (README, CHANGELOG) follows the repo writing-style rules: no em dashes,
  no mid-sentence bold.

## Reference: current wiring (origin/main @ d0e378d)

- `src/inspect_robots_yam/embodiment.py:72-73` — imports `_drain_stdin`,
  `default_poll_end` from `.operator`; `:327` — `_default_status`; `:1168` —
  `status_fn` constructor param; `:1227-1231` — `self._poll_end`,
  `self._deferred_operator_end = False`, `self._status` binding; `:1367-1376` —
  `defer_operator_end()` (docstring + flag); `:1392` — `reset()`; `:1424-1426` — the
  auto_start dead-stdin fail-fast comment block; `:1463-1466` and `:1505-1507` — the
  two `wait_ready(prompt, drain=not deferred, flush_first=deferred)` call sites (home
  gate; start gate); `:1470-1483` — homing/settling status lines; `:1497-1503` — the
  auto_start `_drain_stdin()` skip when deferred; `:1511-1517` — the deferred vs legacy
  "Running:" wording; `:1543` — `_emit_status()` per step; `:1548-1549` — the legacy
  `self._poll_end()` end check (already skipped when deferred); `:1599-1606` — parking
  status lines in `close()`; `:1808-1823` — `_emit_status` body ending in
  `f"t = {span} | any key ends the episode"`.
- `src/inspect_robots_yam/operator.py` — `OperatorIO.wait_ready(drain, flush_first)`,
  `_drain_stdin`, `_flush_stdin_fd`, `stdin_interactive`, `default_poll_end`. None of
  this is deleted; it is the never-connected fallback.
- Framework contract (sibling checkout, context only):
  `inspect-robots/src/inspect_robots/session.py` — `OperatorSession.status(line)`
  (in-place `\r` renderer, `None` closes idempotently), `write_line(text)`,
  `gate(prompt, *, hint=None)` (flushes first; raises `EmbodimentFault` on
  `EOFError`/`OSError` with remedies plus the hint; never drains after);
  `inspect_robots/embodiment.py` Protocol docstring — the hook contract: accepting the
  session is a stand-down promise (never read stdin or print status after the call),
  the hook is optional input (may never fire), and the CLI calls it at most once,
  before `eval()`, POSIX-only.
- `src/inspect_robots_yam/CLAUDE.md` — module map rows for `embodiment.py` and
  `operator.py` to update. `CHANGELOG.md` — `## Unreleased` (Keep a Changelog).
  `README.md` — operator/end-episode wording: `grep -n "press any key\|end the
  episode\|feedback" README.md`.
- Tests: `grep -rln "defer_operator_end\|poll_end\|wait_ready\|status_fn" tests/` for
  the files whose fixtures to mirror (scripted `OperatorIO`, injected `poll_end`,
  recording `status_fn`, fake driver/cameras).

## Design decisions (and why)

1. **Reuse `_deferred_operator_end`, do not add a second flag.** Every stdin
   consumption site already checks it (`step()`'s poll, the auto_start drain,
   `wait_ready`'s drain/flush keywords). `connect_operator_session` sets it and
   additionally stores the session; `self._session is not None` implies
   `self._deferred_operator_end`. A separate boolean would create four flag
   combinations, three of them meaningless.
2. **Route the ticker by rebinding `self._status`, not by editing call sites.** The
   constructor already treats `self._status` as an injectable seam (`status_fn or
   _default_status`); the hook assigns `self._status = session.status`. All nine
   `_status(...)` call sites (`:1470`, `:1479`, `:1483`, `:1512`, `:1517`, `:1549`,
   `:1599`, `:1606`, `:1823` — homing, settling, running, ticker, parking, close)
   route through the session with zero diff noise. A caller-supplied `status_fn` is
   overridden by connection — document in the hook docstring that the framework
   session wins because the terminal must have one owner (the injectable seam remains
   test-visible: tests inject a recording *session*, not a `status_fn`).

   **2a. One print site bypasses the `_status` seam and must be routed explicitly:**
   the auto_start stand-clear notice at `:1459` goes through
   `self._operator.output_fn(...)` (default `print`), not `self._status`. On a
   connected run that raw print would write over the session's status line and break
   the stand-down promise the core Protocol docstring makes binding. When connected,
   this notice goes through `session.write_line(...)` (it is scrollback, not a status
   rewrite — this is why `write_line` is in the `OperatorSessionLike` Protocol);
   never-connected and deferred-only runs keep `output_fn` exactly as today.
3. **Gates route to `session.gate` with a yam-specific hint.** When `self._session`
   is set, both `wait_ready` call sites become
   `self._session.gate(prompt, hint="Set YamConfig(unattended=True) (CLI: -E
   unattended=true) to skip operator prompts.")`. The session's fault message already
   names the generic remedies (real TTY, injectable input, unattended mode); the hint
   carries the yam-specific spelling that `wait_ready`'s `EmbodimentFault` message
   carries today. `OperatorIO.wait_ready` keeps serving the never-connected path
   unchanged; the deferred-but-not-connected combination (old-core `defer_operator_end`
   runs) keeps today's `wait_ready(drain=False, flush_first=True)` behavior exactly.
   The start gate passes no explicit prompt today — its text is `wait_ready`'s
   *default*, `"Position the scene, then press Enter to start..."` (operator.py:43).
   The `session.gate` call site therefore spells that string out explicitly, and the
   test pins it verbatim so the two cannot drift.

   **3a. The auto_start dead-stdin fail-fast (`:1428-1436`) stays unchanged when
   connected.** On the CLI path it is inert (`_attended` implies a TTY); a Python-API
   caller who connects a session without a TTY gets the same pre-motion fault deferred
   runs get today, and its message stays directionally true (the framework console
   reads the same fd). No session branch there — do not improvise one.
4. **Session-aware ticker wording.** `_emit_status` ends with
   `"Enter ends the episode"` when `self._deferred_operator_end` is true (both the
   session and the old-core console mean "a bare Enter ends it"; "any key" is wrong
   for both — plan 0019 fixed the "Running:" line and missed this one) and keeps
   `"any key ends the episode"` for the legacy keypress path. This is a wording fix
   inside already-deferred runs, not a behavior change for legacy runs.
5. **`defer_operator_end()` stays, `connect_operator_session` supersedes it.** New
   cores call only the new hook (core plan 0048 decision 9 tries it first); old cores
   call only the old one. Their effects are compatible by construction (decision 1).
   The old hook's docstring gains one line pointing at the new hook.
6. **`close()` after a connected run must not print through a dead session.** It does
   not: the session outlives the embodiment (CLI scope), `session.status(None)` is
   idempotent, and parking lines simply render through the session. No teardown
   special-casing needed — state this in the hook docstring so nobody adds any.
7. **Unattended config wins, as always.** `_emit_status` returns early and the gates
   are never reached under `unattended=True`; connecting a session to an unattended
   embodiment is a CLI-side impossibility (no TTY prompt gate), but if a Python-API
   caller does it anyway nothing reads stdin and nothing renders — the same
   do-no-harm posture `defer_operator_end` has.

---

### Task 1: the hook and status routing

**Files:**
- Modify: `src/inspect_robots_yam/embodiment.py`
- Test: the embodiment test file that already scripts `defer_operator_end`
  (`grep -rln "defer_operator_end" tests/`)

**Interface:** `connect_operator_session(self, session: OperatorSessionLike) -> None` —
duck-typed structural Protocol `OperatorSessionLike` (module-level, `status`,
`write_line`, `gate` members) so the plugin never imports the concrete core class at
module top and mypy stays strict without a hard version coupling. Sets
`self._deferred_operator_end = True`, `self._session = session`,
`self._status = session.status`. D1 docstring: stand-down promise, seam override
(decision 2), teardown note (decision 6).

- [ ] **Step 1: failing tests.** Connect with a recording fake session: deferred flag
  set; every subsequent `_status` text (homing, running, ticker, parking) lands on the
  fake session in order; a constructor-injected `status_fn` stops receiving after
  connection; `step()` never calls the injected `poll_end` after connection; reset's
  auto_start path never calls `_drain_stdin` (monkeypatch-record) after connection;
  connected + `auto_start=true` lands the stand-clear notice on the fake session's
  `write_line` and never calls `self._operator.output_fn` (decision 2a — mirror the
  fixture at tests/test_embodiment.py:549), while never-connected auto_start keeps
  `output_fn` receiving it.
- [ ] **Step 2: run, confirm FAIL. Step 3: implement. Step 4: green.**

### Task 2: gates through `session.gate`

**Files:** `src/inspect_robots_yam/embodiment.py`, same test file (+ the operator IO
test file if fixtures live there)

- [ ] **Step 1: failing tests.** Connected: both gates (home gate, start gate) call
  `session.gate` with the exact prompts `wait_ready` passes today and the decision-3
  hint; `OperatorIO.wait_ready` is never called; a fake session whose `gate` raises
  `EmbodimentFault` propagates it (no wrapping, no swallow). Not connected: existing
  `wait_ready` tests untouched and passing, including the deferred-not-connected
  combination (`defer_operator_end()` alone keeps `drain=False, flush_first=True`).
- [ ] **Step 2-4: fail → implement → green.**

### Task 3: session-aware ticker wording

**Files:** `src/inspect_robots_yam/embodiment.py`, the `_emit_status` test file
(`grep -rln "_emit_status\|any key" tests/`)

- [ ] **Step 1: failing tests.** Deferred (either hook): ticker text ends with
  `"Enter ends the episode"`; legacy: `"any key ends the episode"` unchanged;
  unattended: no ticker at all (existing test).
- [ ] **Step 2-4: fail → implement → green.**

### Task 4: floor bump, docs, changelog

**Files:** `pyproject.toml`, `src/inspect_robots_yam/CLAUDE.md`, `CHANGELOG.md`,
`README.md` (wording sweep)

- [ ] `inspect-robots>=0.42` with the pin-comment convention; `uv lock` if a lockfile
  is tracked; `uv sync` still resolves.
- [ ] CLAUDE.md rows: `embodiment.py` (hook + routing), `operator.py` (explicitly the
  never-connected fallback).
- [ ] CHANGELOG under Unreleased: Added (the hook; ticker + gates through the
  framework session), Changed (session-connected runs end episodes with Enter, not
  any key; typed lines become policy feedback or logged notes), reference this plan
  and the tracking issue. No em dashes.
- [ ] README sweep for "press any key" / end-episode wording: describe both modes in
  one short paragraph.
- [ ] **Full gates** (with the 3.11 venv): `uv run ruff check . && uv run ruff format
  --check . && uv run mypy && uv run pytest --cov -q` — 100%.
