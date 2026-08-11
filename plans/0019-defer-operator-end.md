# defer_operator_end: stand down stdin polling for the framework console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the duck-typed `defer_operator_end()` hook (#102) so inspect-robots' operator console (inspect-robots#279) activates on YAM hardware: once called, the embodiment stops consuming stdin for the rest of the run — the framework console owns stdin, delivers typed feedback to the policy, and terminates trials with `operator_end` itself. Runs where the hook is never called are byte-for-byte unchanged.

**Architecture:** One boolean instance flag on `YAMEmbodiment`, set by the new public `defer_operator_end()` method and never cleared (the framework calls it at most once per run, before `eval()`). Three consumption sites go quiet when it is set: the per-step `self._poll_end()` check in `step()`, the `auto_start` branch's `_drain_stdin()` in `reset()`, and `wait_ready()`'s trailing drain (controlled by a new `drain: bool = True` keyword on `OperatorIO.wait_ready`) — when a console owns stdin, every pending line is console input, and exactly one reader discipline must consume it. The blocking start prompts themselves stay, but they gain a **deferred-mode safety flush**: the console solicits continuous typing, and its `begin_trial()` drain runs only *after* `embodiment.reset()` returns (framework rollout.py), so a stale feedback line pending when `reset()` starts would otherwise be consumed by `wait_ready`'s blocking `input()` as a bogus "stand clear" confirmation — arms ramping with no real human go-ahead. In deferred mode, `wait_ready` therefore discards pending input at the fd level (select + `os.read`, never buffered `readline`) immediately before printing its prompt. The per-trial "Running:" status line switches to console-aware wording so the operator is not told "press any key to end" when a keypress now means feedback.

**Tech Stack:** Python 3.10+, no new deps; pytest with the repo's injected seams (`operator`, `poll_end`, monkeypatched module-level `_drain_stdin`).

## Global Constraints

- Gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy` (strict), `uv run pytest --cov -q` at **100% coverage**. Ruff D1 docstrings on public defs: state the contract.
- No new inspect-robots version requirement: the hook is duck-typed on the yam side; older frameworks simply never call it, and the new framework treats its absence as "not console-safe". Do not touch `pyproject.toml`.
- Zero behavior change when `defer_operator_end()` is never called: every existing test passes untouched.
- Worktree: `/Users/jeqcho/robocurve/inspect-robots-yam/.claude/worktrees/defer-operator-end`; run everything via `uv run ...` from there. Reference #102 in commit messages; the orchestrator commits (no git operations from Codex).
- Public prose (README, CHANGELOG) follows the repo writing-style rules: no em dashes, no mid-sentence bold.

## Reference: current wiring (origin/main @ 914c0ee)

- `src/inspect_robots_yam/embodiment.py:72-73` — imports `_drain_stdin`, `default_poll_end` from `.operator`. `:1205` — `self._poll_end` binding in `__init__`. `:1339` — `reset()`; `:1371-1382` — the auto_start TTY fail-fast (mentions the keypress; text stays accurate enough, do not touch). `:1410` and `:1447` — the two `wait_ready(...)` call sites (home gate; start gate). `:1442-1445` — the auto_start `_drain_stdin()` with its comment. `:1450` — the "Running: press any key to end the episode and grade it." status line. `:1455` — `step()`; `:1481` — the `if not self._cfg.unattended and self._poll_end():` end check returning `terminated=True, termination_reason=OPERATOR_END`.
- `src/inspect_robots_yam/operator.py:27-46` — `OperatorIO.wait_ready(prompt)` ends with `_drain_stdin()`; `:49-59` — `_drain_stdin` (buffered `sys.stdin.readline` loop); `:69-84` — `default_poll_end`.
- Framework contract (context only, sibling checkout `/Users/jeqcho/robocurve/inspect-robots`): `src/inspect_robots/embodiment.py` Protocol docstring documents `defer_operator_end()`; `src/inspect_robots/cli.py` `_build_operator_console` calls it when present and only then enables the console; the console reads stdin fd-level (`src/inspect_robots/console.py`) and drains via `begin_trial()` after each `embodiment.reset()` returns.
- `src/inspect_robots_yam/CLAUDE.md` — module map to update. `CHANGELOG.md` — entry per release-notes convention (read its head for format). `README.md` — the operator/end-episode wording, `grep -n "press any key\|end the episode" README.md`.
- Tests: `grep -rn "poll_end\|_drain_stdin\|wait_ready" tests/*.py -l` for the files that already script these seams; mirror their fixtures (scripted `OperatorIO` input_fn/output_fn, injected `poll_end`, fake driver/cameras).

## Design decisions (and why)

1. **The flag is set-only.** The framework calls the hook once per CLI run before `eval()`; there is no un-defer path in the contract, and `reset()` must not clear it (trials 2..N still belong to the same console-owning run). A Python-API caller who defers and then reuses the object for an unattended run loses nothing: with `unattended=True` the poll was already skipped.
2. **Skip the drains, not just the poll — as reader discipline, not buffer mechanics.** On a canonical-mode TTY a buffered `readline` consumes exactly one pending line per call, so `_drain_stdin` does not actually strand bytes in the `TextIOWrapper` buffer; do not claim that in docstrings. The real rationale: once the framework console owns stdin, every pending line is console input, and mixing a second (buffered) reader with the console's fd-level reads is fragile by construction — one owner, one discipline. When deferred, the auto_start drain and `wait_ready`'s trailing drain are skipped; anything pending after `reset()` returns is the console's to discard via `begin_trial()`.
3. **`drain` and `flush_first` keywords on `wait_ready`, defaulting to today's behavior** (`drain=True`, `flush_first=False`). Both embodiment call sites pass `drain=not deferred, flush_first=deferred`. `flush_first` is the safety mitigation for the blocking gates: `begin_trial()`'s drain runs only after `reset()` returns, so it cannot protect the gates that run *inside* `reset()` — a feedback line typed during scoring or the trial-start window (the console's own usage hint invites typing) would otherwise auto-confirm "Arms will move to the home pose - stand clear, then press Enter" and the arms would ramp with no real confirmation. The flush discards pending input at the fd level (zero-timeout `select` + `os.read` loop; a new module-private `_flush_stdin_fd()` in `operator.py`, TTY-bound lines pragma-no-cover, injectable on `OperatorIO` as `flush_fn` for tests). Discarding feedback typed between trials is correct and matches the framework contract: `begin_trial()` would discard it anyway.
4. **Status line switches wording when deferred** to: `Running: Enter ends the episode; type a message + Enter to send feedback.{limit}` — same single-line, same `{limit}` suffix. The framework prints its full usage line once at run start; this per-trial line is what the operator actually stares at, and "press any key" would now be wrong twice over (any key does not end it, and typing is meaningful).
5. **`unattended` still wins.** The deferred flag only silences stdin consumption; it does not create any new prompts or output in unattended mode, where all these code paths are already skipped.

---

### Task 1: `OperatorIO.wait_ready(drain=..., flush_first=...)` and `_flush_stdin_fd`

**Files:**
- Modify: `src/inspect_robots_yam/operator.py`
- Test: the existing operator IO test file (`grep -rln "wait_ready" tests/`)

**Interfaces:**
- `_flush_stdin_fd() -> None` (module-private): zero-timeout `select` + `os.read(sys.stdin.fileno(), 65536)` discard loop; no-op off-TTY; stops on empty read (EOF). TTY-bound lines carry `# pragma: no cover` exactly like `default_poll_end`.
- `OperatorIO` gains a `flush_fn: Callable[[], None] = _flush_stdin_fd` field (injectable, matching `input_fn`/`output_fn`).
- `wait_ready(prompt, *, drain: bool = True, flush_first: bool = False)`: when `flush_first`, call `self.flush_fn()` **before** printing the prompt (stale console input must not confirm a safety gate); when `not drain`, skip the trailing `_drain_stdin()`. Docstring states the ownership contract: the trailing drain protects the legacy keypress path from a stray buffered newline; a framework console that owns stdin passes `drain=False` (pending lines belong to the console) and `flush_first=True` (stale feedback must not stand in for a gate confirmation).

- [ ] **Step 1: failing tests** — `wait_ready(drain=False)` never calls `_drain_stdin` (monkeypatch it to record); default and `drain=True` still call it once; `flush_first=True` invokes the injected `flush_fn` before `input_fn` (record ordering) and the default `flush_first=False` never does; the EmbodimentFault-on-dead-stdin path is unaffected by both keywords.
- [ ] **Step 2: run, confirm FAIL. Step 3: implement. Step 4: green.**

---

### Task 2: the embodiment flag and its three quiet sites

**Files:**
- Modify: `src/inspect_robots_yam/embodiment.py`
- Test: the embodiment test files that already script `poll_end`/`_drain_stdin`/`wait_ready`

**Interfaces:**
- `YAMEmbodiment.defer_operator_end() -> None` (public, D1 docstring stating the framework contract: called by inspect-robots when its operator console owns stdin for the run; after the call this embodiment never reads stdin again — no end-of-episode poll, no drains — and the framework terminates trials with `operator_end` itself). Sets `self._deferred_operator_end = True` (initialized `False` in `__init__` near the `_poll_end` binding at `:1205`).
- `step()` `:1481`: `if not self._cfg.unattended and not self._deferred_operator_end and self._poll_end():`.
- `reset()` `:1442-1445`: auto_start drain becomes conditional on `not self._deferred_operator_end` (keep the existing comment, extend it with one line on why deferred skips it: pending lines belong to the console, which drains after reset returns). `:1447` and `:1410`: pass `drain=not self._deferred_operator_end, flush_first=self._deferred_operator_end`.
- `reset()` `:1450`: deferred wording per Design decision 4.

- [ ] **Step 1: failing tests** — (a) after `defer_operator_end()`, a `poll_end` scripted to return `True` is never called and the episode does not terminate (steps continue); (b) undeferred behavior identical (existing tests stay green); (c) deferred + auto_start `reset()` never calls `_drain_stdin` (monkeypatch the module-level name in `embodiment`'s namespace); deferred + gated `reset()` passes `drain=False, flush_first=True` to both `wait_ready` sites (record kwargs via a scripted `OperatorIO`); undeferred still drains and never flushes; (d) the deferred status line says Enter ends the episode and mentions feedback, with the horizon suffix preserved; (e) `defer_operator_end()` survives `reset()` (defer, reset twice, still no poll); (f) the attribute is a callable on the instance (the framework's `getattr` contract).
- [ ] **Step 2: run, confirm FAIL. Step 3: implement. Step 4: green**, full suite + gates.

---

### Task 3: docs surface

**Files:**
- Modify: `src/inspect_robots_yam/CLAUDE.md` (embodiment row: mention the hook), root `CLAUDE.md` (the "Safety invariants" keypress bullet at `:64-66` currently states unconditionally that yam emits `operator_end`; add the deferred case — under the framework console yam never reads the keypress, the console terminates and may carry a verdict that suppresses the prompt), `CHANGELOG.md` (entry: operator console support via `defer_operator_end()`, #102), `README.md` (only where it says the end-episode keypress wording — add one short sentence that with inspect-robots' operator console the framework owns stdin and typing sends feedback; follow the writing-style rules)

- [ ] Update all four; run the full gates one final time (`uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, `uv run pytest --cov -q`).
