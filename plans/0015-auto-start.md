# `auto_start` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `YamConfig.auto_start` so an attended run homes the arms and starts the episode with zero operator Enter presses, while keeping every other attended behavior (issue #87).

**Architecture:** A single boolean on the frozen `YamConfig` dataclass, consumed only inside `YAMEmbodiment.reset()`. When set (and `unattended` is false), the two `OperatorIO.wait_ready()` gates are skipped: the stand-clear gate is replaced by a one-line printed notice (once per connection, same lifecycle as `_home_gate_confirmed`), and the scene-ready gate is replaced by a bare stdin drain so a buffered newline cannot trip the end-episode poll on step 1. Because `wait_ready()` today also fail-fasts on a dead stdin before any motion, the auto-start path adds an equivalent guard: `reset()` raises `EmbodimentFault` before connecting the driver when stdin is not an interactive TTY (an operator who cannot press a key to end episodes or answer the grading prompt must use `unattended` instead). `unattended=True` takes precedence and makes `auto_start` a no-op.

**Tech Stack:** Python 3.10+, frozen dataclasses, pytest with injected seams (no hardware/stdin).

## Global Constraints

- All gates blocking in CI: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy` (strict), `uv run pytest --cov` at **100% coverage**.
- Every public module/class/function needs a docstring (Ruff D1); state the contract, not the symbol name.
- Public-facing text (README): no em dashes in prose; bold only for `**term:**` lead-ins; no decorative emoji; headers use colons.
- `reset()` safety semantics must not weaken for existing configs: with `auto_start` unset (the default), behavior is byte-for-byte identical, prompts included.
- Repo root is the `auto-start` worktree at `.claude/worktrees/auto-start`. Run everything through `uv run ...` there.
- Commit messages: imperative, scoped, and reference `#87` where apt.

## Reference: current `reset()` gate structure (`src/inspect_robots_yam/embodiment.py:1096-1131`)

```python
        if not self._cfg.unattended and not self._home_gate_confirmed:
            self._operator.wait_ready(
                "Arms will move to the home pose - stand clear, then press Enter..."
            )
            self._home_gate_confirmed = True
        if not self._cfg.unattended:
            self._status("homing: ramping arms to start pose")
        ...
        if not self._cfg.unattended:
            self._operator.wait_ready()
            horizon = self._horizon_secs()
            limit = f" Max {horizon:.0f}s." if horizon is not None else ""
            self._status(f"Running: press any key to end the episode and grade it.{limit}")
```

`OperatorIO.wait_ready()` (`src/inspect_robots_yam/operator.py:27`) blocks on Enter, converts a dead stdin into an `EmbodimentFault` with remediation text, and calls `_drain_stdin()` afterwards. The drain is load-bearing: `default_poll_end()` treats any buffered stdin line as an "end episode" keypress, so skipping `wait_ready()` without draining would end the episode on the first step whenever a stray newline is buffered. The fail-fast is load-bearing too: it is what stops a headless attended run before any motion, and the auto-start path must preserve that property with its own TTY check (`default_poll_end()` returns `False` off-TTY, so without the check the arms would home and every episode would run to `max_steps` with no way to end it except the e-stop).

---

### Task 1: `YamConfig.auto_start` field

**Files:**
- Modify: `src/inspect_robots_yam/config.py` (field block around line 164, next to `unattended`)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `YamConfig.auto_start: bool = False`, settable via `YamConfig(auto_start=True)` and `YamConfig.from_kwargs(auto_start=True)`. Task 2 reads `self._cfg.auto_start`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_auto_start_defaults_off_and_binds_via_kwargs() -> None:
    assert YamConfig().auto_start is False
    assert YamConfig.from_kwargs(auto_start=True).auto_start is True
```

(`YamConfig` is already imported at the top of `tests/test_config.py`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_auto_start_defaults_off_and_binds_via_kwargs -v`
Expected: FAIL with `AttributeError: 'YamConfig' object has no attribute 'auto_start'` (and a `TypeError` from `from_kwargs` on the unknown key if the first assert is removed).

- [ ] **Step 3: Write minimal implementation**

In `src/inspect_robots_yam/config.py`, directly below `unattended: bool = False` (keep the existing comment placement conventions — this file documents fields with `#` comments, not docstrings):

```python
    unattended: bool = False
    # Skip both operator Enter gates in reset(): the stand-clear home gate is
    # replaced by a printed one-line notice and homing begins immediately; the
    # scene-ready gate is dropped and the episode starts as soon as the arms
    # settle at the home pose. Every other attended behavior stays: status
    # lines, the end-episode keypress, and operator grading. Stage the scene
    # BEFORE launching the run. Requires an interactive terminal; reset()
    # faults before any motion otherwise (headless runs want unattended).
    # unattended=True takes precedence and makes this a no-op.
    auto_start: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py::test_auto_start_defaults_off_and_binds_via_kwargs -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/inspect_robots_yam/config.py tests/test_config.py
git commit -m "config: add YamConfig.auto_start flag (#87)"
```

---

### Task 2: `reset()` honors `auto_start`

**Files:**
- Modify: `src/inspect_robots_yam/operator.py` (new `stdin_interactive()` helper)
- Modify: `src/inspect_robots_yam/embodiment.py` (import at line 60; TTY guard after the camera-reader check at line 1077; gates at lines 1096-1131; `reset()` docstring at line 1055; stale `_home_gate_confirmed` comments at lines 977-980 and 1225-1227)
- Test: `tests/test_embodiment.py`, `tests/test_operator.py`

**Interfaces:**
- Consumes: `YamConfig.auto_start` from Task 1; `_drain_stdin` from `inspect_robots_yam.operator`; `EmbodimentFault` (already imported in `embodiment.py:37`).
- Produces: `inspect_robots_yam.operator.stdin_interactive() -> bool` (public, docstringed); the runtime behavior documented in Task 3's README text.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_embodiment.py` (near `test_unattended_skips_operator_prompts`, line ~380). `pytest`, `Scene`, `Action`, `np`, `OperatorIO`, `YamConfig`, `EmbodimentFault`, `_build`, and `_build_with_status` are already in scope; add `import inspect_robots_yam.embodiment as embodiment_module` at the top of the first-party import block (straight imports sort before from-imports under ruff isort, so it goes above the `from inspect_robots.embodiment import SELF_PACED` group's first-party section, not beside the `YAMEmbodiment` from-import).

Note on the monkeypatching: under default pytest capture stdin is not a TTY, so the real `stdin_interactive()` returns `False` and every auto-start success path must patch it to `True` (patch `embodiment_module.stdin_interactive` — `embodiment.py` binds the name at import via `from ... import`, and the call site looks it up as a module global). The fault test patches it to `False` rather than relying on that ambient state, so the suite stays deterministic under `pytest -s` from a real terminal.

```python
def test_auto_start_skips_gates_but_keeps_attended_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embodiment_module, "stdin_interactive", lambda: True)
    prompts: list[str] = []
    lines: list[str] = []

    def _input(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    op = OperatorIO(input_fn=_input, output_fn=lines.append)
    emb, _, _ = _build(YamConfig(auto_start=True), poll_end_seq=[True], operator=op)
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(14)))
    assert prompts == []  # neither Enter gate ran
    assert any("stand clear" in line for line in lines)  # notice replaces the home gate
    assert result.terminated is True  # end-episode keypress still active
    assert result.termination_reason == "operator_end"


def test_auto_start_keeps_running_status_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embodiment_module, "stdin_interactive", lambda: True)
    emb, status = _build_with_status(YamConfig(auto_start=True))
    emb.reset(Scene(id="s", instruction="x"))
    assert any(line is not None and line.startswith("Running:") for line in status)


def test_auto_start_notice_prints_once_per_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embodiment_module, "stdin_interactive", lambda: True)
    lines: list[str] = []
    op = OperatorIO(input_fn=lambda _p: "", output_fn=lines.append)
    emb, _, _ = _build(YamConfig(auto_start=True), operator=op)
    emb.reset(Scene(id="s1", instruction="x"))
    emb.reset(Scene(id="s2", instruction="x"))
    assert sum("stand clear" in line for line in lines) == 1
    emb.close()
    emb.reset(Scene(id="s3", instruction="x"))  # new connection: notice again
    assert sum("stand clear" in line for line in lines) == 2


def test_auto_start_drains_stdin_before_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embodiment_module, "stdin_interactive", lambda: True)
    drains: list[bool] = []
    monkeypatch.setattr(embodiment_module, "_drain_stdin", lambda: drains.append(True))
    prompts: list[str] = []

    def _input(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    op = OperatorIO(input_fn=_input, output_fn=lambda _m: None)
    emb, _, _ = _build(YamConfig(auto_start=True), operator=op)
    emb.reset(Scene(id="s", instruction="x"))
    assert drains == [True]  # wait_ready's drain is replaced, not dropped
    assert prompts == []


def test_auto_start_requires_interactive_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embodiment_module, "stdin_interactive", lambda: False)
    emb, drv, _ = _build(YamConfig(auto_start=True))
    with pytest.raises(EmbodimentFault, match="auto_start"):
        emb.reset(Scene(id="s", instruction="x"))
    assert drv.commands == []  # faulted before any motion


def test_unattended_precedes_auto_start() -> None:
    prompts: list[str] = []
    lines: list[str] = []

    def _input(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    op = OperatorIO(input_fn=_input, output_fn=lines.append)
    emb, _, _ = _build(
        YamConfig(unattended=True, auto_start=True), poll_end_seq=[True], operator=op
    )
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(14)))
    assert prompts == []
    assert lines == []  # no stand-clear notice, no TTY fault: unattended wins outright
    assert result.terminated is False  # unattended still disables the end poll
```

Append to `tests/test_operator.py` (extend its `from inspect_robots_yam.operator import ...` line with `stdin_interactive` and add `import sys` to the imports), so the real function body stays covered — and its contract pinned in both directions — even though the embodiment tests patch it out:

```python
def test_stdin_interactive_reports_tty_state(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Stub:
        def __init__(self, tty: bool) -> None:
            self._tty = tty

        def isatty(self) -> bool:
            return self._tty

    monkeypatch.setattr(sys, "stdin", _Stub(True))
    assert stdin_interactive() is True
    monkeypatch.setattr(sys, "stdin", _Stub(False))
    assert stdin_interactive() is False
```

(`stdin_interactive` resolves `sys.stdin` at call time, so patching the `stdin` attribute of the shared `sys` module is what the implementation observes.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embodiment.py tests/test_operator.py -k "auto_start or stdin_interactive" -v`
Expected: every new test FAILS except `test_unattended_precedes_auto_start`, which already passes today (unattended short-circuits both gates; keep it, it pins the precedence contract against regressions). Failure modes: `TypeError: unexpected keyword 'auto_start'` if Task 1 is not merged yet, otherwise `AttributeError` from `monkeypatch.setattr` (no `stdin_interactive` on the embodiment module yet) in the patched tests, and `ImportError` in `tests/test_operator.py`.

- [ ] **Step 3: Implement**

1. In `src/inspect_robots_yam/operator.py`, after `_drain_stdin`:

```python
def stdin_interactive() -> bool:
    """True when stdin is a real TTY that can deliver operator keypresses."""
    import sys

    return sys.stdin.isatty()
```

2. In `src/inspect_robots_yam/embodiment.py`, extend the import at line 60:

```python
from inspect_robots_yam.operator import (
    OperatorIO,
    _drain_stdin,
    default_poll_end,
    stdin_interactive,
)
```

3. Add the TTY guard in `reset()` directly after the camera-reader `ConfigError` block (line 1077) and before `self._driver = self._driver_factory(...)`, so it faults before connecting hardware, on every reset:

```python
        # auto_start still needs stdin: the end-episode keypress and the
        # framework's grading prompt both read it. wait_ready() normally
        # fail-fasts a dead stdin before any motion; with the gates skipped,
        # this check keeps that property (off-TTY, default_poll_end() always
        # returns False, so episodes could otherwise only end at max_steps).
        if self._cfg.auto_start and not self._cfg.unattended and not stdin_interactive():
            raise EmbodimentFault(
                "auto_start needs an interactive terminal: the end-episode "
                "keypress and the operator grading prompt both read stdin, "
                "which is not a TTY here. Run from a real TTY, or set "
                "YamConfig(unattended=True) (CLI: -E unattended=true) for "
                "headless runs."
            )
```

4. Replace the home gate (lines 1096-1100):

```python
        if not self._cfg.unattended and not self._home_gate_confirmed:
            if self._cfg.auto_start:
                # Non-blocking replacement for the stand-clear gate: the
                # operator opted into zero-touch starts, but still gets one
                # line of warning before the first motion of the connection.
                self._operator.output_fn(
                    "auto_start: arms will move to the home pose - stand clear."
                )
            else:
                self._operator.wait_ready(
                    "Arms will move to the home pose - stand clear, then press Enter..."
                )
            self._home_gate_confirmed = True
```

5. Replace the scene-ready gate (lines 1127-1128), leaving the `horizon` / `Running:` status lines that follow exactly where they are, inside the `if not self._cfg.unattended:` block:

```python
        if not self._cfg.unattended:
            if self._cfg.auto_start:
                # wait_ready() owns the stdin drain; skipping the gate must not
                # skip the drain, or a buffered newline ends the episode on the
                # first default_poll_end() check.
                _drain_stdin()
            else:
                self._operator.wait_ready()
```

6. Update the `reset()` docstring (line 1055) to:

```python
        """Connect (if needed), drive to home, and block on operator readiness.

        With ``auto_start`` set, both operator gates are skipped: a printed
        notice replaces the stand-clear prompt and the episode begins as soon
        as the arms settle at the home pose. Requires an interactive stdin
        (faults before any motion otherwise). ``unattended`` skips the gates
        too, along with the rest of the attended flow, and takes precedence.
        """
```

7. Refresh the two comments whose wording assumes the gate is always a prompt:

At lines 977-980 (`__init__`), change

```python
        # Set only after the stand-clear prompt returns, so a gate fault
        # (dead stdin) re-prompts on a retried reset instead of ramping
        # unconfirmed; cleared on close() so every connection re-confirms.
```

to

```python
        # Set only after the stand-clear gate resolves (prompt returned, or
        # the auto_start notice printed), so a gate fault (dead stdin)
        # re-prompts on a retried reset instead of ramping unconfirmed;
        # cleared on close() so every connection re-confirms.
```

At lines 1225-1227 (`close()`), change

```python
                    # Clear connection state even if the driver's own close()
                    # raises, so a later reset() reconnects, re-captures, and
                    # re-confirms the stand-clear gate.
```

to

```python
                    # Clear connection state even if the driver's own close()
                    # raises, so a later reset() reconnects, re-captures, and
                    # re-runs the stand-clear gate (prompt, or the auto_start
                    # notice).
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embodiment.py tests/test_operator.py -v`
Expected: all PASS, including the six new embodiment tests, the operator unit test, and every pre-existing prompt/status test (defaults unchanged).

- [ ] **Step 5: Run the full gate set**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest --cov -q`
Expected: clean, coverage 100%. Both guard exits (fault and fall-through), both arms of each gate branch, and both return paths of the real `stdin_interactive` body (via the operator unit test's stubs) are exercised by the tests above. If coverage still flags something, a test is not exercising it — fix the test, do not add pragmas (none of this code is hardware-bound).

- [ ] **Step 6: Commit**

```bash
git add src/inspect_robots_yam/operator.py src/inspect_robots_yam/embodiment.py tests/test_embodiment.py tests/test_operator.py
git commit -m "embodiment: auto_start skips both operator Enter gates (#87)"
```

---

### Task 3: Documentation

**Files:**
- Modify: `README.md` (attended-flow paragraph around line 278, unattended paragraph around line 315, config reference around line 644)
- Modify: `CHANGELOG.md` (`## Unreleased` → `### Added`)
- Modify: `src/inspect_robots_yam/CLAUDE.md` (module-map rows for `operator.py` and `embodiment.py`)

**Interfaces:**
- Consumes: the behavior implemented in Task 2. No code.

- [ ] **Step 1: README attended-flow section (~line 278)**

After the sentence describing the attended flow ("position the scene, press Enter to start, press any key to..."), add a paragraph (plain prose, no em dashes):

```markdown
To skip both Enter gates, set `auto_start=true` (CLI: `-E auto_start=true`, or
persistently via `[embodiment.args]` in config.ini). The arms home immediately
after a one-line stand-clear notice and the episode starts as soon as they
settle, so stage the scene before launching the run. Everything else about the
attended flow stays: the status line, the press-any-key end, and operator
grading, which is also why `auto_start` refuses to run without an interactive
terminal.
```

Adjust the lead-in wording so the existing paragraph and the new one read as one section; keep the existing safety note about the e-stop nearby if one is present.

- [ ] **Step 2: README unattended section (~line 315)**

Extend the paragraph that introduces `unattended=True` with one sentence distinguishing the two flags (adapt to the surrounding sentence flow):

```markdown
For attended runs that only want to drop the Enter gates, use `auto_start=true`
instead; `unattended` wins when both are set.
```

- [ ] **Step 3: README config reference (~line 644)**

Extend the config list entry right after `unattended`:

```markdown
`auto_start` (default `False`; skip both operator Enter gates but keep the
attended episode flow; needs a TTY; `unattended` takes precedence),
```

Match the exact formatting of the neighboring entries (backticks, semicolons, trailing comma).

- [ ] **Step 4: CHANGELOG entry**

Add a bullet to the `### Added` list under `## Unreleased` in `CHANGELOG.md`, matching the existing entry style:

```markdown
- `YamConfig.auto_start` (plan 0015, #87): opt-in zero-touch attended starts.
  Skips both operator Enter gates in `reset()` (a printed stand-clear notice
  replaces the home gate; the scene-ready gate is dropped and stdin is drained
  in its place) while keeping status lines, the end-episode keypress, and
  operator grading. Faults before any motion when stdin is not an interactive
  TTY; `unattended=True` takes precedence.
```

- [ ] **Step 5: Module map (`src/inspect_robots_yam/CLAUDE.md`)**

In the Modules table, extend the `operator.py` row to mention the new helper, e.g. append to the row text: "; `stdin_interactive`, the TTY probe behind `auto_start`'s pre-motion fail-fast". Leave the `embodiment.py` row unless its wording now misleads (it names the operator-keypress episode end, which is unchanged).

- [ ] **Step 6: Verify docs style and gates**

Re-read the edits against the repo writing rules: no em dashes in prose, no bold mid-sentence, headers unchanged (the CHANGELOG and module map are not README prose, but keep them tell-free too). Run `uv run pytest -q` once more (docs edits cannot break tests, but the commit gate below expects a green tree).

- [ ] **Step 7: Commit**

```bash
git add README.md CHANGELOG.md src/inspect_robots_yam/CLAUDE.md
git commit -m "docs: document auto_start zero-touch starts (#87)"
```

---

## Out of scope

- Changing the default (prompts stay on for everyone who has not opted in): the stand-clear gate is a safety feature and CLAUDE.md forbids weakening safety defaults.
- Core `inspect-robots` changes: both gates live entirely in this plugin.
- A countdown/delay variant (`auto_start_delay_s`): YAGNI until an operator asks for it.
- Piped-stdin support under `auto_start` (stdin that is readable but not a TTY): `default_poll_end()` already ignores non-TTY stdin today, so such runs cannot end episodes interactively; they belong to `unattended`.
