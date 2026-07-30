# `auto_start` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `YamConfig.auto_start` so an attended run homes the arms and starts the episode with zero operator Enter presses, while keeping every other attended behavior (issue #87).

**Architecture:** A single boolean on the frozen `YamConfig` dataclass, consumed only inside `YAMEmbodiment.reset()`. When set (and `unattended` is false), the two `OperatorIO.wait_ready()` gates are skipped: the stand-clear gate is replaced by a one-line printed notice (once per connection, same lifecycle as `_home_gate_confirmed`), and the scene-ready gate is replaced by a bare stdin drain so a buffered newline cannot trip the end-episode poll on step 1. `unattended=True` takes precedence and makes `auto_start` a no-op.

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

`OperatorIO.wait_ready()` (`src/inspect_robots_yam/operator.py:27`) both blocks on Enter and calls `_drain_stdin()` afterwards. The drain is load-bearing: `default_poll_end()` treats any buffered stdin line as an "end episode" keypress, so skipping `wait_ready()` without draining would end the episode on the first step whenever a stray newline is buffered.

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
    # BEFORE launching the run. unattended=True takes precedence and makes
    # this a no-op.
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
- Modify: `src/inspect_robots_yam/embodiment.py` (import at line 60; gates at lines 1096-1131; `reset()` docstring at line 1055)
- Test: `tests/test_embodiment.py`

**Interfaces:**
- Consumes: `YamConfig.auto_start` from Task 1; `_drain_stdin` from `inspect_robots_yam.operator`.
- Produces: the runtime behavior documented in Task 3's README text. No new public symbols.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_embodiment.py` (near `test_unattended_skips_operator_prompts`, line ~380). `Scene`, `Action`, `np`, `OperatorIO`, `YamConfig`, and `_build` are already in scope; add `import inspect_robots_yam.embodiment as embodiment_module` next to the existing `from inspect_robots_yam.embodiment import YAMEmbodiment` import at the top of the file.

```python
def test_auto_start_skips_gates_but_keeps_attended_flow() -> None:
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


def test_auto_start_notice_prints_once_per_connection() -> None:
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
    assert lines == []  # no stand-clear notice either: unattended stays silent
    assert result.terminated is False  # unattended still disables the end poll
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embodiment.py -k auto_start -v` and `uv run pytest tests/test_embodiment.py::test_unattended_precedes_auto_start -v`
Expected: the first three FAIL (prompts are still captured / no notice / no module-level `_drain_stdin` to patch — the monkeypatch raises `AttributeError`); `test_unattended_precedes_auto_start` PASSES already (unattended short-circuits both gates today). Keep it: it pins the precedence contract against regressions.

- [ ] **Step 3: Implement**

In `src/inspect_robots_yam/embodiment.py`:

1. Extend the import at line 60:

```python
from inspect_robots_yam.operator import OperatorIO, _drain_stdin, default_poll_end
```

2. Replace the home gate (lines 1096-1100):

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

3. Replace the scene-ready gate (lines 1127-1128):

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

(The `horizon` / `Running:` status lines that follow stay exactly as they are, inside the `if not self._cfg.unattended:` block.)

4. Update the `reset()` docstring (line 1055) to:

```python
        """Connect (if needed), drive to home, and block on operator readiness.

        With ``auto_start`` set, both operator gates are skipped: a printed
        notice replaces the stand-clear prompt and the episode begins as soon
        as the arms settle at the home pose. ``unattended`` skips them too,
        along with the rest of the attended flow, and takes precedence.
        """
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embodiment.py -v`
Expected: all PASS, including the four new tests and every pre-existing prompt/status test (defaults unchanged).

- [ ] **Step 5: Run the full gate set**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest --cov -q`
Expected: clean, coverage 100%. If coverage flags the new branches, a test above is not exercising them — fix the test, do not add pragmas (none of this code is hardware-bound).

- [ ] **Step 6: Commit**

```bash
git add src/inspect_robots_yam/embodiment.py tests/test_embodiment.py
git commit -m "embodiment: auto_start skips both operator Enter gates (#87)"
```

---

### Task 3: Documentation

**Files:**
- Modify: `README.md` (attended-flow paragraph around line 278, unattended paragraph around line 315, config reference around line 644)
- Modify: `src/inspect_robots_yam/CLAUDE.md` (module map row for `embodiment.py` mentions operator-keypress flow; extend only if it names the prompts — otherwise leave it)

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
grading. Prefer `unattended=true` only when no operator is present at all,
since it also disables those.
```

Adjust the lead-in wording so the existing paragraph and the new one read as one section; keep the existing safety note about the e-stop nearby if one is present.

- [ ] **Step 2: README config reference (~line 644)**

Extend the config list entry right after `unattended`:

```markdown
`auto_start` (default `False`; skip both operator Enter gates but keep the
attended episode flow; `unattended` takes precedence),
```

Match the exact formatting of the neighboring entries (backticks, semicolons, trailing comma).

- [ ] **Step 3: Verify docs style**

Re-read both edits against the repo writing rules: no em dashes in prose, no bold mid-sentence, headers unchanged. Run `uv run pytest -q` once more (README edits cannot break tests, but the commit gate below expects a green tree).

- [ ] **Step 4: Commit**

```bash
git add README.md src/inspect_robots_yam/CLAUDE.md
git commit -m "docs: document auto_start zero-touch starts (#87)"
```

---

## Out of scope

- Changing the default (prompts stay on for everyone who has not opted in): the stand-clear gate is a safety feature and CLAUDE.md forbids weakening safety defaults.
- Core `inspect-robots` changes: both gates live entirely in this plugin.
- A countdown/delay variant (`auto_start_delay_s`): YAGNI until an operator asks for it.
