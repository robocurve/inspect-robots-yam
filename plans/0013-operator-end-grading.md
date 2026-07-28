# Operator End-Episode Grading Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The end-episode keypress terminates the episode with a non-definitive `termination_reason="operator_end"` and asks nothing; grading (y/n/partial/skip + optional grader note) is owned by the framework's single operator prompt. Fixes [#79](https://github.com/robocurve/inspect-robots-yam/issues/79).

**Architecture:** The yam embodiment currently duplicates grading: `_poll_end()` → `OperatorIO.confirm_success()` (binary y/N) → `termination_reason="success"|"failure"`, which core's `_prompt_operator` adopts as definitive, skipping its richer prompt. This plan makes the embodiment *thin*: it only signals "the operator ended the episode" and owns no verdict. Core's side of the contract — prompting for any attended trial that ends with `"operator_end"`, registered tasks included — ships separately as [inspect-robots#194](https://github.com/robocurve/inspect-robots/issues/194).

**Tech Stack:** Python 3.12, uv, pytest. Run everything from the worktree root: `/home/robocurve/robocurve/worktrees/fix-operator-end-grading`.

## Global Constraints

- **Sequencing / dependency:** this PR must not merge before the inspect-robots release that implements #194 (grading trials that end with `"operator_end"`; expected 0.24.0) is on PyPI. Tasks 1-3 are independent of that release and run green on the current lock (verified: `before_scoring` exists in core 0.12.0 and 0.23.3); only Task 4 (the floor bump) requires the release to exist, which is why it is last and explicitly gated. Without the new core, attended **registered-task** runs (kitchenbench) would end ungraded and score 0. Ad-hoc runs with `scorer = operator` grade correctly even on core 0.23.
- Test command: `uv run pytest --cov -q` (baseline: 477 passed, 5 skipped — verified on both core 0.12.0 and 0.23.3 lockfiles). The coverage gate is `fail_under = 100` over `source = ["inspect_robots_yam"]` and only fires with `--cov` — plain `pytest -q` does not check it, so every commit gate below uses `--cov`. Deleted code must take its tests with it; new branches need tests.
- Lint/format gates: `uv run ruff check . && uv run ruff format --check .` must pass before every commit.
- **Breaking change, intended:** `OperatorIO` is exported in `__all__` (guarded by `test_api_snapshot.py`), so deleting `confirm_success` breaks the published API. That is the point of #79; it ships as a minor bump (0.16.0) and the release notes must say so. No deprecation shim.
- The new termination reason string is exactly `"operator_end"` everywhere (code, tests, docs) — it must match the constant core #194 standardizes.
- `OperatorIO.wait_ready`, `_drain_stdin`, and `default_poll_end` are untouched — the readiness prompt and end-episode keypress polling still work exactly as before.
- Unattended behavior is unchanged: `unattended=True` still skips the end poll entirely and runs to `max_steps`, and (because it never emits `"operator_end"`) core never prompts either.
- Scoring context you must not "fix" in this repo: core's `success_at_end` scorer reads only `termination_reason == "success"` and will score attended runs 0 after this change **by design** — the correct pairing for attended runs is core's `operator` scorer (registered name `"operator"`, reads `TrialRecord.operator_judgement`) or KitchenBench's `task_success` (already falls back to `operator_judgement`, `kitchenbench/scoring.py:26-31`). Docs task 3 carries this; no scorer code changes here.

---

### Task 1: Embodiment terminates with `operator_end` and asks nothing

**Files:**
- Modify: `src/inspect_robots_yam/embodiment.py` — the `_poll_end` block in `step()` (lines 1152-1164), the module-docstring bullet (lines 8-11), the status-line text (line 1124), and the settle-warning comment (lines 1493-1495)
- Test: `tests/test_embodiment.py` — replace the two verdict tests (lines 294-311), fix the status-message assertion (line 784) and the closed-line comment (line 818); `tests/test_settle.py` — drop the `operator_confirmed` assert (line 352); `tests/test_eval_end_to_end.py` — grade via `before_scoring` (lines 79-98)

**Interfaces:**
- Produces: `StepResult(terminated=True, termination_reason="operator_end", info=settle_info)` on the end-episode keypress. `info` no longer carries `operator_confirmed`. Task 2 relies on `step()` no longer calling `self._operator.confirm_success`.

- [ ] **Step 1: Replace the two verdict tests with operator_end tests**

In `tests/test_embodiment.py`, replace `test_step_terminates_success_on_operator_yes` and `test_step_terminates_failure_on_operator_no` (lines 294-311) with:

```python
def test_step_terminates_operator_end_without_prompting() -> None:
    prompts: list[str] = []
    emb, _, _ = _build(poll_end_seq=[True], operator=_operator(prompts=prompts))
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(14)))
    assert result.terminated is True
    assert result.termination_reason == "operator_end"
    assert "operator_confirmed" not in result.info
    # Grading is the framework prompt's job; the embodiment asks nothing.
    assert all("succeed" not in prompt for prompt in prompts)
```

Note `_operator()` (helper at `tests/test_embodiment.py:54-64`) takes `answers` first; passing only `prompts=` works because the readiness prompt returns `""` and nothing pops `seq` anymore.

- [ ] **Step 2: Run the new test to verify it fails**

Run: `uv run pytest tests/test_embodiment.py::test_step_terminates_operator_end_without_prompting -q`
Expected: FAIL with `IndexError: pop from empty list` — the old code still calls `confirm_success()`, whose "succeed" prompt makes the `_operator` helper pop an empty answer list (`tests/test_embodiment.py:60-62`).

- [ ] **Step 3: Implement the thin termination**

In `src/inspect_robots_yam/embodiment.py`, replace the block at lines 1152-1164:

```python
        obs = self._observe(self._instruction)
        # Unattended runs have no operator: skip the end poll entirely; the
        # episode runs to the framework's max_steps.
        if not self._cfg.unattended and self._poll_end():
            self._status(None)  # close the status line before control returns
            # The operator only signals *that* the episode is over. The verdict,
            # partial/skip, and grader notes belong to the framework's single
            # operator prompt, which a definitive reason here would suppress —
            # so the reason stays non-definitive (inspect-robots#194).
            return StepResult(
                observation=obs,
                terminated=True,
                termination_reason="operator_end",
                info=settle_info,
            )
        return StepResult(observation=obs, terminated=False, info=settle_info)
```

Update the module-docstring bullet (lines 8-11) to:

```
* **Operator-in-the-loop success** — there is no privileged success oracle; the
  operator's end-of-episode keypress returns
  ``StepResult(terminated=True, termination_reason="operator_end")`` and the
  human verdict (with optional grader notes) is captured afterwards by the
  framework's operator prompt and read by judgement-based scorers.
```

Update the status-line text (line 1124) from
`"Running: press any key to end the episode, then y/N to score.{limit}"` to:

```python
            self._status(f"Running: press any key to end the episode and grade it.{limit}")
```

Update the settle-warning comment (lines 1493-1495) from "the operator's y/N verdict is this embodiment's success signal and no human reads `StepResult.info`, so this is the practical notice..." to:

```python
            # logging, not warnings.warn: success is graded by the operator at
            # the framework prompt and no human reads StepResult.info, so this
            # is the practical notice that a trial's observations degraded.
```

- [ ] **Step 4: Run the new test to verify it passes**

Run: `uv run pytest tests/test_embodiment.py::test_step_terminates_operator_end_without_prompting -q`
Expected: PASS

- [ ] **Step 5: Fix the status-line and settle tests**

In `tests/test_embodiment.py:784`, replace
`assert "any key" in msg and "y/N" in msg  # how to end + how scoring works` with:

```python
    assert "any key" in msg and "grade" in msg  # how to end + how scoring works
```

In `tests/test_embodiment.py:818` (`test_status_finishes_with_none_when_operator_ends_episode`), update the trailing comment: `assert status[-1] is None  # line closed before control returns for grading`.

In `tests/test_settle.py::test_terminal_step_result_keeps_the_settle_keys` (line 352), replace `assert result.info["operator_confirmed"] is False` with `assert result.termination_reason == "operator_end"` (the test's point — settle keys survive on the terminal step — is unchanged).

Dead scaffolding: `_build_with_status` passes `_operator(["y"])` (`tests/test_embodiment.py:766`) whose answer is never consumed once no "succeed" prompt exists — change it to `_operator()`.

Stale comment: `tests/test_embodiment.py:403` says `# neither wait_ready nor confirm_success ran` — change to `# neither wait_ready nor the end poll ran` (this file is already staged by this task's commit; Task 3's sweep must not be left holding edits its commit doesn't stage).

- [ ] **Step 6: Fix the end-to-end test to grade via before_scoring**

In `tests/test_eval_end_to_end.py::test_eval_scores_success_end_to_end`, the embodiment no longer produces `termination_reason="success"`, so KitchenBench's `task_success` falls back to `TrialRecord.operator_judgement` — which, in the real composition (core ≥ 0.24, #194), the CLI captures via `eval(before_scoring=...)`. Emulate that hook:

```python
def _grade_yes(record, scene) -> None:
    del scene
    record.operator_judgement = "y"


def test_eval_scores_success_end_to_end() -> None:
    policy = MolmoAct2Policy(MolmoActConfig(cam_height=4, cam_width=4, num_steps=1), post_fn=_post)
    embodiment = YAMEmbodiment(
        YamConfig(cam_height=4, cam_width=4),
        driver_factory=lambda _c: _FakeDriver(),
        camera_reader=_cameras,
        operator=OperatorIO(input_fn=lambda _p: "", output_fn=lambda _m: None),
        poll_end=lambda: True,  # operator ends every episode immediately
        sleep_fn=lambda _d: None,
        clock=lambda: 0.0,
    )

    logs = rl_eval(
        "kitchenbench/stack", policy, embodiment, sinks=[], seed=0, before_scoring=_grade_yes
    )

    assert len(logs) == 1
    log = logs[0]
    assert log.status == "success"
    assert log.results.metrics["task_success"] == 1.0
```

`_always_yes_operator()` is now unused — delete the helper (it lives at `tests/test_eval_end_to_end.py:45` with exactly one caller). The other test in the file (lines 54-76) is unattended and asserts nothing verdict-derived — leave it alone. Update the module docstring (lines 1-4): it currently says the file proves "the termination_reason -> scorer wiring"; the scored test now proves the judgement-based wiring (`before_scoring` → `operator_judgement` → `task_success` fallback), so say that instead.

- [ ] **Step 7: Run the full suite and gates**

Run: `uv run pytest --cov -q && uv run ruff check . && uv run ruff format --check .`
Expected: all green. (`tests/test_operator.py`'s `confirm_success` tests still pass — they call the method directly on `OperatorIO`; Task 2 deletes method and tests together, which is also what keeps the coverage gate satisfied at each commit.)

- [ ] **Step 8: Commit**

```bash
git add src/inspect_robots_yam/embodiment.py tests/test_embodiment.py tests/test_settle.py tests/test_eval_end_to_end.py
git commit -m "feat(embodiment): end-episode keypress terminates with non-definitive operator_end (#79)"
```

---

### Task 2: Delete the duplicated grading UI from OperatorIO

**Files:**
- Modify: `src/inspect_robots_yam/operator.py` (delete `confirm_success` at lines 49-52 and `_AFFIRMATIVE` at lines 17-18; update module docstring lines 1-8)
- Test: `tests/test_operator.py` (delete the two `confirm_success` tests)

**Interfaces:**
- Consumes: Task 1 removed the only production caller (`embodiment.py` no longer calls `confirm_success`).
- Produces: `OperatorIO` with exactly two responsibilities — `wait_ready()` and (module-level) `default_poll_end` / `_drain_stdin`. No other task depends on this one.

- [ ] **Step 1: Delete the dead code**

In `src/inspect_robots_yam/operator.py`:
- Delete the `_AFFIRMATIVE` constant and its comment (lines 17-18).
- Delete the `confirm_success` method (lines 49-52).
- Update the module docstring's first paragraph to:

```python
"""Operator-in-the-loop I/O for real hardware runs.

The operator readies scenes and signals end-of-episode; the success verdict
itself (with optional grader notes) is collected afterwards by the framework's
operator prompt, so this module owns no grading UI. All stdin/stdout goes
through injectable ``input_fn`` / ``output_fn`` so tests drive these paths
without a real terminal. The one genuinely TTY-bound piece — the non-blocking
"operator pressed end" poll — is isolated in :func:`default_poll_end`, which is
excluded from coverage.
"""
```

- [ ] **Step 2: Delete the tests of the deleted API**

In `tests/test_operator.py`, delete exactly two tests: `test_confirm_success_affirmative` (line 38) and `test_confirm_success_negative` (line 45). The others (`test_wait_ready_calls_input`, `test_wait_ready_dead_stdin_raises_embodiment_fault`, `test_default_poll_end_is_callable`) stay.

- [ ] **Step 3: Run the full suite and gates**

Run: `uv run pytest --cov -q && uv run ruff check . && uv run ruff format --check .`
Expected: all pass, coverage gate green (the deleted branches took their tests with them).

- [ ] **Step 4: Commit**

```bash
git add src/inspect_robots_yam/operator.py tests/test_operator.py
git commit -m "refactor(operator): drop confirm_success — grading belongs to the framework prompt (#79)"
```

---

### Task 3: Docs — new flow, scorer pairing, stale-mention sweep

**Files:**
- Modify: `README.md:209` (config.ini suggestion), `README.md:284-289` (episode-end flow), `CLAUDE.md:64-65` (success-path invariant bullet), `src/inspect_robots_yam/CLAUDE.md:12` (operator.py row) and `:16` (embodiment.py row)

**Interfaces:**
- Consumes: the `operator_end` reason and prompt flow from Tasks 1-2. Nothing downstream.

- [ ] **Step 1: Update the config.ini suggestion**

`README.md:209`: change

```
scorer = success_at_end    # scores the operator's y/N answer at episode end
```

to

```
scorer = operator          # scores the verdict you type at the end-of-episode prompt
```

- [ ] **Step 2: Update the CLI flow line and rewrite the episode-end flow paragraph**

`README.md:261` ("press any key to end the episode, answer y/N to score") is the CLI-section flow line — change it to: "press any key to end the episode, then grade it at the prompt that follows".

Then replace `README.md:284-289` ("At each episode end the embodiment asks the operator (y/N); a `yes` records ... scoring as a failure."). **Context matters here:** this paragraph sits in the *Python API* section, right after a bare `eval("kitchenbench/pour_pasta", pol, emb, ...)` snippet — and `eval()` itself never prompts; prompting is the CLI's `before_scoring` hook. The replacement must distinguish the two:

```markdown
Pressing the end-episode key terminates the episode with
`termination_reason="operator_end"` — the embodiment itself asks nothing.
On CLI runs (inspect-robots ≥ 0.24), the framework then asks once per trial:
`did the robot succeed? [y/n/partial/skip]` plus an optional grader note.
The bare `eval()` call above never prompts: pass
`before_scoring=` a callable that sets `record.operator_judgement` (grade
live, or from your own UI) when driving the Python API directly.
Score attended runs with the `operator` scorer (reads the recorded judgement);
KitchenBench's `task_success` reads it too. `success_at_end` only counts
embodiment-detected success terminations, so it scores operator-graded runs as
failures — don't pair it with attended yam runs.
The operator prompts need an interactive terminal: a dead stdin raises
`EmbodimentFault` (the framework's always-halt path). For runs with no operator,
set `YamConfig(unattended=True)` (CLI: `-E unattended=true`): all operator
prompts are skipped and every episode runs to `max_steps`, scoring as a failure.
```

- [ ] **Step 3: Update both CLAUDE.md files**

Root `CLAUDE.md:64-65`: replace the bullet

```
- Success reaches the scorer **only** via `StepResult.termination_reason="success"`
  (stock `rollout` never sets `operator_judgement`).
```

with:

```
- The end-episode keypress terminates with `termination_reason="operator_end"`;
  the framework prompt then records `operator_judgement`, which is what
  judgement-reading scorers (`operator`, KitchenBench `task_success`) score.
  `success_at_end` counts only embodiment-detected `"success"` terminations,
  which stock yam never emits.
```

`src/inspect_robots_yam/CLAUDE.md:12`: change the `operator.py` row description to: `` `OperatorIO` (injectable stdin/stdout) for the readiness prompt; `default_poll_end` (real TTY poll, `# pragma: no cover`). Verdicts + grader notes are the framework prompt's job. ``

`src/inspect_robots_yam/CLAUDE.md:16`: in the `embodiment.py` row, change "operator-keypress success" to "operator-keypress episode end (`operator_end`)".

- [ ] **Step 4: Sweep for stale mentions**

Run: `grep -rn "y/N\|confirm_success\|success_at_end\|operator_confirmed" README.md src/ tests/ CLAUDE.md`
Expected: only hits that are correct in the new flow — Tasks 1-2 already fixed the test-file mentions (including `tests/test_embodiment.py:403`), so a `tests/` hit here means an earlier task was skipped; do not fix files outside this task's commit list. Judge each remaining hit — e.g. `README.md:31` ("operator-in-the-loop success" feature bullet) stays accurate; a hit that merely *names* `success_at_end` while warning against it (Task 3 Step 2's new text) is correct and stays.

- [ ] **Step 5: Run gates and commit**

Run: `uv run pytest --cov -q && uv run ruff check . && uv run ruff format --check .`
Expected: PASS (docs-only, but the gates are cheap insurance).

```bash
git add README.md CLAUDE.md src/inspect_robots_yam/CLAUDE.md plans/0013-operator-end-grading.md
git commit -m "docs: operator_end grading flow and operator-scorer pairing (#79)"
```

---

### Task 4: Bump the inspect-robots floor and lockfile (GATED on the core release)

**Files:**
- Modify: `pyproject.toml:23` (`"inspect-robots>=0.12",  # EmbodimentInfo.docs (plan 0016)`), `uv.lock`

**Interfaces:**
- Consumes: the published inspect-robots release implementing #194. Produces: a lockfile on the core version whose prompt contract Tasks 1-3 describe.

- [ ] **Step 0: GATE — verify the release exists**

Run: `uv pip index versions inspect-robots 2>/dev/null || curl -s https://pypi.org/pypi/inspect-robots/json | python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"`
Expected: a version ≥ 0.24.0 implementing inspect-robots#194. **If it does not exist yet, STOP here** — Tasks 1-3 are complete and green on the current lock; report that Task 4 is parked on the core release. Do not bump the floor to an unpublished version (`uv lock` would fail resolution).

- [ ] **Step 1: Bump the floor**

In `pyproject.toml:23`, change the dependency to (substitute the actual released version if it isn't 0.24.0):

```toml
    "inspect-robots>=0.24",  # operator_end grading contract (#79, inspect-robots#194)
```

- [ ] **Step 2: Refresh the lock and sync**

Run: `uv lock --upgrade-package inspect-robots && uv sync --extra dev`
Expected: `Updated inspect-robots v0.12.0 -> v0.24.x`. (The 0.12→0.23.3 jump was already verified clean: 477 passed.)

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest --cov -q`
Expected: 477 passed, 5 skipped (counts shift with Tasks 1-2's test changes; the point is all-green). Any failure here is core API drift — stop and report it rather than patching around it.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): require inspect-robots >= 0.24 for the operator_end grading contract (#79)"
```
