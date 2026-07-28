# Parallel Arm Bring-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `_default_driver_factory` brings up the left and right YAM arms concurrently instead of sequentially, cutting driver init wall-clock from `left + right` to `max(left, right)` (each arm's mandatory gripper auto-calibration is 2–5 s), and — fixing a latent leak in the sequential code — tears down whichever arm succeeded when the other one fails to initialize. Fixes [#83](https://github.com/robocurve/inspect-robots-yam/issues/83).

**Architecture:** `_default_driver_factory` (`src/inspect_robots_yam/embodiment.py:178-211`) is a `# pragma: no cover` hardware seam, so the new orchestration logic must not live inside it or the 100% coverage gate goes blind to it. Instead, a new pure-orchestration helper `start_arms_concurrently(left_factory, right_factory)` goes in `src/inspect_robots_yam/_i2rt.py` (already home to the driver-lifecycle workaround `close_robot_safely`): it runs two zero-arg factories on a two-worker `ThreadPoolExecutor`, returns `(left, right)` when both succeed, and on any failure closes each survivor via `close_robot_safely` (guarded so a close error cannot mask the init error) before re-raising. The pragma'd factory shrinks to building the two hardware thunks and calling the helper. The helper is fully testable with fakes — including a deterministic proof of actual concurrency via a `threading.Barrier`.

**Why concurrent bring-up is safe (verified against the pinned i2rt, db582ea):**
- Each arm is independent hardware: its own CAN channel (`cfg.left_channel` / `cfg.right_channel`, separate socketcan sockets), its own `DMChainCanInterface` control thread, its own MuJoCo model instance.
- `combine_arm_and_gripper_xml` writes via `tempfile.NamedTemporaryFile` (unique path per call) — no shared-file race.
- `_load_arm_config` / `_load_gripper_config` are read-only YAML loads; `logging` is thread-safe.
- Gripper auto-calibration only moves the gripper jaws, never the arm joints — both arms calibrating simultaneously is mechanically inert.
- `close_robot_safely` discovers i2rt's control threads by `__self__ is chain`, which is unaffected by which thread constructed the chain.

**Tech Stack:** Python 3.12, uv, pytest. Run everything from the worktree root: `/home/robocurve/robocurve/worktrees/parallel-arm-init`.

## Global Constraints

- Test command: `uv run pytest --cov -q` (baseline: 509 passed, 5 skipped). The coverage gate is `fail_under = 100` and only fires with `--cov`, so every commit gate uses `--cov`. Every line of the new helper must be covered.
- Lint/format/type gates: `uv run ruff check . && uv run ruff format --check . && uv run mypy` must pass before every commit (CI runs mypy strict as a blocking step; the package advertises `mypy --strict` clean). Note the second teardown loop's variable is deliberately named `error`, not `exc` — reusing an except-binding name as a loop target is rejected by mypy strict (the binding is implicitly deleted at handler exit).
- No public API change: `start_arms_concurrently` lives in the private `_i2rt` module and is **not** added to the package `__all__`; `tests/test_api_snapshot.py` must pass unchanged.
- Behavior contract preserved: on success the driver is indistinguishable from today's (`_Real` unchanged: same packing order, same close path). On a single-arm init failure the original exception type and message propagate exactly (survivor teardown must not replace or wrap it).
- If both arms fail, raise the **left** arm's exception (deterministic order) and log the right arm's at ERROR — do not lose it silently.
- A survivor whose `close_robot_safely` itself raises must not mask the init failure: guard the close with `try/except Exception` + `logger.warning`, then re-raise the init error.
- No join timeout in the helper: if one arm's bring-up hangs, the process waits — exactly the sequential code's behavior today. Do not add a timeout knob nobody asked for.
- Hardware cannot be exercised here (and the rig's right-side CAN is currently down): validation is tests + review; note rig verification as a follow-up in the PR body.

---

### Task 1: `start_arms_concurrently` in `_i2rt.py`, test-first

**Files:**
- Modify: `src/inspect_robots_yam/_i2rt.py` — add `start_arms_concurrently` below `close_robot_safely`
- Test: `tests/test_i2rt.py` — new test class/section for the helper

**Interfaces:**
- Produces: `start_arms_concurrently(left_factory: Callable[[], Any], right_factory: Callable[[], Any]) -> tuple[Any, Any]`. Task 2 calls it from `_default_driver_factory`.

- [ ] **Step 1: Write the tests (expected to fail: `ImportError`)**

Append to `tests/test_i2rt.py` (match the file's existing fake style; `close_robot_safely` falls back to plain `robot.close()` when the fake has no `motor_chain`, so a `close()`-recording fake suffices):

1. `test_start_arms_concurrently_returns_left_right_in_order` — factories return sentinel objects; assert the tuple is `(left_sentinel, right_sentinel)`.
2. `test_start_arms_concurrently_actually_overlaps` — both factories wait on a shared `threading.Barrier(2, timeout=5)` before returning; under sequential execution the first factory would raise `BrokenBarrierError` after the 5 s timeout (a clean failure, not a hang), so passage proves overlap. Assert the returned tuple too.
3. `test_left_failure_closes_right_and_reraises` — left factory raises `RuntimeError("left CAN down")`; right returns a fake recording `close()` calls. Assert `pytest.raises(RuntimeError, match="left CAN down")` and the right fake was closed exactly once.
4. `test_right_failure_closes_left_and_reraises` — mirror of 3.
5. `test_both_failures_raise_left_and_log_right` — each factory constructs a `close()`-recording fake and then raises a distinct exception (so there is something to assert about); assert the left exception propagates, `caplog` (ERROR level) contains the right one's message, and neither fake's `close()` ran (nothing succeeded, so no teardown).
6. `test_survivor_close_failure_does_not_mask_init_error` — left raises `RuntimeError("init boom")`; right's `close()` raises `ValueError`. Assert the `RuntimeError` still propagates and `caplog` (WARNING) records the close failure.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_i2rt.py -q`
Expected: one **collection error** for the whole file (the module-level import of `start_arms_concurrently` fails), which takes the existing tests in the file down with it. That is the expected TDD checkpoint — nothing else is broken.

- [ ] **Step 3: Implement the helper**

In `src/inspect_robots_yam/_i2rt.py`, add exactly these imports: `from collections.abc import Callable` (NOT `from typing` — ruff UP035 fires on that under this repo's config) and `from concurrent.futures import ThreadPoolExecutor`. Then add below `close_robot_safely`:

```python
def start_arms_concurrently(
    left_factory: Callable[[], Any],
    right_factory: Callable[[], Any],
) -> tuple[Any, Any]:
    """Bring up both arms concurrently, tearing down the survivor on failure.

    Each arm's driver init includes a mandatory gripper hard-stop calibration
    (multiple seconds), and the arms are fully independent hardware (own CAN
    channel, own control thread), so running the two factories on worker
    threads cuts bring-up wall-clock to max(left, right). If either factory
    raises, any arm that did come up is closed via ``close_robot_safely`` —
    the sequential code leaked it (robocurve/inspect-robots-yam#83) — and the
    init error is re-raised (left's first when both fail).
    """
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="yam-arm-init") as pool:
        futures = [pool.submit(left_factory), pool.submit(right_factory)]
        results: list[Any] = []
        errors: list[tuple[str, BaseException]] = []
        for side, future in zip(("left", "right"), futures, strict=True):
            try:
                results.append(future.result())
            except BaseException as exc:  # re-raised below
                errors.append((side, exc))

    if not errors:
        return results[0], results[1]

    for robot in results:
        try:
            close_robot_safely(robot)
        except Exception:
            logger.warning("closing the surviving arm after an init failure failed", exc_info=True)
    for side, error in errors[1:]:
        logger.error("%s arm bring-up also failed: %s", side, error, exc_info=error)
    raise errors[0][1]
```

Notes for the implementer:
- `zip` over `("left", "right")` keeps result/error attribution deterministic (`strict=True` is required by ruff B905 and correct — both iterables are length 2); `errors[0]` is left's when both fail because futures are consumed in submission order.
- The `results` list holds only successes, so the teardown loop is exactly "survivors".
- Do not narrow `BaseException` to `Exception` in the gather. Real semantics to preserve (and not "improve"): Ctrl-C is delivered to the *main* thread and interrupts `future.result()`'s wait, so a `KeyboardInterrupt` is recorded as that side's failure and triggers survivor teardown; an arm whose factory is still running at that moment can leak on Ctrl-C — exactly like today's sequential code. No worse, no redesign.

- [ ] **Step 4: Run the tests + gates**

Run: `uv run pytest --cov -q && uv run ruff check . && uv run ruff format --check . && uv run mypy`
Expected: 515 passed, 5 skipped; coverage 100%; lint clean.

- [ ] **Step 5: Commit**

`git commit` with a message explaining concurrent bring-up + survivor teardown.

---

### Task 2: `_default_driver_factory` uses the helper

**Files:**
- Modify: `src/inspect_robots_yam/embodiment.py:178-211` — construct thunks, call `start_arms_concurrently`
- Modify: `src/inspect_robots_yam/CLAUDE.md` — `_i2rt.py` row: mention `start_arms_concurrently`
- Modify: `CHANGELOG.md` — Unreleased → Changed

**Interfaces:**
- Consumes: `start_arms_concurrently` from Task 1. `_Real` and everything downstream unchanged.

- [ ] **Step 1: Rewrite the factory head**

Replace `embodiment.py:185-194` with (note the leading blank line — the retained line 184 `gripper = GripperType[cfg.gripper_type]` needs one before the nested `def`, or `ruff format --check` fails):

```python

    def _make_arm(channel: str) -> Any:
        return get_yam_robot(
            channel=channel,
            gripper_type=gripper,
            zero_gravity_mode=cfg.zero_gravity_mode,
        )

    # Both arms are independent hardware (own CAN channel, own control
    # thread) and each pays a multi-second gripper calibration on every
    # boot (encoder frame resets at power-off), so bring them up together.
    left, right = start_arms_concurrently(
        lambda: _make_arm(cfg.left_channel),
        lambda: _make_arm(cfg.right_channel),
    )
```

Add `start_arms_concurrently` to the existing `from inspect_robots_yam._i2rt import (...)` statement at `embodiment.py:43-48` (it is an absolute import, not `from ._i2rt`). The whole factory stays `# pragma: no cover` — it now contains only hardware thunk construction.

- [ ] **Step 2: Docs**

- `src/inspect_robots_yam/CLAUDE.md`, `_i2rt.py` row: add that it also owns `start_arms_concurrently` (concurrent bring-up + survivor teardown, #83).
- `CHANGELOG.md` Unreleased → Changed: concurrent arm bring-up (init wall-clock ≈ max of the two arms instead of their sum) and Fixed: an init failure on one arm no longer leaks the other arm's control thread, CAN socket, and torque-enabled motors.
- Check `README.md` for any sequential-init or startup-time claims (`grep -in "calibrat\|sequential\|bring.\?up" README.md`) and update if any exist; skip if none.

- [ ] **Step 3: Run all gates**

Run: `uv run pytest --cov -q && uv run ruff check . && uv run ruff format --check . && uv run mypy`
Expected: 515 passed, 5 skipped; coverage 100%; lint clean (`test_api_snapshot.py` and `test_embodiment_docs.py` in particular must pass untouched).

- [ ] **Step 4: Commit**
