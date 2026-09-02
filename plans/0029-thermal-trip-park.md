# 0029: A thermal trip parks to rest to cool, in every mode

Issue: #150 part 1 (follow-up to #144 / plan 0028). Branch:
`feat/thermal-trip-park`. Part 2 of #150 (the wizard slot) is plan 0030,
deliberately a separate PR: it needs an unreleased core API, and this safety
fix must not wait on it.

## Problem

A mid-run trip ramps to rest only in graded runs with `park_before_grade=true`
(the framework's `observe_parked` lifecycle, plan 0028's documented caveat).
In ungraded or unattended runs the arms keep holding their trip pose under
power until `close()`. Holding pose under torque generates heat, which is
exactly wrong after a thermal trip; the arm should return to the
gravity-stable rest pose to cool in every mode.

## Design

In `step()`'s trip branch (added by plan 0028): after the notice line and
BEFORE returning the terminated `StepResult`,

1. capture the returned observation first (`self._observe(self._instruction)`)
   so the log documents the scene as it was at the trip, not after parking —
   verified safe for grading: `record.parked_observation` is populated only
   by `observe_parked()` (framework `eval.py:577-593`) and the VLM grader
   reads only that field, so graded runs still get post-park frames from
   `observe_parked`'s own capture;
2. resolve the park target: `cfg.rest_pose` if set, else `self._init_pose`.
   **If both are None, skip the park** (the notice already printed) and
   return the terminated result as today. This state is reachable: plan
   0028's pre-run gate sits before the `_init_pose` capture by design, so a
   caller that swallows the gate's `EmbodimentFault` and calls `step()` on a
   still-hot rig with `rest_pose = none` arrives here with no target. The
   guard mirrors `close()`'s `if self._init_pose is not None:`
   (`embodiment.py:1832`) and `observe_parked`'s early return (`:1767`), and
   the narrowing branch gets its own test;
3. emit a status line ("thermal guardrail: parking to rest to cool") and
   close it in a `finally` — both wrapped in `if not self._cfg.unattended:`,
   restating `observe_parked`'s guard verbatim (`embodiment.py:1778-1785`:
   status open AND close are inside the unattended guard);
4. ramp via the existing `self._ramp_to(target)` + `self._settle(sent)`,
   exactly the `observe_parked()` sequence;
5. return the terminated `StepResult` built from the pre-park observation.

Retained invariants, restated at the point of modification:

- Every ramp waypoint passes `_send()`'s hard joint clamp — the clamp stays
  the unmodified last line of defense.
- The termination reason stays the non-definitive `"overheat"`: graded runs
  still reach `observe_parked()` and the verdict prompt; repeat ramps to the
  same target are motion no-ops (see Risks for the wall-time cost).
- **`park_before_grade=false` is deliberately overridden by a thermal trip**:
  that flag expresses a scene-preservation preference for grading, and
  thermal safety outranks it (`close()` already ramps unconditionally). The
  trip park runs regardless; stated in the README paragraph too.
- The trip branch runs before the EEF conversion, so joint-space parking from
  an EEF-mode trial is exactly what `observe_parked`/`close` already do;
  kinematics re-seed on the next reset.
- Settle bookkeeping: the trip park's settle CAN increment `settle_timeouts`
  and even set `_settle_disabled` at trial end — accepted: the trial is over,
  `observe_parked`'s subsequent settle skip is the mechanism working as
  designed, and the budget-exhaustion warning's "rest of this trial" wording
  firing at trial end is cosmetic. The trip result's `info` keeps carrying
  only `"overheat"`; plan 0028's "no settle ran" comment is updated to say
  the park settle is deliberately not reported (the trial is already over).

## Tests (100% line+branch)

- mid-run trip parks: driver commands end at the rest pose (last command ==
  `cfg.rest_pose` — identity holds because the default gripper range is
  `gripper_open=1.0`/`gripper_closed=0.0`, so `_send`'s de-norm is the
  identity; the test keeps the default gripper config), terminated/`overheat`
  unchanged, and the returned observation reflects the trip pose, not the
  rest pose (EchoDriver-style fake).
- `rest_pose = none` variant parks to `_init_pose`.
- no-target guard: pre-run gate fault swallowed (hot at reset,
  `rest_pose=none`), then a hot `step()` → trip returns terminated with NO
  ramp commanded (commands unchanged).
- status line: recording session sees the park status then None (attended);
  the unattended e2e below covers the skip branch of the status guard.
- e2e (`tests/test_eval_end_to_end.py`): an **ungraded, unattended** eval
  (`before_scoring=None`, `unattended=True`, the `:78` precedent) whose trip
  still ends with the driver at rest pose.
- Sanctioned updates, all plan-0028 tests (pre-0028 assertions untouched):
  exactly one existing assertion changes —
  `tests/test_embodiment.py:472` (`len(driver.commands) == command_count` in
  `test_motor_temp_mid_run_trip_uses_session_notice_and_skips_motion`)
  becomes "no command from the policy action; the only commands are the park
  ramp ending at rest". The notice (`len(lines) == 1`) and status
  (`statuses[-1] is None`) assertions keep passing as written; the 0028
  graded e2e passes unchanged. Enumerate anything further in the report.

## Docs/meta

- README: the plan-0028 caveat sentence ("Ungraded or unattended runs hold
  until close()") is replaced: a trip parks to rest immediately in every
  mode, including `park_before_grade=false` (thermal safety outranks the
  scene-preservation preference).
- CHANGELOG `## Unreleased` / `### Changed`: prose entry referencing (#150).
- Root CLAUDE.md safety bullet: extend the plan-0028 line — the trip parks to
  rest from inside `step()` before terminating.
- src/inspect_robots_yam/CLAUDE.md: embodiment.py row mentions the trip park.

## Out of scope

- The wizard slot (plan 0030, after core 0.58 ships NUMBER_SLOTS).
- No pre-run-gate park (the arms have not moved; refusing in place stays
  correct), no cooldown wait/retry loop, no floor bump (this part runs on
  0.57).

## Risks

- The trip path commands motion from inside `step()` after declaring the
  trial over — the same ramp `observe_parked`/`close` already perform,
  through the same clamp. A second motor faulting during the park ramp
  propagates as `EmbodimentFault` exactly as it would today without the
  ramp; the `finally` still closes the status line.
- Wall time: `rest_secs` defaults to **3.0 s** (`config.py:178`), and a
  graded trip now stacks up to three ramps to the same target (trip park +
  `observe_parked` + `close`), about 9 s of redundant-but-harmless ramp/sleep
  at defaults. Accepted: trips are rare and terminal.
