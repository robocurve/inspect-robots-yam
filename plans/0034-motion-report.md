# 0034 — Motion report: IK diagnostics for the policy

Status: implemented 2026-10-13 (CoRL harness rework)
Consumer: `inspect-robots-agent` 0.30.0 renders `extra["motion_report"]["lines"]`.

## 1. Problem

Under `eef_pos` a move that falls short looks the same to the policy whether
the target was unreachable as posed, the arm hit a joint limit, the command
was still rate-limited, the controller froze the arm in an oscillation hold,
or the arm was held off its setpoint by contact. The policy saw only the
measured readback and an approver count, and the transcripts show models
re-sending targets the IK could never pose. Every one of those causes is
already known inside the kinematics wrapper and discarded.

## 2. Design

- `kinematics._ArmKinematics.solve()` records a `_SolveTrace` (held, hold
  remaining, resynced, non-finite, converged flag, joints on an effective
  range bound within 1e-3 rad, rate-limited, clipped solution). `report(
  q_sent, q_measured)` turns it into a frozen `SolveReport` with three pose
  errors (solution, command, tracking: each a position distance in meters and
  a geodesic angle in radians) using FK of the solution, the sent command,
  and the measured joints. Solution errors are `None` on hold and non-finite
  steps. `seed()` and `clear()` forget the trace.
- `YAMEmbodiment.step()` calls `report()` per arm after `_observe()` with the
  post-step joints, adds it to a per-arm `_MotionWindow` (counts plus the
  last report), and attaches `extra["motion_report"]` to the observation:
  `interface`, `complete`, `arms` (structured record and `cause`), `lines`.
  The window closes when the stepped action carries `meta["chunk_final"]`,
  the agent policy's marker for the last waypoint of a move; policies that do
  not mark chunks get a trial-long window of O(1) counters. `reset()` clears
  the windows.
- Cause priority: `ik_failed` > `oscillation_hold` > `unreachable`
  (solution error > 1 cm or > 0.1 rad) > `unfinished` (rate-limited at the
  last step) > `behind` (tracking error > 2 cm) > `on_target`. A joint on a
  limit is named inside the `unreachable` line and appended to `on_target`.
- Joint control reports joints more than 0.1 rad behind their command on the
  current step, with no window.
- The built-in `_DOCS_*` strings tell the model the report exists and to
  trust it over guesses.

## 3. Non-goals

The solver's own success flag is recorded (`converged`, counted as
`nonconverged_steps`) but never drives a cause: plan 0006 documents that it
fails on its orientation threshold at sub-millimetre position error.
Contact is not inferred here; `behind` states the gap and the rig facts
sheet teaches the effort signatures.
