# Ship a factory default rest_pose

Date: 2026-07-14
Status: revised after critique round 1 (blank opt-out replaced by the core's
existing `none` spelling; park gated on a completed first reset; provenance
clarified; test sweep enumerated); round 2 pending

## Problem

All YAM pairs share geometry, so the gravity-safe resting pose is
rig-invariant, yet every fresh install starts with `rest_pose = None`. On the
released 0.8.0 that means close() releases torque wherever the episode ended
(arms fall from task poses); since #36 it means parking at whatever pose the
arms happened to be in at first reset. Neither gives a fresh user the
canonical resting pose, and the setup wizard never prompts for one (#43).

## Captured reference pose

Read from a physical YAM pair posed at rest on 2026-07-14 (zero-gravity
connect; `gripper_limits_override` used, so only the gripper range sweep was
skipped — joint readings come from the motors' own encoders and do not
depend on that calibration; the two gripper slots are hand-set to 0.0 =
open, not read):

```
0.0372,0.0044,0.0006,-0.0898,-0.0517,-0.1539,0.0,0.0082,0.0010,0.0006,-0.0818,0.0727,0.1410,0.0
```

Layout per `packing.py`: left arm indices 0-6, right arm 7-13, gripper last
per arm, grippers normalized. Provenance comment in code must record: date,
physical capture, grippers hand-normalized open, and the standard-upright-
mounting assumption. Exotic mounts override per rig.

## Design

1. `config.py`: module-level `DEFAULT_REST_POSE: tuple[float, ...]` with the
   captured values and the provenance comment above. Not exported in
   `__all__` (tests/test_api_snapshot.py unchanged); it is reachable as
   `inspect_robots_yam.config.DEFAULT_REST_POSE` for the README to name.
2. `YamConfig.rest_pose` default changes `None` -> `DEFAULT_REST_POSE`; the
   type stays `tuple[float, ...] | None`.
3. Opt-out is the core's existing spelling, no plugin parsing changes:
   `rest_pose = none` in config.ini or `-E rest_pose=none` already reaches
   the plugin as Python `None` (core `_defaults.parse_value`). `None` means
   #36's captured-init fallback. A blank value stays a loud `ValueError`
   (it catches half-written lines; do not make it meaningful).
4. Park gating — close() parks only after a completed first reset. Today an
   explicitly-set rest_pose ramps on close() even when connect() faulted
   before any reset captured a pose; with a factory default that would mean
   commanding motion at teardown of a session that never commanded motion.
   New rule, applied uniformly to explicit and default values: if no reset
   has completed, close() releases in place (the pre-reset arms are wherever
   the operator left them). After a reset: explicit-or-default tuple ->
   ramp there; `None` -> captured-init pose. This is a deliberate behavior
   change for the explicit-rest_pose-without-reset corner and must be
   called out in the PR body.
5. Park path (not just endpoint): the park ramp is the same linear
   joint-space interpolation as #36's captured-init park and reset()'s
   homing ramp (`_ramp_to`); no waypoint is collision-checked, and the
   per-joint clamp bounds waypoints, not the swept volume. The factory
   target is a folded near-zero pose adjacent to typical operator init
   poses, so the sweep class is unchanged from #36; README must say the
   park path is not collision-checked and the workspace should be clear at
   episode end (this is true today and merely undocumented). Unattended
   runs (`unattended=True`) park silently like every other close(); the
   reset gate above is what prevents motion on never-reset teardowns.
6. README: update the `rest_pose` config-table row, the worked example
   (currently shows grippers 1.0), and the "Park pose must rest under
   gravity" safety bullet: factory default, per-rig override, `none`
   opt-out, path-not-collision-checked warning, and one sentence noting the
   default parks with grippers open, so anything still held is released at
   park. Keep every existing safety qualifier.
7. Stale prose sweep: the `rest_pose` field comment (config.py:99-101), the
   close() docstring (embodiment.py:309-316), and plans/0002-rest-pose-
   design.md gets an "Amended by PR #44" note mirroring the existing #36
   amendment style.

## Tests

New:
- `DEFAULT_REST_POSE`: 14 entries; gripper slots (6, 13) in [0, 1]; arm
  slots within `_DEFAULT_LOW`/`_DEFAULT_HIGH` (regression guard against a
  recapture typo); `YamConfig()` constructs with it.
- `YamConfig().rest_pose == DEFAULT_REST_POSE`.
- kwargs `rest_pose=None` (the parsed form of `none`) -> captured-init
  fallback still works end to end.
- close() with default config after a reset ramps to `DEFAULT_REST_POSE`
  (fake driver records waypoints; final target equals the constant).
- close() before any reset releases in place for BOTH default and explicit
  rest_pose (the new gate).
- Explicit override wins over the default.

Existing tests that break (enumerated; #36's invariants must be
re-expressed under `rest_pose=None`, not weakened):
- tests/test_config.py:113-116 `test_yam_rest_defaults`
- tests/test_embodiment.py:364 `test_close_without_rest_pose_ramps_to_captured_init_pose`
- tests/test_embodiment.py:395 `test_close_init_pose_grippers_round_trip_through_normalized_units`
- tests/test_embodiment.py:410 `test_close_parks_at_first_reset_pose_across_episodes`
- tests/test_embodiment.py:425 `test_close_parks_at_pre_home_pose_when_home_pose_configured`
- tests/test_embodiment.py:439 `test_failed_driver_close_still_clears_connection_state`
- tests/test_embodiment.py:469 `test_reconnect_after_close_recaptures_init_pose`
- tests/test_embodiment.py:493 `test_close_connected_without_park_target_only_releases`
  (survives as-is under the reset gate; re-point its intent at the gate)
- tests/test_eval_end_to_end.py:54 `test_timer_runout_parks_at_init_pose_on_close`

## Constraints

- Repo gates: ruff, mypy, pytest at this repo's configured coverage; keep
  docstring conventions. README style rules (no em dashes in prose).
- Ships in yam 0.8.1 together with #36; the release should follow promptly
  because 0.8.0's no-park behavior is a hardware-safety gap.
