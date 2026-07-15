# Ship a factory default rest_pose

Date: 2026-07-14
Status: revised after critique rounds 1 (blank opt-out replaced by the
core's existing `none` spelling; provenance clarified; test sweep
enumerated), 2 (park gate predicate corrected to pose-captured, which
preserves the #36 mid-reset-fault park), and 3 (verdict ready; wording and
test-sharpness nits folded in); renumbered 0003 -> 0004 after #42 took 0003.
Polarity correction (plan 0005): the wire gripper convention is now
1 = open, 0 = closed. The captured value 0.0 below is unchanged and still
correct as a *value*, but it is the CLOSED end of the stroke; this plan's
original "0.0 = open" labels and the "parks with grippers open" rationale
used the pre-0005 inverted doc convention and read backwards today. See
plans/0005-gripper-open-polarity.md.

## Problem

All YAM pairs share geometry, so the gravity-safe resting pose is
rig-invariant, yet every fresh install starts with `rest_pose = None`. On the
released 0.8.0 that means close() releases torque wherever the episode ended
(arms fall from task poses); since #36 it means parking at whatever pose the
arms happened to be in at first reset. Neither gives a fresh user the
canonical resting pose, and the setup wizard never prompts for one (#43).

## Reference pose

The canonical rest pose is all zeros: every joint at encoder zero, both
grippers 0.0 = open.

```
0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

Operator-confirmed on 2026-07-14 against two physical captures of a YAM
pair posed at rest (zero-gravity connect, gripper range sweep skipped;
joint readings from the motors' own encoders): every reading was within
0.09 rad of zero, and the operator confirmed true rest is encoder zero.
Layout per `packing.py`: left arm indices 0-6, right arm 7-13, gripper
last per arm, grippers normalized. Provenance comment in code must record:
date, operator confirmation over physical captures, grippers open, and the
standard-upright-mounting assumption. Exotic mounts override per rig.

## Design

1. `config.py`: module-level `DEFAULT_REST_POSE: tuple[float, ...]` holding
   the canonical zero pose with the provenance comment above. Not exported in
   `__all__` (tests/test_api_snapshot.py unchanged); it is reachable as
   `inspect_robots_yam.config.DEFAULT_REST_POSE` for the README to name.
2. `YamConfig.rest_pose` default changes `None` -> `DEFAULT_REST_POSE`; the
   type stays `tuple[float, ...] | None`.
3. Opt-out is the core's existing spelling, no plugin parsing changes:
   `rest_pose = none` in config.ini or `-E rest_pose=none` already reaches
   the plugin as Python `None` (core `_defaults.parse_value`). `None` means
   #36's captured-init fallback. A blank value stays a loud `ValueError`
   (it catches half-written lines; do not make it meaningful).
4. Park gating — close() parks only if a pose was ever captured
   (`_init_pose is not None`). Capture happens at the top of the first
   reset, strictly before any embodiment-commanded motion, so this
   predicate is exactly the "did we possibly move the arms" boundary and
   needs no new state (close() already clears `_init_pose`). Path by path:
   never connected -> no-op; connect-then-fault-before-capture -> release
   in place (we never moved the arms); any fault after capture, including
   a camera-open failure that fires after the home ramp -> park at the
   explicit-or-default tuple (or the captured pose when `None`), matching
   #36's operator expectations; normal end -> park; double-close -> no-op;
   reconnect -> re-capture, then park. Deliberate behavior change to call
   out in the PR body: an explicitly-set rest_pose no longer ramps when
   close() runs before any capture. NOT "completed first reset": that
   predicate would leave the arms limp at the raised home pose on a
   mid-reset camera fault, regressing #36 and contradicting 0002's
   amendment ("release-in-place only happens when no pose was ever
   captured").
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
6. README: update the `rest_pose` entry in the `YamConfig` field list
   (README.md:288-294; it is a prose list, not a table), the worked example
   (currently shows grippers 1.0), and the "Park pose must rest under
   gravity" safety bullet: factory default, per-rig override, `none`
   opt-out, path-not-collision-checked warning, and one sentence noting the
   default parks with grippers open, so anything still held is released at
   park. Name `DEFAULT_REST_POSE` as an informational constant, not a
   stable import (it stays out of `__all__` on purpose; do not "fix" the
   snapshot test by exporting it). Add one clause: override rest_pose on
   rigs whose joint limits exclude zero, since the park target is clamped
   through the same per-joint box as every command. Keep every existing
   safety qualifier.
7. Stale prose sweep: the `rest_pose` field comment (config.py:98-101), the
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
- close() with default config after a reset ramps to `DEFAULT_REST_POSE`,
  with the fake driver STARTED AT A NON-ZERO POSE so park-at-default is
  distinguishable from park-at-captured-init (both are zeros otherwise).
- close() before any pose capture releases in place for BOTH default and
  explicit rest_pose (the new gate); close() after a capture but with a
  mid-reset fault still parks (the #36 invariant the gate must preserve).
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
  (survives as-is under the capture gate; re-point its intent at the gate)
- tests/test_eval_end_to_end.py:54 `test_timer_runout_parks_at_init_pose_on_close`

## Constraints

- Repo gates: ruff, mypy, pytest at this repo's configured coverage; keep
  docstring conventions. README style rules (no em dashes in prose).
- Ships in yam 0.8.1 together with #36; the release should follow promptly
  because 0.8.0's no-park behavior is a hardware-safety gap.
