# Ship a factory default rest_pose

Date: 2026-07-14
Status: draft; subagent critique round 1 pending

## Problem

All YAM pairs share geometry, so the gravity-safe resting pose is
rig-invariant, yet every fresh install starts with `rest_pose = None`. On the
released 0.8.0 that means close() releases torque wherever the episode ended
(arms fall from task poses); since #36 it means parking at whatever pose the
arms happened to be in at first reset. Neither gives a fresh user the
canonical resting pose, and the setup wizard never prompts for one (#43).

## Captured reference pose

Read from a physical YAM pair posed at rest on 2026-07-14 (zero-gravity
connect, calibration sweep skipped, grippers recorded open):

```
0.0372,0.0044,0.0006,-0.0898,-0.0517,-0.1539,0.0,0.0082,0.0010,0.0006,-0.0818,0.0727,0.1410,0.0
```

Layout per `packing.py`: left arm indices 0-6, right arm 7-13, gripper last
per arm, grippers normalized (0.0 = open).

## Design

1. `config.py`: add a module-level `DEFAULT_REST_POSE: tuple[float, ...]`
   with the captured values and a comment recording provenance (physical
   capture, date, grippers open) and the assumption it encodes: standard
   upright mounting. Exotic mounts override per rig.
2. `YamConfig.rest_pose` default changes `None` -> `DEFAULT_REST_POSE`. The
   type stays `tuple[float, ...] | None`.
3. Opt-out that preserves #36's captured-init fallback: a blank string
   (`rest_pose =` in config.ini, or `-E rest_pose=`) parses to `None`.
   Today a blank string raises `ValueError` from `float("")` inside the
   `_FromKwargs` float-tuple parsing, so this is a strict extension, applied
   uniformly to every `_FLOAT_TUPLE_FIELDS` member whose field accepts
   `None` (`home_pose`, `rest_pose`); blank stays an error for fields with
   non-None defaults (`joint_low`, `joint_high`, `step_limits`), since
   blank-means-default there would be indistinguishable from a typo.
4. `close()` logic is untouched: explicit tuple -> ramp there; `None` ->
   captured-init fallback (#36). Only the default value and the blank
   parsing change.
5. README: update the `rest_pose` config-table row, the worked example, and
   the "Park pose must rest under gravity" safety bullet to describe the
   factory default, the per-rig override, and the blank opt-out. Keep the
   safety qualifier that the pose the park ends in is the pose the arms go
   limp from.

## Tests

- `DEFAULT_REST_POSE` itself: 14 entries, passes `YamConfig()` validation,
  gripper slots (indices 6 and 13) within [0, 1].
- `YamConfig().rest_pose == DEFAULT_REST_POSE` (new default).
- Blank-string parsing: `rest_pose=""` -> `None`; `home_pose=""` -> `None`;
  `step_limits=""` -> still `ValueError`.
- Explicit tuple/string override wins over the default.
- close() with default config ramps to `DEFAULT_REST_POSE` (fake driver
  records commanded waypoints; final target equals the constant).
- close() with blank opt-out parks at the captured init pose (existing #36
  behavior, now behind the opt-out).
- Sweep existing tests for assumptions that `YamConfig().rest_pose is None`
  or that close() without config skips ramping / parks at init; update them
  to construct the opt-out explicitly where they mean it. The implementer
  must enumerate these in the PR body.

## Constraints

- Repo gates: ruff, mypy, pytest at this repo's configured coverage; keep
  docstring conventions.
- README style rules (no em dashes in prose, alert syntax preserved).
- Ships in yam 0.8.1 together with #36; note in the PR body that the
  release should follow promptly because 0.8.0's no-park behavior is a
  hardware-safety gap.
