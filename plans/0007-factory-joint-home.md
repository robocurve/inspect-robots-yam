# 0007: Factory default joints-mode home pose

Issue: robocurve/inspect-robots-yam#51
Status: draft

## Problem

In joints mode, `home_pose=None` (the field default) makes `_home_pose()` return
`None`, and `reset()` then performs no homing ramp: the episode starts wherever
the arms physically sit. After any prior run's `close()`, that is
`DEFAULT_REST_POSE`: all joints zero with both grippers commanded closed
(normalized 0.0).

Closed grippers at episode start are out of distribution for MolmoAct2.
Verified against 2,260 MolmoAct2-BimanualYAM training episodes: joints within
noise of zero and both grippers open (normalized 0.98 to 0.99) at effectively
100% of episode starts.

EEF mode does not have this failure because `_home_pose()` falls back to the
mandatory `DEFAULT_EEF_HOME_POSE`. The joints-mode gap has been patched twice
via per-rig `config.ini` (`home_pose = 0,...,1.0` lines), and lost twice,
because release testing deliberately starts from a first-time-user (fresh
config) state. Anything required for correct out-of-the-box behavior cannot
live only in the per-rig override layer.

## Decision (approved)

Give joints mode the same treatment EEF mode already has:

1. `DEFAULT_JOINT_HOME_POSE`: zero joints, grippers open, per arm
   `(0, 0, 0, 0, 0, 0, 1.0)`, doubled for both arms. `_home_pose()` falls back
   to it when `home_pose is None` in joints mode.
2. No opt-out. `_home_pose()` never returns `None`; the dead no-ramp branch in
   `reset()` is removed. This matches EEF mode, where the default is mandatory.
3. Park == home: `DEFAULT_REST_POSE` becomes the same tuple (gripper slots
   change from 0.0 to 1.0). Back-to-back runs need no corrective motion and
   the next episode starts in distribution with no gripper re-open ramp.

Field semantics that do NOT change:

- `home_pose=<explicit tuple>` still overrides per rig, in both modes.
- `rest_pose=None` still means "park at the pose captured at first reset";
  `rest_pose=<tuple>` still overrides. Only the factory tuple's gripper slots
  change.
- `home_pose` remains a 14-D joint-space vector in both control interfaces.

## Changes

### `src/inspect_robots_yam/config.py`

- Add, above `DEFAULT_REST_POSE`:

  ```python
  # Dataset-verified MolmoAct2-BimanualYAM start pose (2,260 episodes,
  # 2026-07-14 audit): joints within noise of encoder zero, both grippers open
  # in effectively every episode start. Joints match the physically captured
  # rest; the gripper slots are commanded open (1.0) rather than the captured
  # closed reading so episodes begin in the training distribution.
  # Assumes standard upright mounting; exotic mounts override per rig.
  _JOINT_HOME_ARM = (0.0,) * ARM_DOF + (1.0,)
  DEFAULT_JOINT_HOME_POSE: tuple[float, ...] = _JOINT_HOME_ARM * 2
  ```

- Redefine `DEFAULT_REST_POSE` as an alias of the same tuple, replacing its
  current `(0.0,) * TOTAL_DIM` value and its 2026-07-14 capture comment
  (the joints part of that comment moves into the block above):

  ```python
  # Park target == home target so consecutive episodes need no corrective
  # motion: close() leaves the arms exactly where the next reset() starts.
  DEFAULT_REST_POSE: tuple[float, ...] = DEFAULT_JOINT_HOME_POSE
  ```

- `home_pose` field comment: "Optional reset target" becomes "Reset target;
  `None` selects the per-mode factory default (`DEFAULT_JOINT_HOME_POSE` /
  `DEFAULT_EEF_HOME_POSE`)."

- If the package exports `DEFAULT_EEF_HOME_POSE` from `__init__.py`, export
  `DEFAULT_JOINT_HOME_POSE` alongside it (check at implementation time; keep
  the public surface symmetric).

### `src/inspect_robots_yam/embodiment.py`

- `_home_pose()` narrows to `-> Vec` and mirrors the EEF branch:

  ```python
  def _home_pose(self) -> Vec:
      """Select the configured joint home, defaulting per control interface."""
      if self._cfg.control_interface == "eef_pos":
          values = self._cfg.home_pose or DEFAULT_EEF_HOME_POSE
      else:
          values = self._cfg.home_pose or DEFAULT_JOINT_HOME_POSE
      return np.asarray(values, dtype=np.float64)
  ```

  (`YamConfig.__post_init__` rejects tuples that are not exactly 14 entries,
  so only `None` can reach the fallback; the `or` spelling stays consistent
  with the existing EEF branch.)

- `reset()` drops the `home_pose is not None` conditional and its
  else-branch (`final_home_command = <current measured pose>`); homing always
  ramps. The EEF-home validation call keeps its existing
  `control_interface == "eef_pos"` guard.

### Tests

- `tests/test_embodiment.py`:
  - `test_reset_without_home_pose_issues_no_command` inverts into
    `test_reset_without_home_pose_ramps_to_factory_joint_home`: with
    `home_pose=None` in joints mode, the last command of the reset ramp
    equals `DEFAULT_JOINT_HOME_POSE` (gripper slots de-normalized per
    `gripper_open`/`gripper_closed` at the driver boundary, matching how the
    existing homing tests assert).
  - Audit tests that assume the old all-zero `DEFAULT_REST_POSE` (parking
    assertions) and update gripper-slot expectations to 1.0.
- `tests/test_config.py`: add
  `test_default_joint_home_pose_is_zero_joints_open_grippers`, mirroring the
  existing `test_default_eef_home_pose_...`; also assert
  `DEFAULT_REST_POSE == DEFAULT_JOINT_HOME_POSE`.
- Coverage must stay at the repo gate with the `None`-branch removal (the
  branch is deleted, not skipped, so no dead code remains).

### `README.md`

- EEF section sentence "In EEF mode, `home_pose=None` selects the mandatory
  `DEFAULT_EEF_HOME_POSE` instead of skipping homing" generalizes: both modes
  select a factory default; name `DEFAULT_JOINT_HOME_POSE` (zero joints,
  grippers open, dataset-verified) for joints mode.
- Safety bullet "set `home_pose` so episodes begin from a validated start
  state" updates: episodes now home by default in every mode; `home_pose`
  remains the per-rig override. Keep the stand-clear warning, and state that
  reset always moves the arms.
- `YamConfig` field list: `home_pose` entry documents the per-mode defaults;
  `rest_pose` entry now says the factory pose is zero joints with grippers
  open and equals the factory home.
- Follow the repo writing-style rules (no em dashes in prose, alert syntax
  preserved).

## Behavior change and release

Two physical behavior changes, both intended:

1. Joints-mode `reset()` with an unset `home_pose` now moves the arms (ramp
   over `rest_secs` to zeros with grippers open) where it previously issued no
   motion.
2. `close()` with the factory `rest_pose` now parks grippers open instead of
   closed. If a policy ends an episode holding an object, parking releases it.

Release as v0.11.0 (minor, pre-1.0) with release notes calling out both, plus
"stand clear at reset" phrasing consistent with the README warning.

Out of scope: per-policy home poses (the embodiment default is
policy-neutral: it is simultaneously the physical rest joints and the
dataset-verified start), collision checking, and any core inspect-robots
changes.
