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
2. No opt-out. `_home_pose()` never returns `None`; the no-ramp else-branch in
   `reset()` (now unreachable except as the joints-mode no-home path being
   removed) goes away. This matches EEF mode, where the default is mandatory.
3. Park == home: `DEFAULT_REST_POSE` becomes the same tuple (gripper slots
   change from 0.0 to 1.0). The next episode starts in distribution with no
   gripper re-open transient. (Not "motionless back-to-back starts": `close()`
   releases torque after parking so the arms sag slightly, and `reset()`
   always runs the full `rest_secs` ramp even when already near the target.)
4. Safety gate before first motion: today the out-of-the-box joints-mode
   config issues no motion until the operator answers the ready prompt; under
   this plan `reset()` would move both arms immediately, before
   `wait_ready()`, on an uncollision-checked linear joint sweep from wherever
   the operator left them. So attended mode gains a pre-homing confirmation on
   the first reset of a connection (exactly when the arm state is unknown and
   hands may be on the hardware):

   - In `reset()`, record `first_connect = self._init_pose is None` before the
     init-pose capture; when `first_connect and not self._cfg.unattended`,
     call `self._operator.wait_ready("Arms will move to the home pose - stand "
     "clear, then press Enter...")` before `_ramp_to(home_pose)`.
   - `OperatorIO.wait_ready` already accepts a custom prompt and already maps
     dead stdin to `EmbodimentFault` with the `unattended=True` escape hatch;
     no operator changes needed.
   - Ordering within `reset()`: the motionless EEF home FK-in-box validation
     (fail-fast on configuration errors) runs FIRST, then the gate, then the
     ramp. The operator must not confirm "stand clear" only for reset to
     error out configuration-side.
   - Later resets do not re-prompt. So the homing motion is not silent, add a
     `self._status("homing: ramping arms to start pose")` before `_ramp_to`
     on EVERY attended reset (including the first, right after the gate; no
     `first_connect` condition on the status), cleared with
     `self._status(None)` in a `try/finally` around the ramp exactly like
     close()'s "parking:" status, so a mid-ramp fault's traceback never
     prints appended to the status line. Unattended mode is unchanged:
     motion without prompts is what unattended opts into.
   - Scope honesty: the gate's "no motion before confirmation" covers
     `reset()` only. If the gate itself raises `EmbodimentFault` (dead
     stdin), `_init_pose` is already captured, so a subsequent `close()`
     still parks per plan 0004's deliberate always-park-after-capture rule;
     with rest == home that sweep is identical to the one the gate refused.
     Release notes must not overclaim beyond reset.
   - This gate applies in both control interfaces, closing the same
     pre-existing ordering gap for EEF mode and explicit `home_pose` configs.

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
  # Park target == home target: the next episode starts in distribution with
  # no gripper re-open transient. (Torque release after parking still lets
  # the arms sag slightly, and reset always re-runs the homing ramp.)
  DEFAULT_REST_POSE: tuple[float, ...] = DEFAULT_JOINT_HOME_POSE
  ```

- `home_pose` field comment: "Optional reset target" becomes "Reset target;
  `None` selects the per-mode factory default (`DEFAULT_JOINT_HOME_POSE` /
  `DEFAULT_EEF_HOME_POSE`)."

- Do NOT export `DEFAULT_JOINT_HOME_POSE` from `__init__.py`:
  `DEFAULT_EEF_HOME_POSE` is not exported either, and plan 0004 deliberately
  keeps pose constants out of the public surface (the README calls
  `DEFAULT_REST_POSE` an informational constant, not a stable import).

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

- `reset()` adds the attended first-connect pre-homing gate described in
  Decision item 4, placed after driver/kinematics construction, the
  init-pose capture, AND the motionless EEF home validation, immediately
  before the homing ramp; plus the attended "homing:" status around the ramp
  (Decision item 4).

### Tests

- `tests/test_embodiment.py`:
  - `test_reset_without_home_pose_issues_no_command` (line 111) inverts into
    `test_reset_without_home_pose_ramps_to_factory_joint_home`: with
    `home_pose=None` in joints mode, the last command of the reset ramp
    equals `DEFAULT_JOINT_HOME_POSE` (gripper slots de-normalized per
    `gripper_open`/`gripper_closed` at the driver boundary, matching how the
    existing homing tests assert).
  - Test helper change FIRST: the scripted `_operator()` helper (line 50)
    pops from a finite list and raises `IndexError` on exhaustion, and the
    gate adds one consumed input to every attended first reset. Nearly every
    attended test in the file would die with `IndexError` inside `reset()`
    (e.g. `test_reset_twice_reuses_driver` line 251, the `_build_with_status`
    tests line 648, `test_close_after_mid_reset_fault_parks` line 515 which
    dies before its expected camera fault). A bare return-`""`-on-exhaustion
    fix is NOT enough: it fixes the crash class but shifts answer-bearing
    scripts by one prompt, e.g. `test_step_terminates_success_on_operator_yes`
    (line 273, `["", "y"]`) has its "y" eaten by the ready prompt and fails,
    while `test_step_terminates_failure_on_operator_no` (line 282,
    `["", "n"]`) keeps passing vacuously with the "n" never reaching
    `confirm_success`. Make the fake prompt-aware instead: ready/stand-clear
    prompts (any prompt that is not the success prompt) consume nothing and
    return `""`; only the "Did the robot succeed" prompt pops from the
    scripted answers. Scripts then read as verdict sequences and are immune
    to prompt-count changes. Verify the "n" test actually exercises
    `confirm_success` afterward (no vacuous pass).
  - New: attended first-connect gate ordering test. With a scripted
    `OperatorIO` whose `input_fn` records prompts, the first `reset()`
    issues the stand-clear prompt (matched by prompt text) before the driver
    sees any command; a second `reset()` on the same connection does not
    issue the stand-clear prompt before the ramp (it still issues the
    post-home ready prompt, so assert on prompt text or
    command-count-at-call-time, not on zero `input_fn` calls). Unattended
    mode never prompts.
  - Known breakage from "reset now always ramps under default config" (the
    dominant class; each needs restructuring or re-expectation, not just
    gripper-slot edits):
    - `test_gripper_default_calibration_is_identity_both_directions`
      (line 171): homing drives grippers to wire 1.0 before the assertion on
      the echoed 0.35.
    - `test_pacing_skipped_when_hz_zero` (line 304): `assert sleeps == []`
      fails because `_ramp_to` at the 10 Hz fallback sleeps ~30 times during
      reset.
    - `test_close_ramps_to_rest_pose_then_releases` (line 384): command count
      doubles (reset ramp + park ramp) and `commands[0]` is now a homing
      waypoint.
    - `test_close_rest_fault_still_releases_driver` (line 626): the
      always-faulting driver now faults during `reset()` itself, before the
      `close()` under test; needs a driver that faults only after reset.
    - `test_close_rest_pose_zero_hz_falls_back_to_10hz` (line 640): command
      count doubles.
  - Known breakage from the new "homing:" status line (an attended reset now
    emits three status entries - "homing: ...", `None`, "Running: ..." -
    where today it emits one; `_build_with_status` at line 648 records all of
    them). Re-anchor assertions on prompt text or on the last pre-step
    status entry, not positional `status[0]`/`status[1:]`:
    - `test_reset_announces_run_instructions` (line 665): `len(status) == 1`
      becomes 3.
    - `test_status_line_updates_once_per_second_with_horizon` (line 677):
      the `status[1:]` filter now includes the "Running:" line; count and
      indexed asserts shift.
    - `test_status_line_without_hint_shows_elapsed_only` (line 691):
      `updates[0]` is now the "Running:" line.
    - `test_bind_task_drives_the_countdown_horizon` (line 726),
      `test_bound_horizon_wins_over_deprecated_hint` (line 737),
      `test_rebind_latest_envelope_wins` (line 747): all assert on
      `status[0]`, now the homing line without "Max ...".
    - `test_close_clears_the_bound_horizon` (line 755): `"Max" not in
      status[0]` would pass vacuously against the homing line; re-anchor it
      to the "Running:" entry so it still tests the countdown fallback.
  - Audit remaining tests that assume the old all-zero `DEFAULT_REST_POSE`
    (parking assertions) and update gripper-slot expectations to 1.0.
- `tests/test_config.py`: add
  `test_default_joint_home_pose_is_zero_joints_open_grippers`, mirroring the
  existing `test_default_eef_home_pose_...`; also assert
  `DEFAULT_REST_POSE == DEFAULT_JOINT_HOME_POSE`.
- Coverage must stay at the repo gate with the `None`-branch removal (the
  branch is deleted outright, so no unreachable code remains).

### `README.md`

- EEF section sentence "In EEF mode, `home_pose=None` selects the mandatory
  `DEFAULT_EEF_HOME_POSE` instead of skipping homing" generalizes: both modes
  select a factory default; name `DEFAULT_JOINT_HOME_POSE` (zero joints,
  grippers open, dataset-verified) for joints mode.
- Factory resting pose paragraph (near line 273): "YAM ships with a factory
  resting pose at encoder zero for every joint and 0.0 (the closed end of the
  stroke) for both grippers" is wrong after this change; rewrite for the new
  tuple (encoder-zero joints, grippers open) and re-anchor the `rest_pose`
  config.ini example it introduces.
- "Park pose must rest under gravity" bullet (near lines 320-332): the
  gripper half of this warning inverts. It currently warns that the default
  parks with grippers closed so held objects stay gripped, and recommends
  parking open via per-rig `rest_pose`. Rewrite: the factory default now
  parks with grippers open, so parking releases anything still held (during
  the ramp, wherever the arm happens to be); rigs that must keep objects
  gripped at park override `rest_pose` with gripper slots 0.0. Also extend
  the existing "rigs whose joint limits exclude zero must override
  `rest_pose`" clause to cover `home_pose`, since homing now clamps through
  the same per-joint box on every reset.
- Safety bullet "set `home_pose` so episodes begin from a validated start
  state" updates: episodes now home by default in every mode; `home_pose`
  remains the per-rig override. Keep the stand-clear warning, state that
  reset always moves the arms, and mention the attended first-connect
  stand-clear prompt.
- `YamConfig` field list: `home_pose` entry documents the per-mode defaults;
  `rest_pose` entry now says the factory pose is zero joints with grippers
  open and equals the factory home.
- Follow the repo writing-style rules (no em dashes in prose, alert syntax
  preserved).

### Prior-plan supersession notes

Repo convention (0004's header amendment, 0005's edits into 0001) is a short
note at the superseded site:

- `plans/0004-default-rest-pose.md` (near lines 26-31): note that 0007
  changes the canonical factory rest tuple's gripper slots to open (1.0),
  park == home; the operator-confirmed joint zeros stand.
- `plans/0006-eef-control-interface.md` (near lines 348-349): note that
  "`home_pose=None` in joint mode means skip homing" is superseded by 0007's
  mandatory `DEFAULT_JOINT_HOME_POSE`.

## Behavior change and release

Three behavior changes, all intended:

1. Joints-mode `reset()` with an unset `home_pose` now moves the arms (ramp
   over `rest_secs` to zeros with grippers open) where it previously issued no
   motion.
2. `close()` with the factory `rest_pose` now parks grippers open instead of
   closed. If a policy ends an episode holding an object, parking releases it.
3. Attended runs gain one extra prompt: a stand-clear confirmation before the
   first homing ramp of each connection.

Release as v0.11.0 (minor, pre-1.0) with release notes calling out all three,
plus "stand clear at reset" phrasing consistent with the README warning. Do
not promise motionless back-to-back runs: torque release at park lets the
arms sag, and every reset runs the full ramp.

Out of scope: per-policy home poses (the embodiment default is
policy-neutral: it is simultaneously the physical rest joints and the
dataset-verified start), collision checking, and any core inspect-robots
changes.
