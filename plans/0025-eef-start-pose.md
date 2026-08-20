# 0025: Named start poses in EEF mode

Issue: #131. Branch: `feat/eef-start-pose`.

## Goal

`YamConfig.start_pose` works with `control_interface="eef_pos"`: a named
joint-space pose is resolved, bounds-checked (joint limits at resolution,
then the EEF action box before the ramp), ramped to in joint space, and the
trial's IK seed and yaw reference derive from the arrival pose — the same
path the default EEF home already takes.

## Why the veto is unnecessary (what already exists — do not rebuild)

- **Homing in EEF mode is already joint-space.** `_ramp_to` reads
  `driver.get_joint_pos()` and sends joint waypoints regardless of control
  interface, and `DEFAULT_EEF_HOME_POSE` is itself a joint-space pose.
- **The reset flow already routes a resolved start pose correctly.**
  `reset()` resolves `start_pose` (with the joint-limit check) before
  `_home_pose()` is consulted, and `_home_pose()` prefers the resolved
  start pose. From there the pose flows through `_validate_eef_home`
  (FK position, gripper aperture, and relative yaw 0 against
  `eef_low`/`eef_high`), the joint-space ramp, IK seeding, and
  yaw-reference capture without modification.
- The `__post_init__` veto in `config.py` was a scoping decision recorded in
  plan 0024 / #128 ("EEF conversion is out of scope"), not a missing
  capability: no conversion is needed because nothing in the homing path
  consumes an EEF-space pose.

## Changes

1. `config.py`: drop the `control_interface == "eef_pos"` veto inside the
   `start_pose` validation in `__post_init__`. The name-rule check
   (`validate_pose_name`) is retained.
2. `embodiment.py`: `close()` clears `_eef_home_validated`. Today the latch
   is sound because the home cannot change within an embodiment instance;
   with named start poses it is wrong across reconnects: `close()` clears
   `_resolved_start_pose` precisely so a reconnect re-reads the (possibly
   edited) pose file — plan 0024's invariant — and the re-read pose must
   re-pass the EEF box check. Revalidation of a static home is idempotent,
   so clearing the latch is harmless in the existing modes.
3. `embodiment.py`: `_validate_eef_home` names the configured start pose in
   its error ("start pose 'ready'") when one is set, and keeps the existing
   "EEF home state … workspace bounds" wording otherwise — three existing
   tests match `left EEF home state.*workspace` and must keep passing.
4. `config.py` `start_pose` comment gains one sentence: in EEF mode the
   resolved pose must start inside the EEF action box — FK grasp-point
   position, gripper aperture, and relative yaw 0 are all checked against
   `eef_low`/`eef_high` before the homing ramp.

## Retained invariants (restated at the point of modification)

- The joint-limit check at pose resolution (`reset()`, packed-index error
  message) is untouched and still runs before the driver factory. The EEF
  box check, by contrast, runs after the driver connects (kinematics need
  the driver-facing config) but before any motion or operator prompt — the
  new failure test must NOT assert the factory was never called. (The
  post-connect placement is pre-existing ordering in `reset()`, nothing
  about the driver is needed for the check itself.)
- `close()` keeps clearing `_resolved_start_pose` (reconnect re-reads the
  file); change 2 extends the same reconnect-freshness rule to the EEF box
  validation latch.
- The yaw reference is captured from the settled arrival pose, exactly as
  for the default home. A start pose with a non-default tool orientation
  therefore fixes that orientation family for the whole trial — same
  behavior the default home already has, with an operator-chosen pose.
- `start_pose` and `home_pose` remain mutually exclusive.

## Tests (CI enforces 100% line+branch coverage)

- `test_config.py`: `test_start_pose_rejects_eef_control` becomes
  `test_start_pose_accepted_with_eef_control` (construction succeeds).
- `test_eef_embodiment.py` (add a local `_save_start_pose` helper mirroring
  `test_embodiment.py`):
  - a named start pose resolves and the homing ramp lands on it in EEF mode
    (asserts the EchoDriver's final command equals the pose).
  - a start pose whose FK grasp point is outside the EEF box raises
    `ValueError` naming the pose (covers the new error branch; does not
    assert factory-never-called).
  - close-then-reset revalidation: run with `eef_high` narrowed so its
    gripper bound is 0.9 (pose gripper 0.8 passes both checks on the first
    reset). After `close()`, rewrite the pose file with gripper 0.95 —
    inside the joint-limit range [0, 1], outside the EEF box — and assert
    the next `reset()` raises the box error naming the pose. The gripper
    route is required because the harness's `FakeRawKinematics.fk` ignores
    joint input (a joint edit cannot move the FK position), and a value
    outside [0, 1] would be pre-empted by the joint-limit resolution check.
    This exercises both the file re-read and the revalidation (change 2).

## Docs

- CHANGELOG "Unreleased / Added": named start poses now work in EEF mode,
  with the box-check semantics and the reconnect-revalidation note (#131).
- README EEF section: one line noting `start_pose` works in EEF mode and the
  pose must start inside the EEF action box.

## Out of scope

- Authoring or storing poses in EEF space (poses stay joint-space JSON).
- Any change to yaw-reference or orientation semantics.
