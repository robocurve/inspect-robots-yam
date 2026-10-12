# Working under joint control (`move_joints`)

How to get things done on this rig when moves are absolute joint targets.
Read alongside the rig facts.

## Planar kinematics

The arm's shoulder, elbow and wrist pitch joints move the fingertips in a
vertical plane through the base yaw. In that plane, with radius r from the
base axis and height z above the table:

- Constants: upper arm L1 = 0.264 m, forearm L2 = 0.245 m, wrist to
  fingertip L3 = 0.101 m, shoulder axis L0 = 0.114 m above the table.
- Angles: A = π − j1 (upper arm), B = j2 − j1 (forearm), C = B + j3
  (fingertip axis; gripper pointing down is C = −π/2).
- Forward: fingertip ≈ (0, L0) + L1·(cos A, sin A) + L2·(cos B, sin B)
  + L3·(cos C, sin C).
- Inverse, for a target (r, z) and approach angle C: let
  P = (r, z) − L3·(cos C, sin C) − (0, L0) and d = |P|; then
  A = atan2(P_z, P_r) + acos((d² + L1² − L2²) / (2·L1·d)),
  B = atan2(P_z − L1·sin A, P_r − L1·cos A), and j1 = π − A,
  j2 = B + j1, j3 = C − B. This is the elbow-up branch the arms use.
  Reachable when d ≤ L1 + L2 = 0.509.
- Bias: in the gripper-down working envelope (r ≈ 0.3–0.5 m) the
  approximation reads about 3 cm short in r and 9 cm high in z, on both
  arms. The bias is nearly constant for a given arm, so relative moves
  and the local Jacobian are accurate. Correct absolute heights by
  −0.09 m before the first grasp; a first effort-confirmed table contact
  refines it for free. Outside that envelope (arm folded) the
  approximation degrades sharply.
- Base yaw j0 sweeps the plane: a target at bearing θ from the base is
  reached with j0 = θ, and the wrist-camera image rotates with it.

## Commands and undershoot

- Solve the planar IK for a target and issue j1/j2/j3 together; nudging
  single joints converges in far more calls.
- Moves undershoot: a command tracks only partway and stalls near
  contact (gravity-loaded j1 and j3 sag 0.05–0.1 rad, more with a load).
  Re-issue the same target, or command 0.1–0.2 rad beyond it.
- Large commands are rate-limited per step and each call plays only a
  few steps; a separate `approver: N step(s) modified (delta_clamped)`
  line reports it. Nothing is broken: re-issue.
- Commands under ~0.05 rad are dropped entirely. For a small correction
  command at least 0.05–0.1 rad, or go past the goal and come back.
- A descent whose position error stops shrinking has reached the table
  or the object. Prefer joint effort when it is reported: the shoulder
  stops tracking with negative effort on a light touch, and effort
  flipping positive means pressing.

## Reaching an object

1. Locate it in the overhead frame; convert with the far-region scale,
   remembering the view is oblique. Compute its bearing and radius from
   the arm's base.
2. Set j0 to the bearing, solve the IK for a point 10 cm above the target
   with the gripper down, and go there.
3. Look at the wrist camera. Keep the target on the jaw column
   (x ≈ 110) and re-locate the jaw row at your current pitch with a
   brief empty close if pixel precision matters. If the target is fully
   visible with empty frame below it, extend 2–3 cm radially before
   descending.
4. Descend in 2–3 cm steps until effort confirms contact (or until the
   FK, corrected by the z bias, says the fingertips are at the object's
   mid-height), then close fully.
5. Settled gripper reading below 0.04 is empty. Lift 5 cm and confirm in
   the overhead frame that the object moved before transporting.
6. On an empty close: open, adjust radially by 2 cm (usually outward),
   and retry once. A second miss means re-measure from the overhead
   frame rather than repeating. Failed closes shove the object several
   centimetres, so re-find it before the next attempt.

## Grasp geometry

- Roll the wrist until the target's long axis looks vertical in the wrist
  image to grasp across it. Gripper yaw follows j0, so wrist roll ≈ j0
  keeps the jaws square to a world-aligned target from any bearing.
- A carried object hangs below the grasp point and rotates with j0
  between pick and place; re-match roll at the place bearing or the
  object lands turned by the j0 difference.
- When grasping near an object's end, plan the release around where the
  object's far end will land, not where the gripper is.

## Placing

- Carry with 10 cm of clearance; expect to re-issue lifts under load.
- Descend until contact (stopped descent plus rising wrist effort),
  open, lift clear, then verify in the overhead frame after the arm has
  moved away. The working arm's occlusion of the top view is expected.

## Two arms

- Each arm reaches past the table's centre line; choose the arm whose
  fence is not in the way, and avoid cross-arm handoffs (about 10 calls).
- Park the idle arm near its base (j1 ≈ 0.6, j2 ≈ 1.2, j3 ≈ −0.6) before
  sweeping the other across the table; an idle arm left hovering is the
  usual collision. Its wrist camera is a useful second viewpoint when
  brought back out.

## Spend calls on the task

- The facts sheet's numbers replace calibration. Do not open with survey
  moves or scale probes; the first task motion is the sanity check, and
  only the quantity that disagrees gets re-measured (scale from a
  known move read off joint readback, not the commanded target; z bias
  from an effort-confirmed table touch; wrist scale from a pure radial
  move, never a j0 nudge).
- Budget 6–10 move calls per verified pick-and-place.
- Smooth rounded objects eject sideways when closed on off-centre; align
  in the wrist view first and close no further than the object needs.
- If the object is out of reach or against a fence, say so and give up
  early; the operator can move it for the next trial.
