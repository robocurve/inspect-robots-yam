# Rig facts (rig-1)

Measured and derived facts about this specific rig. Kinematic constants
come from forward kinematics on the i2rt YAM model that drives the robot
(i2rt 1.1.2, `robot_models/arm/yam/yam.urdf`); workspace dimensions were
measured at the rig. Planar FK/IK for the arm lives in
[formulas.md](formulas.md); working practice in [advice.md](advice.md).

rig-1 is built identically to rig-2, so design constants (URDF
kinematics, mounting geometry, camera placement, frame conventions) are
shared and stated plainly. Measured behavior was confirmed on this rig
only where the text says so; numbers tagged *(rig-2 value — unverified
here)* carry over from rig-2's verification — treat them as priors and
verify each one before the first action that depends on it (see
[advice.md](advice.md)).

## Kinematics (per arm)

- Upper arm (shoulder pitch to elbow): 0.264 m. Forearm (elbow to wrist
  pitch): 0.245 m.
- Shoulder pitch axis sits 0.114 m above the arm's mounting plane.
- Wrist pitch axis to fingertip midpoint: 0.101 m (gripper flange to
  fingertip: 0.057 m, stock linear gripper). Contact happens at the
  fingertips, and this offset rotates with wrist pitch.
- Gripper stroke is ~9.5 cm jaw-to-jaw fully open, and gripper commands/
  readings are the normalized fraction of that: residual × 9.5 cm ≈
  grasped thickness (confirmed here: a ~4.1 cm block read a 0.435
  settled residual), and anything wider than ~9 cm cannot be grasped
  across. The reading drifts when empty: a closed empty gripper reads
  ≈0.01 immediately and creeps up to ≈0.04 within seconds *(drift
  ceiling is a rig-2 value — unverified here)*, so treat any residual
  under ~0.04 as empty, not as a few-mm object. The command is
  accurately linear in jaw opening (commanding 0.5 gives half the gap)
  *(rig-2 value — unverified here)*.
- Gripper yaw follows j0: the direction the jaws close along rotates
  with the base joint, and is compensable with wrist roll.
- Episodes start with both arms folded near all-zero joints and grippers
  fully open (reading ≈ 1.0).

## Cameras

- `top_cam` is a fixed camera mounted above and in front of the rig
  (observations name the cameras `top_cam`/`left_cam`/`right_cam`). Its
  view is oblique (~45°), not a true overhead — confirmed on this rig:
  you see the sides of objects, and image-y is compressed relative to
  image-x. `left_cam` and `right_cam` are wrist cameras mounted on the
  rolling gripper body of the same-named arm, tilted forward of the
  finger axis; the mounting is identical on both arms, so the
  wrist-image facts below apply to either.
- Top-camera frame orientation: arm-forward = image-up, arm-right =
  image-right. Both arm bases sit just below the bottom edge of the
  frame; a folded gripper shows as a dark blob near the bottom-left/
  bottom-right corner — but because of the tilt, an arm extended to
  r ≈ 0.45 m appears near the TOP corners of the frame.
- Because the view is oblique the top camera has no single scale, and it
  has not been grid-measured on this rig *(rig-2 values — unverified
  here)*: rig-2 reads ≈3 px/cm in the upper (far) third of the frame
  where the arms work, rising to ≈8 px/cm near the bottom edge — a
  ~2.5× change across the frame — and an identically built sibling
  measured ≈2.2–2.7 at table level in the working region. Image-x and
  image-y scales also differ at the same spot, so calibrate the two
  axes separately, and localize the fingertip rather than the arm
  body — elevated links project displaced outward from the tip. Derive
  this rig's numbers from your first sizeable move and use the figure
  for the region you are working in. Scale also grows with height,
  roughly H/(H − h) with camera height H ≈ 0.72 m in the working
  region (H measured on rig-2; the mount is shared), so a gripper
  carrying at 10 cm reads ~15% larger than it will at table level, and
  things shrink slightly as you lower them.
- In each wrist image the grasp point (where the jaws meet) projects
  near the bottom of the 224-px frame — (x≈115, y≈200) with the gripper
  down *(rig-2 value — unverified here)* — and it is NOT
  pitch-independent: pitched forward it shifts up and left, and on this
  rig one episode measured the converged-jaw point at (x≈95–110, y≈168)
  at pitched-forward grasp poses (exact pitch unrecorded). Re-locate it
  with a brief empty close at your current pitch before any grasp that
  needs pixel precision — but use that close only for the grasp point's
  position, never for scale: the jaws sit ~5 cm from the lens versus
  ~20 cm to the table, so the jaw gap implies a px/cm figure 2–3×
  larger than the table-plane scale. The camera looks beyond the
  fingertips, so anything at or nearer than the grasp point is out of
  frame, and the dark wedge at the bottom-centre of the frame is the
  jaws themselves, not the object. A held object is never visible in
  its own wrist camera.
- At wrist roll = 0: image-right = the arm's right (−y in arm
  coordinates, reached with negative j0), image-down = radially inward
  toward the arm's base. Rolling the wrist rotates the scene in the
  image, but the jaws always close along image-x.
- Wrist-cam scale varies strongly with height and pose: measured values
  within ~10 cm of the table range from ~3 px/cm (rig-2 runs) to ~16
  (measured here: a 2.4 cm radial move shifted the scene 40 px), so
  never trust a remembered scale — close
  to the table the 224-px frame may span only a few tens of
  centimetres, and a hand-sized object can fill it. Scale also varies
  within a single frame: the camera is close and forward-tilted, so
  px/cm grows toward the bottom edge — calibrate near where the target
  sits in the image. To calibrate this view, make a pure radial move
  (j1/j2/j3, no yaw) of known FK size and read the pixel shift. A j0
  nudge is tempting (it translates the camera by r·Δj0) but it also
  rolls the wrist image about its centre by Δj0, mixing rotation into
  the shift — on this rig a 0.1 rad yaw probe implied ≈5.6 px/cm and
  the corrections based on it missed repeatedly.
- The wrist image's vertical axis conflates radial distance and height:
  a 3 cm radial move and a 1.4 cm descent shift features by a similar
  number of pixels *(rig-2 measurement — unverified here)*, and radial
  (depth) distance cannot be eyeballed from the top-down wrist view.

## Workspace geometry

- Arm bases are 0.584 m (23 in) apart, mounted parallel (facing the
  same way) — shared build with rig-2.
- Table plane: the tabletop coincides with the arm mounting plane —
  true fingertip z = 0 at table contact *(verified on rig-2 by
  replaying contact poses through the framework's grasp-site forward
  kinematics; unverified here)*.
- Planar-FK z-bias at table contact: not yet measured on this rig
  *(rig-2 measured ≈+0.08–0.09 m on both arms)*. Do one closed-gripper
  table touch before the first grasp and correct absolute heights by
  the result.
- With the gripper pointing straight down, reach caps at ~0.51 m wrist
  radius; pitching the gripper 30–45° forward buys another 10–20 cm.

## Controller behavior

- Position moves undershoot: a `move_joints` call tracks only partway
  to the target and stalls near contact — confirmed here (a commanded
  descent stalled ~4 cm short and needed a stronger re-issue); rig-2
  measured the gravity-loaded j1/j3 sag at 0.05–0.1 rad. Re-issue the
  same target, or
  command slightly beyond it, to converge. Undershoot grows when
  carrying a load — expect lifts with an object in the gripper to need
  more re-issues or stronger targets.
- Larger commands are clamped by a per-step delta limit — reported as a
  separate `approver: N step(s) modified (delta_clamped)` line, not in
  the move result itself — and each call reports only a few steps.
  Nothing is broken — re-issue, or command 0.1–0.2 rad beyond the goal.
- Very small commands are dropped entirely: the minimum useful command
  step is ≈0.05–0.1 rad *(rig-2 value — unverified here)* — for a small
  correction, command at least that far (or go beyond the goal and come
  back) rather than nudging below it.
- Position error that stops shrinking while descending usually means
  the fingers or object reached the table. When joint effort is
  available (below), prefer it over this heuristic.

## Joint effort (when `report_joint_eff` is enabled)

The effort signatures and numeric ranges in this section are rig-2
measurements — unverified here; the same drivetrain makes them strong
priors.

- Contact vs. sag on descents: gravity assists a downward j1 move, so
  j1 falling short during a descent means contact, not controller sag.
  The effort reading distinguishes how hard you are contacting: on a
  light touch j1 simply stops tracking while its effort stays negative
  (still holding the arm against gravity); if effort flips positive,
  the arm is actively pressing into the surface — back off.
- After a full close, the gripper residual settles at the grasped
  object's thickness in gripper units (confirmed here — see
  Kinematics), and gripper effort reads high (≈1.0–2.0) around an
  object. But effort right after closing does NOT distinguish grasp
  from no-grasp: an empty gripper also spikes on the close (≈0.85–0.9,
  decaying to ≈0.05–0.15 over several seconds). Judge grasps by the
  settled residual (empty reads up to
  ≈0.04 — see Kinematics), not by the immediate effort. A partial close
  gives an ambiguous readback for thin objects.
- A grasped object touching down reads as stopped descent plus rising
  effort on the wrist joints.
