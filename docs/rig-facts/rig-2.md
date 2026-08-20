# Rig facts (rig-2)

Measured and derived facts about this specific rig. Kinematic constants
come from forward kinematics on the i2rt YAM model that drives the robot
(i2rt 1.1.2, `robot_models/arm/yam/yam.urdf`); workspace dimensions were
measured at the rig. Planar FK/IK for the arm lives in
[formulas.md](formulas.md); working practice in [advice.md](advice.md).

## Kinematics (per arm)

- Upper arm (shoulder pitch to elbow): 0.264 m. Forearm (elbow to wrist
  pitch): 0.245 m.
- Shoulder pitch axis sits 0.114 m above the arm's mounting plane.
- Wrist pitch axis to fingertip midpoint: 0.101 m (gripper flange to
  fingertip: 0.057 m, stock linear gripper). Contact happens at the
  fingertips, and this offset rotates with wrist pitch.
- Gripper stroke is ~9.5 cm jaw-to-jaw fully open, and gripper commands/
  readings are the normalized fraction of that: residual × 9.5 cm ≈
  grasped thickness (e.g. a reading of 0.04 ≈ 4 mm — a thin handle), and
  anything wider than ~9 cm cannot be grasped across.
- Gripper yaw follows j0: the direction the jaws close along rotates
  with the base joint, and is compensable with wrist roll.
- Episodes start with both arms folded near all-zero joints and grippers
  fully open (reading ≈ 1.0).

## Cameras

- `top` is a fixed overhead camera. `left` and `right` are wrist cameras
  mounted on the rolling gripper body of the same-named arm, tilted
  forward of the finger axis; the mounting is identical on both arms, so
  the wrist-image facts below apply to either.
- Overhead frame orientation: arm-forward = image-up, arm-right =
  image-right. Both arm bases sit just below the bottom edge of the frame;
  a folded gripper shows as a dark blob near the bottom-left/bottom-right
  corner.
- Top-camera scale at table level differs per rig and mounting — expect
  roughly 2–3 px/cm and never trust a remembered number (≈3.1 px/cm was
  derived at this rig from the 0.584 m base spacing spanning ~180 px; a
  sibling rig built identically measures ≈2.2–2.7). Scale grows with
  height: an object h metres above the table reads larger by H/(H − h)
  with camera height H ≈ 0.72 m, so a gripper carrying at 10 cm reads
  ~15% larger than it will at table level, and things shrink slightly as
  you lower them.
- In each wrist image the grasp point (where the jaws meet) projects at
  (x≈115, y≈200) in the 224-px frame, just off the bottom edge —
  independent of wrist roll, but NOT of pitch: pitched 60–70° forward it
  shifts up to y≈150–190. The camera looks beyond the fingertips, so
  anything at or nearer than the grasp point is out of frame, and the
  dark wedge at the bottom-centre of the frame is the jaws themselves,
  not the object. A held object is never visible in its own wrist
  camera. To re-calibrate at the current pitch: briefly close the empty
  gripper and watch where the jaws converge.
- At wrist roll = 0: image-right = the arm's right (−y in arm
  coordinates, reached with negative j0), image-down = radially inward
  toward the arm's base. Rolling the wrist rotates the scene in the
  image, but the jaws always close along image-x.
- Wrist-cam scale varies strongly with height and pose: measured values
  within ~10 cm of the table ranged from ~3 to ~12 px/cm across runs, so
  never trust a remembered scale — close to the table the 224-px frame
  may span only a few tens of centimetres, and a hand-sized object can
  fill it. A pure j0 nudge shifts the scene laterally by r·Δj0 metres,
  which calibrates this view in a single call.
- The wrist image's vertical axis conflates radial distance and height:
  a 3 cm radial move and a 1.4 cm descent shift features by a similar
  number of pixels, and radial (depth) distance cannot be eyeballed from
  the top-down wrist view.

## Workspace geometry

- Arm bases are 0.584 m (23 in) apart, mounted parallel (facing the same
  way).
- Table plane: the tabletop coincides with the arm mounting plane — true
  fingertip z = 0 at table contact (verified by replaying contact poses
  through the framework's grasp-site forward kinematics).
- With the gripper pointing straight down, reach caps at ~0.51 m wrist
  radius; pitching the gripper 30–45° forward buys another 10–20 cm.

## Controller behavior

- Position moves undershoot: a `move_joints` call tracks only partway to
  the target and stalls near contact (gravity-loaded j1/j3 sag 0.05–0.1
  rad). Re-issue the same target, or command slightly beyond it, to
  converge. Undershoot grows when carrying a load — expect lifts with an
  object in the gripper to need more re-issues or stronger targets.
- Larger commands are clamped by a per-step delta limit
  (`delta_clamped` in the move result) and each call reports only a few
  steps. Nothing is broken — re-issue, or command 0.1–0.2 rad beyond the
  goal.
- Position error that stops shrinking while descending usually means the
  fingers or object reached the table. When joint effort is available
  (below), prefer it over this heuristic.

## Joint effort (when `report_joint_eff` is enabled)

- Contact vs. sag on descents: gravity assists a downward j1 move, so j1
  falling short during a descent means contact, not controller sag. The
  effort reading distinguishes how hard you are contacting: on a light
  touch j1 simply stops tracking while its effort stays negative (still
  holding the arm against gravity); if effort flips positive, the arm is
  actively pressing into the surface — back off.
- After a full close, the gripper residual settles at the grasped
  object's thickness in gripper units, and gripper effort reads high
  (≈1.0–2.0) around an object versus low (≈0.1) when empty. A partial
  close gives an ambiguous readback for thin objects.
- A grasped object touching down reads as stopped descent plus rising
  effort on the wrist joints.
