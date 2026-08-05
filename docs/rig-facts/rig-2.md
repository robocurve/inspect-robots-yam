# Rig facts (rig-2)

Measured and derived facts about this specific rig. Kinematic constants
come from forward kinematics on the i2rt YAM model that drives the robot
(i2rt 1.1.2, `robot_models/arm/yam/yam.urdf`); workspace dimensions were
measured at the rig.

## Kinematics (per arm)

- Upper arm (shoulder pitch to elbow): 0.264 m. Forearm (elbow to wrist
  pitch): 0.245 m.
- Shoulder pitch axis sits 0.114 m above the arm's mounting plane.
- Wrist pitch axis to fingertip midpoint: 0.101 m (gripper flange to
  fingertip: 0.057 m, stock linear gripper). Contact happens at the
  fingertips, and this offset rotates with wrist pitch.

## Cameras

- `top` is a fixed overhead camera. `left` and `right` are wrist cameras
  mounted on the rolling gripper body of the same-named arm, tilted
  forward of the finger axis.
- Overhead frame orientation: arm-forward = image-up, arm-right =
  image-right. Both arm bases sit just below the bottom edge of the frame;
  a folded gripper shows as a dark blob near the bottom-left/bottom-right
  corner. The bases are ~180 px apart in the top frame, which at 0.584 m
  spacing gives ≈3.1 px/cm at base depth.
- In each wrist image the grasp point (where the jaws meet) projects at
  (x≈115, y≈200) in the 224-px frame, just off the bottom edge —
  independent of wrist roll. To re-calibrate: briefly close the empty
  gripper and watch where the jaws converge.
- At wrist roll = 0: image-right = the arm's right (−y in arm
  coordinates, reached with negative j0), image-down = radially inward
  toward the arm's base. Rolling the wrist rotates the scene in the image
  but the jaws always close along image-x, so roll until the target's
  long axis looks vertical in the image to align a grasp across it.
- Wrist-cam scale varies strongly with height and pose: measured values
  within ~10 cm of the table ranged from ~3 to ~12 px/cm across runs, so
  never trust a remembered scale — close to the table the 224-px frame
  may span only a few tens of centimetres, and a hand-sized object can
  fill it. Calibrate in a single call with one small pure-j0 nudge: the
  lateral pixel shift corresponds to r·Δj0 metres of travel.
- Radial (depth) distance cannot be eyeballed from the top-down wrist
  view. Do an early calibrated test move: translate a known 3 cm and
  measure the pixel shift; expect first estimates to be off by 5–10 cm
  otherwise.
- The wrist image's vertical axis conflates radial distance and height:
  a 3 cm radial move and a 1.4 cm descent shift features by a similar
  number of pixels. Never read depth from image-y alone — descend in
  1–2 cm steps, keeping the target column-aligned with the jaw gap
  (x≈115), and re-check between steps.
- Overhead metric scale: no rulers or tags in the scene. Use rig geometry
  instead: the arm bases are 0.584 m apart (see Workspace geometry), and
  any known commanded move of the gripper observed in the top frame gives
  px/m at table height.

## Workspace geometry

- Arm bases are 0.584 m (23 in) apart, mounted parallel (facing the same
  way).
- Table plane: the tabletop coincides with the arm mounting plane — true
  fingertip z = 0 at table contact (verified by replaying contact poses
  through the framework's grasp-site forward kinematics). If you estimate
  height with the planar approximation below, the table will appear at
  z ≈ +0.08, not 0 — that offset is the approximation's bias, not the
  table's height.
- Planar FK approximation: in the radial–vertical plane, upper-arm angle
  A = π − j1, forearm B = j2 − j1, fingertip axis C = B + j3, and
  fingertip ≈ (0, 0.114) + 0.264·(cos A, sin A) + 0.245·(cos B, sin B)
  + 0.101·(cos C, sin C); gripper-down is C = −π/2. In the gripper-down
  working envelope (r ≈ 0.3–0.5 m) this reads ~3–4 cm short in r and
  ~9 cm high in z, with the bias nearly constant — so relative moves and
  its local Jacobian are accurate, but calibrate absolute height with one
  table touch. Outside that envelope (arm folded) the approximation
  degrades sharply.
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
- Grasp verification: command the gripper fully closed (0.0) and read the
  residual — it settles at the grasped object's thickness in gripper
  units, and gripper effort reads high (≈1.0–2.0) around an object versus
  low (≈0.1) when empty. Never test a grasp by commanding a partial
  close; that readback is ambiguous for thin objects.
- High effort alone does not confirm a good grasp: the residual must also
  match the expected thickness of the part you meant to grip. A residual
  several times too large with high effort means you clamped something
  bulky or at an angle, and it will slip during transport — regrasp. A
  near-zero residual (≈0.01) on a rigid object is the opposite failure: a
  shallow edge pinch that will slip — regrasp deeper so more material
  sits between the jaws.
- A grasped object touching down reads as stopped descent plus rising
  effort on the wrist joints — use it to time release when placing.

## Manipulation practice

- For thin objects lying on a surface, lower the open gripper until joint
  effort confirms fingertip–table contact, then close — closing at an
  unverified height is the main cause of angled, slip-prone grasps.
- When grasping near an object's end, the object extends well beyond the
  grasp point. Plan the release around where the object's extent will
  land, not where the gripper is, and verify placement in the overhead
  frame after release before declaring the task done.
- A carried object hangs and pivots below the grasp point, and wrist roll
  shifts where its center sits relative to the gripper — adjust roll
  after grasping so the load hangs where you want it before transporting.
- Failed close attempts drag the object. After any failed grasp, re-find
  the object's actual position in the overhead frame before retrying —
  don't assume it is where it was.
- The idle arm's wrist camera is a useful second viewpoint, but keep that
  arm retracted — it can collide with the working arm if brought close.
