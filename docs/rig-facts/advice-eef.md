# Working under Cartesian control (`move_to`)

How to get things done on this rig when moves are absolute grasp-point
targets. Read alongside the rig facts and the workspace box.

## The readback is the truth

- `eef_state` is the arm's forward kinematics at the grasp point. Trust it
  over any estimate from an image. There is no height bias to correct.
- After every move compare the readback with the target. A shortfall of
  more than ~1.5 cm means one of four things: the target was clipped to
  the workspace box (the tool description lists the bounds), the arm is
  at its reach or folded limit (the arm simply stops), you are in
  contact, or the target has no joint solution at that orientation. The
  last is the common one on far reaches: beyond x ≈ 0.39, and usually
  beyond 0.45, a low target with the tool vertical may be unreachable as
  posed even though it sits inside the box. Do not re-send the same
  target hoping it converges; change it: tip the tool forward, raise the
  target, or come at it from nearer.
- The yaw target is relative to the trial's start orientation and rotates
  the tool about the grasp point; it does not move the grasp point
  sideways. Pitch and roll are pinned unless the bounds say otherwise.

## Reaching an object

1. Locate it in the overhead frame at the start. Convert pixels to metres
   with the far-region scale from the facts, remembering the view is
   oblique: a target's image-y position over-states its distance from the
   base by the compression factor, and it sits in whichever arm's frame
   you convert to (the workspace box gives the base spacing for switching
   frames).
2. Fly to a point ~10 cm above it at z ≈ 0.15 with the gripper open, then
   look at the wrist camera. The object should sit on the jaw column
   (x ≈ 110). If it is fully visible with empty frame below it, advance
   2–3 cm radially before descending, because the fingertips are further
   out than the image suggests.
3. Descend straight down to z ≈ half the object's height. For table-level
   objects go to z = 0 to −0.01; the box allows a small press.
4. Close fully. Read the settled gripper value. Above 0.05 something is
   between the jaws. Below that tells you nothing: an empty close and a
   held thin object (a pen lid, a bowl wall) both land in 0.03–0.05. Do
   not open and re-grip on the reading alone.
5. Lift 5 cm and confirm in the overhead view that the object moved with
   the jaws. That is the grasp test. Only then transport.
6. If the lift left the object behind: open, adjust radially by 2 cm
   (usually outward), and retry once. If the second attempt also fails,
   do not repeat it: re-measure from the overhead frame, or use the
   other arm.

## Placing

- Carry at z ≥ 0.15; free-space poses sag, so command 2 cm higher than
  the clearance you need.
- Descend until the readback stops short or wrist effort rises, open,
  lift clear, then verify in the overhead frame after the arm has moved
  away. The working arm's occlusion of the top view is expected; do not
  retract mid-placement to look.

## Two arms and the middle of the table

- Each arm reaches well past the table's centre line (the box's
  cross-over bound), so any object on the table is reachable by at least
  one arm; pick the arm whose fence-side bound is not in the way.
- Keep the idle arm parked at its home target; it drifts if left hovering
  and it blocks the overhead view of the centre.
- Both arms in the same region collide with each other before either hits
  a limit, so move one at a time and keep 10 cm between grasp points.

## Spend calls on the task

- The facts sheet's numbers replace calibration. Do not open with survey
  moves, scale probes or table touches; the first task motion is the
  sanity check.
- Thin flat objects (under ~1.5 cm) cannot be pushed at the z floor: the
  jaws ride over them. Grasp uprights instead of pushing.
- Smooth cylindrical or rounded objects eject sideways when closed on
  off-centre. Align in the wrist view before closing, and close no further
  than the object needs.
- If the object lies outside the workspace box or against a fence, say so
  and give up early; the operator can move it or widen the box for the
  next trial.
