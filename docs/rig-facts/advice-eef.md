# Working under Cartesian control (`move_to`)

How to get things done on this rig when moves are absolute grasp-point
targets. Read alongside the rig facts and the workspace box.

## The readback is the truth

- `eef_state` is the arm's forward kinematics at the grasp point. Trust it
  over any estimate from an image. There is no height bias to correct.
- After every move compare the readback with the target. A shortfall of
  more than ~1.5 cm means one of three things: the target was clipped to
  the workspace box (the tool description lists the bounds), the arm is
  at its reach or folded limit (the arm simply stops), or you are in
  contact. Do not re-send the same target hoping it converges; change it.
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
4. Close fully. Read the settled gripper value: below 0.04 is empty.
5. Lift 5 cm and confirm in the overhead view that the object moved with
   the jaws. Only then transport.
6. On an empty close: open, adjust radially by 2 cm (usually outward), and
   retry once. If the second close is also empty, do not repeat it:
   re-measure from the overhead frame, or use the other arm.

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
