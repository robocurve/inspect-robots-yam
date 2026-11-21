# Working with the IR+ tools (`go`, `grip`, `sequence`)

How to get things done on this rig with IR+: `go` solves the arm's
inverse kinematics on i2rt's model of the arm and moves all six joints
there together. Read alongside the rig facts.

## The result is the truth

- Every move's result is the grasp-point pose read back from the measured
  joints, in the convention `go` takes. Trust it over any estimate from an
  image. There is no height bias to correct.
- A result short of its target by more than ~1.5 cm means the arm is in
  contact or sagging under load. Do not re-send the same target hoping it
  converges; change it.
- An unreachable pose moves nothing and says how close the best solution
  came. Change the pose (usually the tilt, or the reach), not the timing.

## Orientation

- All angles zero points the tool straight down, jaws as at home. `yaw`
  turns the jaws about the vertical without moving the grasp point; the
  wrist reaches about ±120° of yaw from there.
- `tilt` leans the tip forward, away from the base: 90 points the tool
  horizontally forward. `roll` leans it toward the arm's right. Tilted
  approaches reach objects that are lying down or standing tall without
  folding the arm against the table.

## Reaching an object

1. Locate it in the overhead frame at the start. Convert pixels to metres
   with the far-region scale from the facts, remembering the view is
   oblique, and convert to the frame of the arm you will use.
2. One `sequence`: go ~10 cm above it at z ≈ 0.15 with the gripper open,
   then `look` at that arm's wrist camera. The object should sit on the jaw
   column (x ≈ 110). If it is fully visible with empty frame below it,
   come 2–3 cm further out radially before descending.
3. Next `sequence`: descend straight down to z ≈ half the object's height
   (table-level objects: z = 0 to −0.01), `grip` closed, go up 5 cm, and
   `look` at the overhead camera.
4. A settled gripper reading below 0.04 is empty. Confirm in the overhead
   frame that the object moved with the jaws before transporting.
5. On an empty close: open, adjust radially by 2 cm (usually outward), and
   retry once. If the second close is also empty, re-measure from the
   overhead frame, or use the other arm.

## Placing

- Carry at z ≥ 0.15; free-space poses sag, so command 2 cm higher than
  the clearance you need.
- Descend until the result stops short or wrist effort rises, open, lift
  clear, `park` or move away, then `look` at the overhead camera.

## Two arms

- Each arm reaches past the table's centre line; pick the arm whose fence
  is not in the way.
- `park` the idle arm; it drifts if left hovering and it blocks the
  overhead view of the centre.
- Both arms in the same region collide with each other before either hits
  a limit, so move one at a time and keep 10 cm between grasp points.

## Spend calls on the task

- Put each routine in one `sequence` (approach, descend, grip, lift, look)
  and check with `look` where it matters, not after every step.
- The facts sheet's numbers replace calibration. Do not open with survey
  moves, scale probes or table touches.
- Smooth rounded objects eject sideways when closed on off-centre. Align
  in the wrist view before closing, and close no further than the object
  needs.
- If the object is out of reach or against a fence, say so and give up
  early; the operator can move it for the next trial.
