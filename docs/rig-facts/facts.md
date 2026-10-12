# Rig facts

Measured facts about this rig. Numbers here are verified; use them
instead of re-deriving them. Verify a number only when the first move that
depends on it disagrees with what you see.

## Arms and gripper

- Two identical 6-DoF arms with parallel-jaw grippers, mounted parallel
  (both face forward), each with its own base frame: +x forward, +y left,
  +z up. Grasp-point readings and targets are in the owning arm's frame.
- Link lengths: upper arm 0.264 m, forearm 0.245 m, wrist-pitch axis to
  fingertip 0.101 m. The shoulder pitch axis is 0.114 m above the table.
  With the tool pointing straight down the wrist reaches ~0.51 m from the
  base.
- The tabletop is the arm mounting plane: fingertip z = 0 at table
  contact. Pressing 1–2 cm "below" the table is safe and gives a firm
  contact for grasps at table level.
- Gripper stroke is 9.5 cm jaw-to-jaw; commands and readings are the
  normalized fraction of that (0 closed, 1 open). After a full close the
  settled reading × 9.5 cm ≈ the grasped thickness. A closed empty gripper
  reads ≈ 0.01 and creeps to ≈ 0.04 over a few seconds, so any settled
  reading below 0.04 is empty. Nothing wider than ~9 cm can be grasped
  across.
- Gripper effort spikes to ≈ 0.7–1.2 on every full close, empty or not,
  and stays there for many seconds. Effort cannot tell an empty close
  from a grasp. Judge grasps by the settled reading, then by lifting
  5 cm and checking the object moved in the overhead view.
- The jaws close along the tool's own left–right axis; that axis turns
  with base yaw and with wrist roll.

## Cameras

Observations carry `top_cam`, `left_cam` and `right_cam`, 224 × 224 px.

- `top_cam` is fixed above and in front of the rig, looking back and down
  at ~45°. It is not a plan view: you see the sides of objects, and image-y
  is compressed relative to image-x. Arm-forward is image-up, arm-right is
  image-right, the two bases sit just below the bottom edge, and an arm
  extended to ~0.45 m appears near the top corners.
- There is no single top-camera scale. At table level expect ≈ 2.2–3.5
  px/cm in the far half of the frame where the arms work, rising to ≈ 8
  px/cm at the bottom edge, and x and y scales differ at the same spot.
  Derive the local scale from your first sizeable move and reuse it.
  Things above the table read larger by H/(H − h) with H ≈ 0.72 m (a
  gripper carrying at 10 cm reads ~15% larger than at table level).
- `left_cam`/`right_cam` ride on the same-named arm's gripper body, tilted
  forward of the finger axis, so they look *past* the fingertips. Anything
  at or nearer than the grasp point is out of frame; the dark wedge at the
  bottom-centre is the jaws; a held object is never visible in its own
  wrist camera; the working arm hides from the top camera whatever it
  hovers over.
- Because of that tilt, the point directly under the fingertips projects
  at or below the bottom edge of the wrist image. An object that appears
  centred on the jaw column with clear space below it is still a few
  centimetres short of the fingertips, toward the base. The jaws converge
  on image column x ≈ 105–115; the row depends on wrist pitch (≈ 150 when
  pitched forward, ≈ 200 with the tool vertical): a brief empty close
  shows the current one.
- Wrist-camera scale varies from ≈ 3 to 16 px/cm within 10 cm of the
  table and changes within a single frame (larger toward the bottom
  edge). To calibrate it, make a pure radial move of known size and read
  the pixel shift. Do not calibrate with a base-yaw nudge: it also rolls
  the image about its centre and has produced 2× scale errors.
- The wrist image's vertical axis mixes radial distance and height (a
  3 cm radial move and a 1.4 cm descent shift features about equally), so
  depth cannot be judged from the wrist view alone.
- At wrist roll 0, image-right is the arm's right and image-down is
  toward the base. Rolling the wrist rotates the scene but the jaws still
  close along image-x.

## Workspace

- Targets are clipped to a per-arm box (listed in the move tool's
  description). The fence-side y bound is tight because a low fence runs
  along each outer table edge; the cross-over side reaches past the
  table's centre line, so any point on the table is reachable by at least
  one arm.
- The bases are ~0.65 m apart: `right_y ≈ left_y + 0.65` for the same
  point, and the centre line is at y ≈ −0.32 for the left arm / +0.33 for
  the right.
- The table is z = 0; the box floor of −0.02 lets a descent press firmly.
- Reach: x ≈ 0.59 at z = 0.15 with the tool pitched forward; table
  contact is possible out to x ≈ 0.49.

## Controller

- Free-space poses sag 1–3 cm below the commanded height, more at long
  reach and with a load. Aim ~2 cm high when carrying and when clearing
  obstacles.
- An arm left uncommanded drifts slowly downward over a minute. Park the
  idle arm at its home target rather than leaving it hovering.
- Near the base the arm cannot fold closer than x ≈ 0.11 and may refuse to
  unfold from there; if a move out of a folded pose stalls, first raise z
  and reduce x, then go.
- Descents stop at contact. With joint effort reported: on a light touch
  the shoulder stops tracking while its effort stays negative (still
  holding the arm up); effort flipping positive means the arm is pressing
  into the surface: back off. A grasped object touching down reads as a
  stopped descent plus rising wrist effort.
- Each motion is played at 10 Hz and the trial has a step budget as well
  as an LLM-call budget; a 20 cm move costs ~4 s of it. Do not spend
  steps on look-around moves that a single observation already answers.
