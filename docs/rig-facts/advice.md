# Rig advice (YAM rigs)

Working practice distilled from episode hindsight on these rigs.
Task-agnostic; shared by all YAM rigs.

## Trust the sheet; calibrate on disagreement

The fact sheet's numbers were measured on the rig you are driving —
start the task using them directly rather than spending calls
re-deriving them: a full upfront calibration pass costs about as much
as a verified pick-and-place.

- Make your first task motion double as a sanity check: after the first
  sizeable move, compare the observed pixel shift and joint readback
  against what the documented scale and FK predict. If they agree
  within ~20–30%, proceed on the sheet's numbers.
- Recalibrate only the quantity that disagrees, using these probes:
  - Pixel displacement far from predicted → re-derive px/cm from a
    known gripper move observed in the top frame, computing the
    travelled distance from joint readback, not the commanded target
    (controller undershoot otherwise skews the scale low). A grasped
    object of known size also works. Calibrate in the image region you
    are working in, and image-x and image-y separately — the oblique
    top view has no single scale.
  - Contact earlier or later than FK plus the documented z-bias
    predicts → re-measure the z bias: close the empty gripper and
    descend until joint effort shows table contact (j1 stops tracking
    while its holding torque relaxes), then record what the planar FK
    reads at that pose. A 2 cm height error is the difference between
    grasping a 4 cm object and closing on air — so if your rig's fact
    sheet carries no measured z-bias, do this once before the first
    grasp; and note the first effort-confirmed contact of a real grasp
    gives you the same measurement for free.
  - Wrist-view estimates repeatedly missing → do a calibrated test
    move: translate a known 3 cm purely radially (j1/j2/j3, no yaw — a
    j0 nudge rolls the wrist image and corrupts the reading) and
    measure the pixel shift in the wrist view.
- Prefer solving the planar IK for a target and issuing j1/j2/j3
  together over nudging single joints — it converges in far fewer
  calls.

## Keep the target in view

- A target vanishing mid-approach is ambiguous: still ahead beyond the
  frame, or already at/behind the grasp point. Never close or keep
  advancing on a target you can't see — retract 3–5 cm to re-acquire it,
  and close only when it shows between the two jaw silhouettes. Blind
  closes at a computed position usually miss and shove the object
  5–9 cm, forcing a re-find.
- Never read depth from image-y alone: descend in 1–2 cm steps, keeping
  the target column-aligned with the jaw gap (x≈115), and re-check
  between steps.

## Grasping

- Roll the wrist until the target's long axis looks vertical in the
  wrist image to align a grasp across it — and since gripper yaw follows
  j0, setting wrist roll ≈ j0 keeps the jaws square to a world-aligned
  target from any bearing.
- For thin objects lying on a surface, lower the open gripper until
  joint effort confirms fingertip–table contact, then close — closing at
  an unverified height is the main cause of angled, slip-prone grasps.
- Verify every grasp with a full close and check the settled residual
  against the expected thickness of the part you meant to grip; never
  test a grasp with a partial close, and never judge by gripper effort
  immediately after the close — an empty gripper spikes to ≈0.9 too and
  only decays over several seconds. A residual several times too large
  with high effort means you clamped something bulky or at an angle,
  and it will slip during transport — regrasp. A residual under ~0.04
  means you are holding nothing, or at best a shallow edge pinch that
  will slip (an empty closed gripper itself drifts up to ≈0.04) —
  regrasp deeper so more material sits between the jaws.
- Failed close attempts drag the object. After any failed grasp, re-find
  the object's actual position in the overhead frame before retrying —
  don't assume it is where it was.

## Transport and placement

- A carried object hangs and pivots below the grasp point, and wrist
  roll shifts where its center sits relative to the gripper — adjust
  roll after grasping so the load hangs where you want it before
  transporting. An object picked at one bearing and placed at another
  lands rotated by (j0 at place − j0 at pick) unless you re-match roll.
- When grasping near an object's end, the object extends well beyond the
  grasp point. Plan the release around where the object's extent will
  land, not where the gripper is, and verify placement in the overhead
  frame after release before declaring the task done.
- Time the release off joint effort: a grasped object touching down
  reads as stopped descent plus rising wrist effort.

## Planning the episode

- Before moving anything, choose where the goal arrangement will be
  built: keep it inside both arms' reach overlap — nearer the bases — so
  no object needs a cross-arm handoff; a handoff costs ~10 calls. Budget
  ~6–10 move calls per verified pick-and-place when planning against a
  call limit.
- The idle arm's wrist camera is a genuinely useful second viewpoint —
  fly it in for close-ups the overhead view can't resolve. The only
  hazard is shared space, and it cuts both ways: the classic collision
  is the working arm swinging into an idle arm left hovering over the
  table after its last use. Before sweeping one arm across the
  workspace, park the other near its own base (j1 ≈ 0.6, j2 ≈ 1.2,
  j3 ≈ −0.6), then bring it back out when you need the viewpoint again.
