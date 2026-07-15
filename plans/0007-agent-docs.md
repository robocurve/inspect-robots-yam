# 0007 — Kinematic operating notes for LLM agent policies

Status: draft
Issue: robocurve/inspect-robots-yam#53
Design authority: core plan 0016 (robocurve/inspect-robots, approved after 4
critique rounds) §3.4 and §4. This plan only fixes the yam-side content and
wiring; do not re-litigate the channel design.

## 1. Scope

- Set `EmbodimentInfo.docs` on the YAM embodiment according to
  `control_interface` ("joints" vs "eef_pos"), using the normative strings in
  §3 below.
- New config key `docs_extra: str = ""` (embodiment arg, reachable via
  `-E docs_extra="…"` and config.ini): operator-supplied rig-specific notes
  (e.g. how the two arm bases are mounted relative to each other and the
  table). When non-empty after stripping outer whitespace, the stripped text
  is appended to the built-in text separated by one blank line, with no other
  transformation (it may contain braces; nothing passes through str.format).
  The built-in text stays rig-agnostic.
- Raise the `inspect-robots` lower bound to `>=0.12` (the release carrying
  `EmbodimentInfo.docs`; constructing `EmbodimentInfo(docs=…)` raises
  TypeError on older cores). Cut a yam minor release after merge.
- Tests per core plan 0016 §4 "YAM plugin" bullets.

## 2. Content ground truth

The prose in §3 derives from the combined arm+gripper MuJoCo model in the
i2rt repo (`yam.xml` + `linear_4310` via `combine_arm_and_gripper_xml`),
sign-verified by FK probes on 2026-07-15; hardware motor
`directions: [1,1,1,1,1,1]` makes model signs match hardware. Label naming
and gripper polarity must additionally match this repo's code
(`packing.py` DIM_LABELS / `config.py` EEF labels; `embodiment.py`
`_denorm_grippers`: command 1 = open). The controller's IK/FK site is
`grasp_site`, verified in the combined model to sit between the fingertips
(0.135 m beyond the wrist-roll flange), so "grasp-point position between the
fingertips" is the accurate description of what eef targets move.

## 3. Normative docs strings

Implementation stores these as module constants and must reproduce them
character for character, where the constant is the fenced block with
per-line trailing whitespace and the final newline stripped (so the
`docs_extra` join in section 4 yields exactly one blank line). They avoid
restating action bounds (the tool description owns bounds) and describe
joint-level motion, not end-effector effect, because the folded rest pose
makes end-effector effects counterintuitive.

### 3.1 joints mode

```text
Two identical 6-DoF arms, prefixed left_ and right_, each with a parallel-jaw
gripper. Each arm has its own base frame: +x points forward out of the base
(the direction the folded gripper points at all-zero joints), +y left, +z up;
how the two bases are mounted relative to each other depends on the rig.
Joint guide (positive direction, identical for both arms):
- left_j0 / right_j0: base yaw about the vertical axis; positive swings the
  arm counterclockwise seen from above (a forward-pointing gripper moves
  toward +y).
- left_j1 / right_j1: shoulder pitch; 0 points the upper arm horizontally
  backward and is the lower hard stop (it cannot go negative), positive
  raises it (about 1.57 is straight up, about 3.14 is horizontal forward).
- left_j2 / right_j2: elbow; 0 is fully folded with the forearm doubled back
  against the upper arm and is the lower hard stop, positive opens it.
- left_j3 / right_j3: wrist pitch, axis parallel to the elbow; positive tilts
  the gripper up.
- left_j4 / right_j4: wrist yaw; positive swings the gripper toward the arm's
  right seen from above (opposite sign sense of j0).
- left_j5 / right_j5: wrist roll about the gripper's pointing axis; positive
  turns clockwise when viewed from behind the gripper looking out along the
  fingers.
- left_gripper / right_gripper: 0 is fully closed, 1 is fully open (about
  9.5 cm between the jaws).
Proportions: upper arm 0.26 m, forearm 0.25 m, wrist to grasp point 0.25 m
when straight; reach from the shoulder about 0.76 m.
At all-zero joints the arm rests folded low with the gripper pointing
forward. While the arm is folded, a single joint's effect on the gripper
position can be counterintuitive; move deliberately and re-check the
observation after each motion.
```

### 3.2 eef_pos mode

```text
Two identical 6-DoF arms, prefixed left_ and right_, each with a parallel-jaw
gripper, controlled by Cartesian end-effector targets. Each arm's targets are
in that arm's own base frame: +x points forward out of the base, +y left, +z
up; how the two bases are mounted relative to each other depends on the rig.
- left_x / right_x, left_y / right_y, left_z / right_z: grasp-point position
  in meters in the arm's base frame (the grasp point sits between the
  fingertips).
- left_yaw / right_yaw: tool rotation in radians about vertical, relative to
  the trial's start orientation (0 keeps the start orientation).
- left_gripper / right_gripper: 0 is fully closed, 1 is fully open (about
  9.5 cm between the jaws).
Proportions: upper arm 0.26 m, forearm 0.25 m, wrist to grasp point 0.25 m
when straight; reach from the shoulder about 0.76 m.
An inverse-kinematics layer converts targets into joint motion; unreachable
or awkward targets may be tracked slowly or held, so prefer modest steps and
re-check the observation after each motion.
```

## 4. Wiring

- The docs string is chosen at `EmbodimentInfo` construction time from
  `control_interface`; `docs_extra` (stripped of outer whitespace, appended
  as `built_in + "\n\n" + extra` when non-empty after strip) completes it.
- `docs_extra` follows the existing scalar-kwargs config pattern
  (`_FromKwargs`, plain string field); document it in the `YamConfig` args
  list in the README's configuration reference (a prose list, not a table).

## 5. Tests (from core plan 0016 §4, concretized)

- joints mode: every one of the 14 DIM_LABELS appears exactly once across
  the lines starting with `"- "`; the polarity literal "0 is fully closed,
  1 is fully open" present.
- eef mode: all 10 EEF labels appear in the `"- "` lines; word-boundary
  regexes for tokens `0.48`, `0.15`, `0.03` find nothing (bounds tripwire;
  `0.25`, `0.4`, yaw and gripper endpoints deliberately excluded per core
  plan).
- `docs_extra="rig note {with braces}"` appears verbatim at the end,
  separated by a blank line; default adds nothing (docs equal the built-in
  constant exactly).
- Whitespace-only `docs_extra` treated as empty.
- Both modes: `info.docs` is non-empty and mode-appropriate (joints docs
  mention `left_j0`, eef docs mention `left_x`).
