# Rig facts: agent-facing docs for YAM rigs

Text that gets appended to the agent's system prompt (via
`YamConfig.docs_extra`) so an LLM policy starts a trial knowing what has
already been measured about the rig instead of re-deriving it.

| file | injected when | contents |
|---|---|---|
| [facts.md](facts.md) | always | kinematic constants, gripper and camera behaviour, workspace, controller behaviour (verified numbers only) |
| [advice-eef.md](advice-eef.md) | `control_interface=eef_pos` (`move_to`) | how to reach, grasp, place and verify with absolute Cartesian targets |
| [advice-joints.md](advice-joints.md) | `control_interface=joints` (`move_joints`) | the same under joint-space control, including the planar FK/IK and its biases |

Together they are ~9–10 KB (≈2.5k tokens), sent once per trial in the
cached system prefix.

## Rules for the injected text

- **One anonymous rig:** No rig numbers, machine names, sibling
  comparisons or provenance in these files: the eval does not depend on
  which rig it runs on, and a name in the prompt is only noise. Which rig
  a number came from lives in git history and in this README.
- **Verified numbers only:** A number that has not been measured is
  either stated as a range with a rule for deriving the local value
  ("derive the scale from your first sizeable move"), or left out.
- **Task-agnostic:** Nothing about particular objects, scenes or layouts;
  that belongs in the task instruction.
- **Short:** The sheet replaces calibration calls; if it takes longer to
  read than to measure, it is too long.
- The workspace box itself is not repeated here: the `move_to` tool
  description lists the configured bounds. Set them per rig in
  `config.ini` (`eef_low`/`eef_high`).

## Feeding the docs to a run

```bash
docs="$(cat facts.md advice-eef.md)"
inspect-robots run --policy agent -E control_interface=eef_pos \
  -E "docs_extra=$docs" ...
```

A per-rig `run` wrapper should select `advice-eef.md` or
`advice-joints.md` from the run's `control_interface` and cat it after
`facts.md`. Policies that don't read embodiment docs (VLAs) ignore
`docs_extra`.

## Measuring a rig

The numbers in `facts.md` were measured on one rig; the rigs are the same
build, so they carry over, but the workspace box in `config.ini` should
be checked on each rig once:

1. **Envelope by hand-posing:** (~10 min, arms in zero-g, nothing
   commanded): fingertips on each fence's inner face, closest-to-base,
   fingertips on the table in grasp posture, and on the table's centre
   line, for each arm. Read the grasp-point FK for each pose and derive
   the box: fence-side y = fence − 5 cm (open-jaw half-width plus IK
   overshoot), cross-over y = 0.40, z floor = −0.02, x = 0.08–0.60.
2. **Perimeter tour:** under the real `eef_pos` controller: drive each arm
   to the box corners at z = 0.15 and press the table at two radii, with
   an operator watching; a corner that stalls or clips is a bound to
   revise.

Both were done on 2026-08-26 (`rig-1/measure-envelope.py`,
`rig-1/tour-bounds.py` in the rig directories). If a rig's fences or base
spacing differ by more than a few centimetres from the ranges in
`facts.md`, widen the range there rather than forking the file.
