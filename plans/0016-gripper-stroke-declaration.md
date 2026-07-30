# 0016 — Declare gripper `max_step`: ~1 s full stroke under agent control

Yam side of inspect-robots plan 0033 (core PR #224, released in core v0.30.0).
The design and its edge cases were settled by that plan's 5-round critique;
this plan records only the yam-specific decisions.

## Problem

Under the LLM agent policy a full gripper open↔close is paced by
fraction-of-range speed limits at ~10 s (and a lone full stroke trips the 10 s
playout cap, forcing the model to split the move). Joints are fine; the 0–1
normalized gripper stroke is the mis-scaled dimension. Core v0.30.0 adds
per-dimension `ActionSemantics.max_step` honored by the approver default and
the agent toolset; this change declares it for the YAM grippers.

## Changes

- `YamConfig.gripper_stroke_s: float = 1.0` — seconds for a full 0→1 stroke
  under agent pacing. Validation in `__post_init__`: finite, > 0.
- Derivation: `min(1.0, 1.0 / (gripper_stroke_s × hz))` with
  `hz = control_hz if control_hz > 0 else 10.0` (the embodiment's existing
  fallback; `control_hz` stays deliberately unvalidated). Non-finite inputs or
  products (e.g. `control_hz = inf`, overflowing product) declare **no** limit
  rather than raising — a config whose `info` constructs today must keep
  constructing. At defaults: 0.1/step → 11 interpolation steps (headroom ceil)
  → ~1 s full stroke.
- `action_semantics()` and `action_box()` gain `gripper_max_step: float | None
  = None`, plumbed from `action_box` through to `action_semantics`. The
  embodiment (`_action_space`) passes the derived value unconditionally;
  `action_semantics` applies it only to **absolute** modes (`joint_pos`
  gripper indices 6 and 13; `eef_abs_pose` indices 4 and 9) and ignores it for
  `joint_delta` (whose delta box already is the per-step limit — core rejects
  displacement-mode declarations). The `/act` policy client passes nothing and
  declares no limits; that asymmetry is safe because core's `compat` compares
  semantics field-by-field and deliberately excludes `max_step`. Soften the
  builder docstring's "one function guarantees a clean check" claim
  accordingly.
- `pyproject.toml`: `inspect-robots>=0.30` (the release carrying `max_step`;
  v0.29.0 predates it).

## Tests

- Config validation (`gripper_stroke_s` non-finite / ≤ 0 raises).
- Derivation arithmetic: defaults → 0.1; the 1-step cap for tiny stroke times;
  non-positive and non-finite `control_hz` fall back / decline to declare;
  overflow product declines to declare.
- Declaration present on both absolute interfaces at exactly the gripper
  indices, `None` elsewhere; absent for `joint_delta`; absent when the builder
  gets no `gripper_max_step` (the `/act` client path).
- Existing semantics/box tests unchanged.
