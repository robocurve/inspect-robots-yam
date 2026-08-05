# 0021 — Per-joint effort in observations

## Motivation

Agent-policy episodes on real YAM rigs use "commanded pose stopped tracking"
as their only contact signal. Two independent episode reports (rig-2,
2026-08-04) hit the same wall: gravity-loaded j1/j3 sag 0.05–0.1 rad under
position control, so the agent cannot distinguish "blocked by the table"
from "controller lag", and asked outright for "a reported joint
torque/effort or contact flag" to make grasp-height detection reliable.

The signal already exists and is free: every DM motor CAN feedback frame
carries estimated torque; i2rt parses it (`MotorInfo.eff`, sign-corrected
N·m) and caches it in the same `_joint_state` snapshot that
`get_joint_pos()` reads. Nothing new touches the bus — the plugin just
never surfaces it.

With effort visible, the two failure modes separate cleanly: position error
with flat effort = controller sag (re-issue the target); position error
with rising effort = contact (stop descending).

## Design

One opt-in config key; one new driver-protocol method; one new
`Observation.state` entry; one docs paragraph.

### Config: `report_joint_eff: bool = False`

- Field in `YamConfig` near `depth_fps` (`config.py:234` area), frozen
  dataclass default `False`.
- Pre-construction guard in `from_kwargs` mirroring `collision_guardrail`
  (`config.py:249-253`): the CLI coerces a literal `none` to `None`, which
  must be rejected as non-bool, not fall through as falsy.
- `OPTION_SLOTS` entry on the embodiment factory so the setup wizard can
  toggle it (pattern: existing `OptionSlot` declarations at
  `embodiment.py:1107-1137`).
- Opt-in (default off) because it changes the observation contract; VLA
  policies (`molmoact2`) never see it unless a rig turns it on.

### Driver: `get_joint_eff()` on `BimanualDriver`

- Add to the `BimanualDriver` Protocol (`embodiment.py:150`):
  `get_joint_eff() -> np.ndarray` returning the packed 14-dim vector
  (7 per arm: 6 arm joints + gripper), raw N·m.
- `_Real` (`embodiment.py:259`): per arm,
  `obs = arm.get_observations()`; effort7 =
  `np.append(obs["joint_eff"], obs["gripper_eff"])`; pack left/right with
  the existing `packing.pack`. `get_observations()` reads the same cached,
  lock-protected `_joint_state` as `get_joint_pos()` — no extra bus round
  trip, same freshness.
- Gripper slots stay **raw effort** (N·m) — deliberately NOT normalized the
  way gripper position slots are, since normalization is a position-range
  concept. The docs paragraph states this.
- Real-hardware body sits inside the existing `# pragma: no cover` seam
  (`_default_driver_factory`).

### Observation: `state["joint_eff"]`

- In `_observe()` (`embodiment.py:1907`): when `cfg.report_joint_eff`,
  `values["joint_eff"] = packing.validate_dim(driver.get_joint_eff())`
  (float, shape `(14,)`), inserted after `joint_pos` / `eef_state`.
- **Deliberately NOT declared** as a `StateField` in `observation_space()`.
  This is forced, not merely prudent: core conformance
  (`inspect_robots/conformance.py:218-227`) errors when an absolute control
  mode declares more than one state field of shape `(dim,)`, and the agent
  plugin's tool-builder has the same exactly-one rule
  (`_tools.py:577-583`). In joint mode the action dim is 14, so a declared
  14-dim `joint_eff` is a conformance failure. Undeclared keys are safe:
  `Observation.state` is a plain Mapping, and no core consumer (rollout,
  eval, logging, transcripts, frames) validates runtime keys against the
  space — while the agent policy renders all keys generically
  (`policy.py:1065-1083`).
- A pinning test must still cover a full `reset()`/`step()` cycle with the
  flag on, so any future core-side validation surfaces here before a rig
  does.
- Runtime guard: `_observe()` looks the method up defensively — if
  `report_joint_eff=true` but the injected driver lacks `get_joint_eff`
  (external `driver_factory` implementations predating this change), raise
  a clear `RuntimeError` naming the flag and the missing method, instead of
  a bare `AttributeError`.
- Rerun visibility: explicitly accepted as prompt-only for this change. The
  rerun sink logs all runtime state keys, but the shipped two-row blueprint
  builds panels only from *declared* fields, so `joint_eff` will not appear
  in the viewer. (Declaring it would be worse: the blueprint would overlay
  N·m onto the radian joint-position plots.) A viewer panel is a possible
  follow-up issue, not part of this change.

### Docs

- When the flag is on, append a paragraph to the embodiment docs before
  `EmbodimentInfo` construction, following the conditional depth block
  pattern (`embodiment.py:1240-1258`) — NOT inside `_DOCS_JOINTS`
  (`test_embodiment_docs.py:44-51` pins default docs to the mode constants
  verbatim, so unconditional text would break that contract).
- Content: `state[joint_eff]` is per-joint estimated torque, N·m,
  sign-corrected, same 14-slot layout as `joint_pos` but gripper slots raw;
  includes gravity load, so compare against a moving baseline; rising
  effort while position stops tracking = contact, flat effort with position
  error = controller sag; may lag `joint_pos` by up to one control tick
  (read in separate lock acquisitions).

## Tests

- `FakeDriver` (`tests/test_embodiment.py:26`), `EchoDriver` (`:42`),
  `SettleDriver` (conftest) each grow `get_joint_eff` returning a canned
  14-dim vector. The other fake drivers (`test_eef_embodiment.py:67`,
  `test_health.py:134`, `test_depth_reader.py:886`,
  `test_eval_end_to_end.py:25`, `test_camera_reader.py:334`,
  `test_embodiment.py:877`) get the method too so every in-repo
  `BimanualDriver` implementer conforms, even where the flag stays off.
  (`test_hold_check.py` fakes implement the separate `SingleArm` protocol —
  unaffected.)
- `tests/test_i2rt.py:354-366` pins the wizard option set
  (`{"auto_start", "collision_guardrail"}`) — must be updated to include
  `report_joint_eff`, plus the sibling wizard-default-vs-config-default
  pinning test that convention expects per slot.
- New tests:
  - flag off (default): `"joint_eff" not in obs.state` — contract
    unchanged.
  - flag on: key present, shape `(14,)`, dtype float, values pass through
    from the fake driver un-normalized (gripper slots included).
  - full `reset()` + `step()` cycle with flag on: no warning/error from
    core (pins the undeclared-key assumption).
  - docs: paragraph present iff flag on; default docs still equal the mode
    constants (existing assertions must stay green).
  - config: `report_joint_eff` default False; rejects `none`-coerced
    `None`; rejects non-bool (mirror the bool-flag rejection
    parametrization at `test_config.py:541-549`).
  - eef_pos mode with flag on: `joint_eff` coexists with `eef_state` and
    `joint_pos` without tripping conformance.
  - flag on + driver without `get_joint_eff`: clear `RuntimeError`.
- Gates: `ruff check`, `ruff format --check`, `mypy --strict`,
  `pytest --cov` at 100% (per repo CLAUDE.md).

## Docs/meta

- README option list (`README.md:660-690` area): one entry.
- CHANGELOG `## Unreleased`: feature note.
- Root `CLAUDE.md` (the "14-D `joint_pos` contract" paragraph) and
  `src/inspect_robots_yam/CLAUDE.md` (config-surface table): document the
  optional `joint_eff` state key and the new flag.

## Non-goals

- Joint velocity reporting (same plumbing, easy follow-up if effort proves
  insufficient for stall detection).
- Any effort-based safety/guardrail logic.
- Core or agent-plugin changes (rendering is already generic).
- Labeling/units surfacing in the agent's state line (agent renders new
  keys as bare lists; the docs paragraph carries the semantics).

## Risks

- If a future core version starts validating runtime state keys against
  the declared space, this feature breaks — and declaring the field is not
  a fallback (core conformance forbids a second 14-dim field). The pinning
  test surfaces this at upgrade time; the then-fix is a core-side
  name-aware disambiguation, coordinated upstream.
- `-E report_joint_eff=true` arrives as bool via `_parse_value`; wizard
  writes `true`/`false` strings — both covered by the `OptionSlot` pattern.
- `joint_pos` and `joint_eff` are read in separate lock acquisitions per
  arm, so the two vectors in one Observation can be one control tick
  apart. Negligible at 10 Hz for contact detection; noted in the docs
  paragraph rather than engineered away.
