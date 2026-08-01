# Plan 0011: Collision guardrail (MuJoCo CollisionChecker + CollisionApprover)

Issue: [#85](https://github.com/robocurve/inspect-robots-yam/issues/85)
Status: revised after adversarial critique rounds 1 (14 findings) and 2
(9 findings), all addressed below.

## 1. Problem

The safety layer is geometry-blind. `YAMEmbodiment.step()` clamps to
`joint_low/high`, EEF mode clamps per-step IK deltas and a Cartesian box, and
the framework guardrails (`ClampApprover`, `DeltaLimitApprover`) clamp boxes
and deltas. None of them can see that a within-limits 14-D target folds the
left arm into the right one or drags a link through the table.
`plans/0006-eef-control-interface.md` (lines 325-327) states this gap loudly:
no arm-table or arm-arm collision checking exists, and the two arms' default
y-ranges overlap in a bimanual workspace.

The failure mode this plan targets: a policy (VLA or LLM agent) emits a
plausible-looking joint target, every existing clamp passes it, and the arms
collide at hardware speed. Today the only defense is the operator's hand on
the e-stop.

## 2. Goals

- Predictive collision checking for `yam_arms` in absolute joint-position
  mode: self-collision per arm, cross-arm collision, and arm-table collision,
  evaluated for every action before it reaches the driver.
- Ship it as an `inspect_robots.approver.Approver` so it slots into the
  existing guardrail chain with zero framework changes.
- Rejection must not kill the eval by default: a blocked action becomes a
  hold at the last known-safe pose, recorded in the transcript. A strict mode
  escalates to `SafetyAbort` for users who want hard-stop semantics.
- Negligible runtime cost (measured: 16 us per configuration for the
  bimanual-plus-table scene, 0.33 ms for a 20-waypoint sweep on one CPU core).
- mujoco stays an optional dependency, lazily imported behind a seam, so the
  `import-hygiene` gate keeps passing.

## 3. Non-goals

- No support for the EEF control interface (10-D Cartesian actions) or the
  `joints_are_delta` mode in v1. The approver runs before
  `embodiment.step()`, so in those modes it would see Cartesian vectors or
  deltas, not absolute joint targets; interpreting them as qpos would be
  wrong in both directions, and a "hold" replacement would be executed as a
  delta (a violent jump). `CollisionApprover` and `build_yam_guardrails`
  therefore refuse at construction unless
  `action_space.semantics.control_mode == "joint_pos"` and the space is
  14-D, with a `ValueError` naming this plan. This guard is sound because
  the embodiment declares `joint_delta` when `joints_are_delta=True` and
  `eef_abs_pose` in EEF mode; the CLAUDE.md safety-invariants paragraph
  still claiming "there is no `joint_delta` control mode in Inspect
  Robots" predates the framework adding it and is updated in this PR. EEF
  support means checking post-IK joint paths inside the embodiment and is
  future work.
- No motion planning, no collision-aware IK (mink's
  `CollisionAvoidanceLimit` in EEF mode is future work; noted in 0006).
- No feedback path into LLM policies. Surfacing "that would collide" as a
  correctable tool error lives in the inspect-robots agent plugin and is
  tracked as a separate issue there.
- No perception: obstacles are the modeled table plane and the arms
  themselves. Scene objects (props on the table) are out of scope for v1.
- No changes to `YAMEmbodiment` or the core framework.

## 4. Design overview

Three pieces, one new module (`src/inspect_robots_yam/collision.py`):

1. A vendored, collision-only MJCF of a single YAM arm derived from MuJoCo
   Menagerie's `i2rt_yam` model (MIT, i2rt robotics). Primitive capsule,
   sphere, and box collision geoms only: no meshes, no textures, no lights,
   no actuators. Small file, loads in milliseconds.
2. `CollisionChecker`: composes a bimanual scene at init (two arms attached
   at configured base offsets plus a table plane) via `mujoco.MjSpec`, then
   answers `check(q14) -> CollisionReport` queries with
   `mj_kinematics` + `mj_collision` on a scratch `MjData`.
3. `CollisionApprover`: implements the framework `Approver` protocol.
   Sweeps from the last approved pose to the incoming action's target,
   queries the checker, and either passes the action through untouched
   (identity preserved, so no spurious transcript events), replaces it with
   a hold at the last safe pose, or raises `SafetyAbort` in strict mode.

What the sweep is and is not: it interpolates between successive commanded
targets. With `settle_tolerance=None` (the default) the physical arm lags
its targets, so the checked path models commanded intent, not measured
motion. That is the right contract for a predictive guardrail, and the
README says so explicitly: this reduces collision risk; it does not replace
the operator's e-stop.

## 5. Vendored model and generation script

- `scripts/gen_collision_model.py` (dev-time only, not shipped): takes a path
  to a `mujoco_menagerie` checkout, loads `i2rt_yam/yam.xml` with `MjSpec`,
  deletes visual-class geoms, meshes, textures, materials, lights, cameras,
  actuators, keyframes, `<option>` and compiler settings (the source model
  sets `integrator="implicitfast"`, which otherwise triggers an MjSpec
  attach-conflict warning twice per composition at user runtime), and the
  finger equality constraint (inert under
  `mj_kinematics` + `mj_collision`, which run no constraint solve; both
  fingers are posed explicitly instead, see §7), keeps joints, inertials,
  and collision geoms, and writes
  `src/inspect_robots_yam/assets/yam_collision.xml` with a provenance header
  (menagerie commit hash, source path, license name, regeneration command).
- The generator finishes by self-checking its output: it composes the
  default bimanual scene exactly as `CollisionChecker` does and compiles it,
  so a menagerie update that breaks stripping (dangling class or name
  references, default-class collisions across the two attached subtrees)
  fails at generation time, not at user runtime.
- `src/inspect_robots_yam/assets/MENAGERIE_LICENSE`: verbatim copy of the
  upstream MIT license text.
- The artifact is committed. Regeneration is manual and rare (upstream model
  updates). A unit test asserts the provenance header exists and the XML
  parses.
- Packaging: `assets/` lives inside the package dir, so the existing
  hatchling wheel config ships it. Add a test that resolves the asset via
  `importlib.resources` rather than `__file__` math.

## 6. Scene composition and configuration

New frozen dataclass `CollisionConfig` in `collision.py` (kwargs-driven like
`YamConfig`, but independent of it; the approver is constructed explicitly,
not by the embodiment):

- `left_base_pos`, `right_base_pos`: `(x, y, z)` tuples, defaults
  `(0.0, +0.3, 0.0)` and `(0.0, -0.3, 0.0)` to match the default rig
  described in plan 0001. `left_base_yaw`, `right_base_yaw` in radians,
  default 0.0. These are unvalidatable claims about the physical rig: if
  the measured mounting differs, cross-arm answers are silently wrong in
  both directions. The README documents this with the same weight as the
  CLAUDE.md safety invariants: measure the rig, set the offsets.
- `table_height`: float or `None` (None removes the table plane), default
  `0.0` (arm bases sit on the table surface). Known tradeoff, stated like
  the fingers-open one below: a legitimate tabletop grasp commands
  fingertips within millimeters of the plane, and demo-derived VLA targets
  often press slightly into it, so grasp-moment holds are possible. The
  remedies, in preference order, are raising `penetration_threshold`,
  lowering `table_height` by a few millimeters of margin, or (future work)
  excluding fingertip geoms from plane checks. The README documents this
  next to the cross-arm tradeoff.
- `penetration_threshold`: float, default `1e-3` m. A contact counts as a
  collision only when `contact.dist < -penetration_threshold`. Same
  filtering recipe i2rt uses in
  `mujoco_control_interface._has_self_collision`; avoids flagging grazing
  or numerically-jittery contacts.
- `sweep_resolution`: float, default `0.05` rad. The sweep between the last
  safe pose and the target samples
  `ceil(max_abs_joint_delta / sweep_resolution)` interior points, clamped to
  [1, 64], endpoint always included. `max_abs_joint_delta` is computed over
  the 12 arm-joint dims only: the gripper dims (6 and 13) are ignored by
  the qpos mapping (§7), and letting a normalized gripper toggle (delta
  1.0) force 20 geometrically identical substeps would be pure waste.
  Resolution-based sampling (rather than a fixed count) keeps fine geoms
  (millimeter fingertip spheres) covered even when the approver is used
  without a delta limiter upstream. Honesty note: the 64-substep cap means
  a full-range standalone swing (~2 pi) is sampled at ~0.1 rad, coarse
  enough to hop past millimeter geoms; in the documented
  `build_yam_guardrails` chain the delta limiter bounds each sweep well
  under the cap, so the cap never binds there. The guardrail reduces risk;
  it does not certify paths.
- `gripper_qpos`: `"open"` (default) or `"command"`. `"open"` poses both
  finger joints at their open extremes for every query: conservative
  (maximal footprint). Note the sign trap: after stripping, `left_finger`
  opens toward its range maximum but `right_finger` opens toward its range
  minimum; the checker reads each joint's open extreme by name, never by a
  shared sign convention, and a test locks both. `"command"` is future work
  (requires the plan-0005 polarity mapping) and rejected in v1 construction
  with `ValueError`. Known tradeoff: fingers-open inflates the cross-arm
  footprint by a few centimeters exactly where bimanual handovers operate;
  if a legitimate handover trips the guardrail, `"command"` mode is the
  remedy and this tradeoff is documented in the README rather than hidden.

Composition at `CollisionChecker.__init__`:

- `MjSpec` parent with a table plane geom (unless `table_height is None`),
  two `MjSpec.from_string(vendored_xml)` children attached at frames built
  from the configured base poses, prefixes `left_` and `right_`.
- No contact excludes are needed for the base-vs-table case, and none are
  added. The Menagerie base collision capsule does extend 17 mm below the
  base frame origin (`geom size="0.033 0.01" pos="0 0 0.026"`, a z capsule
  whose bottom is at -0.017), but the arm's root body (named `arm`, so
  `left_arm`/`right_arm` after prefixed attach) is jointless: frame-attach
  welds it to the world, and MuJoCo's default parent/weld contact filtering
  already suppresses base-vs-plane pairs. Verified empirically on the
  composed scene: at qpos=0 with the plane present, the only contacts are
  finger-vs-finger at -0.0004 m (under the 1e-3 threshold), and no
  base-table contact is generated, while reach-down poses still produce
  link/finger-vs-plane contacts as desired. The §10 sanity tests pin this
  behavior (home and folded poses collision-free with the plane present)
  so a future MuJoCo filtering change fails loudly here.
- Compile once; allocate one scratch `MjData`. The checker is not
  thread-safe and documents it (rollout is single-threaded).
- Fail fast at init with actionable errors: mujoco missing (guided install
  message, mirroring the `_i2rt.py` style), malformed vendored XML, or a
  compiled model whose joint names do not match the expected
  `{left,right}_joint1..6` + `{left,right}_{left,right}_finger` set.
- Testability seam: `CollisionChecker` accepts an optional `model_xml`
  string overriding the packaged asset (default `None` loads it via
  `importlib.resources`). This is how the malformed-XML and
  missing-joint-name failure paths are covered without shipping a second
  bad asset (§10).

## 7. Action-to-qpos mapping

`yam_arms` actions in `joint_pos` mode are 14-D absolute:
`[left j0..j5, left gripper, right j0..j5, right gripper]` (packing.py
`DIM_LABELS`). The compiled scene has nq = 16: per arm 6 revolute joints +
2 finger slide joints. Mapping:

- Arm joints: direct copy by joint name lookup. The package labels joints
  `j0..j5` while the Menagerie model names them `joint1..joint6`; the
  correspondence is explicit and off-by-one by design: action dim
  `left_j0` -> model joint `left_joint1`, ... `left_j5` -> `left_joint6`,
  and likewise for the right block. A table in the module docstring and a
  test lock this down; no raw index arithmetic anywhere.
- Finger joints: posed from the `gripper_qpos` policy (v1: open extremes,
  per-joint sign-aware as in §6). The action's gripper dims are ignored in
  v1 and this is documented.
- Non-finite values in the incoming action: raise `SafetyAbort`. This is
  deliberately stricter than core (`ClampApprover` aborts on NaN but clamps
  inf): a qpos query with inf is meaningless, and in the documented chain
  Clamp runs first so inf never reaches this approver anyway.

## 8. CollisionApprover semantics

```
CollisionApprover(checker, start_pose, *, on_violation="hold")  # or "abort"
```

`start_pose` is the 14-D pose the trial physically starts from (the
embodiment's home/reset pose). `build_yam_guardrails` derives it from
`YamConfig` the same way the embodiment's reset path does:
`home_pose or DEFAULT_JOINT_HOME_POSE` (the config field defaults to
`None`; the factory default lives in `config.py`), then clamped to
`joint_low/high`, because reset commands the clamped home and the seed must
equal what was actually commanded. `build_yam_guardrails` also cross-checks
its two inputs: `yam_config.joints_are_delta=True` raises `ValueError` even
if the caller hands it a hand-built `joint_pos` box, so an inconsistent
pairing cannot slip past the §3 guard.

`review(action, store)`:

1. Read `last = store.get("yam_collision:last_safe")`; if absent (first
   action of the trial), seed `last = start_pose`. There is no special
   first-action abort: the first sweep runs from the physical start pose to
   the first target, exactly like every later step. (If `start_pose` itself
   is in collision under the model, the `CollisionApprover` constructor
   queries the checker and refuses with a clear error: that is a rig or
   config modeling error and should fail before any trial runs, not during
   one. The check lives in the approver because the checker never sees
   `start_pose`.)
2. Sweep from `last` to the target at `sweep_resolution` (§6), stop at the
   first hit.
3. All safe: update `yam_collision:last_safe`, return the original action
   object (identity preserved: the rollout logs no approval event).
4. Hit with `on_violation="hold"`: return a new `Action` whose values are
   `last_safe` and whose meta merges the original meta with
   `{"collision_blocked": True, "collision_detail": "<geom1>:<geom2>@<k>/<n>"}`,
   minus the core clamp flags (`clamped`, `delta_clamped`) if upstream
   approvers set them. The rollout builds the approval event's `detail`
   from those core flags; carrying them through would mislabel a collision
   block as a mere clamp, and their referent (the clamped values) is being
   discarded anyway. Additionally re-anchor `store["delta_limit:last"] = last_safe` if that
   key exists. Without this, `DeltaLimitApprover`'s reference keeps walking
   toward the blocked region during a run of holds, and the first target
   that clears the collision check could then be executed as one unbounded
   jump from the held pose; re-anchoring restores the invariant that
   consecutive *commanded* actions differ by at most the delta limit. This
   deliberately couples to `DeltaLimitApprover`'s store key; the coupling
   lives inside `build_yam_guardrails`'s documented chain and a test pins
   the upstream key name so a rename fails loudly here rather than
   silently.
5. Hit with `on_violation="abort"`: raise `SafetyAbort` with the same
   detail string.

Transcript visibility, stated precisely: the rollout's identity check logs
`approval_event(modified=True)` for every hold, so blocks are visible and
countable in the transcript. The event's `detail` field only understands
the core clamp flags, which the hold strips (§8.4), so it stays empty; the
geom-pair string travels in the recorded action's `meta`
(`StepRecord.action.meta`), which scorers and the operator report can
read. No core changes.

Notes:

- `store` is the per-trial scratch dict the framework passes every approver;
  keys are namespaced `yam_collision:` alongside `delta_limit:last`.
- Chain position: after `ClampApprover`/`DeltaLimitApprover`, so the checker
  sees the post-clamp action that will actually be executed.
- Holding repeatedly is safe on hardware: the embodiment re-commands the
  held pose each step (its own clamp still applies), the arm stays put, and
  a stuck policy either recovers (proprioception shows no progress) or the
  trial times out via `max_steps` and gets scored as a failure. That is the
  entire point: reject the action, not the eval.
- `build_yam_guardrails(action_space, yam_config, collision_config) ->
  ChainApprover` wires clamp + delta-limit + collision for programmatic
  `eval()` callers, mirroring what the framework CLI builds and appending
  the collision gate. It applies the same `control_mode == "joint_pos"`
  construction guard (§3). CLI users cannot attach custom approvers today
  (core has no approver registry); the README documents the programmatic
  path and the limitation.

## 9. Dependencies, packaging, typing

- New optional extra: `collision = ["mujoco>=3.3.1"]`. The floor is 3.3.1,
  not 3.1: the unified `MjSpec.attach` / `mjs_attach` API with prefix
  namespacing that the composition relies on landed in 3.3.1 (MuJoCo
  changelog). The implementer verifies the exact floor locally before
  committing it; py3.10 wheels exist at that floor, so `requires-python`
  is unaffected.
- The `dev` extra gains the same mujoco pin so lint, mypy, and the coverage
  gate exercise the real library in CI (manylinux and macOS wheels; no GPU
  or GL context involved because nothing renders).
- `uv lock` regenerated (CI installs from the lockfile).
- Lazy import seam: `collision.py` imports mujoco inside a
  `_load_mujoco()` helper with a guided-install `RuntimeError` on
  `ImportError`, following `_i2rt.py`. Module import stays clean for the
  `import-hygiene` job.
- mypy: add `mujoco.*` to the `ignore_missing_imports` override list if the
  shipped stubs are incomplete for `MjSpec`; keep strict mode for our code.

## 10. Testing strategy

CI has mujoco (dev extra), so tests run the real physics; no physics mocks.
100% coverage targets:

- Model asset: parses, provenance header present, expected prefixed joint
  names after composition (`left_joint1`..`left_joint6`,
  `left_left_finger`, `left_right_finger`, right-side mirrors), all geoms
  are primitives (no mesh geoms), resolvable via `importlib.resources`.
- Default scene sanity (the round-1 blocker class): the configured home
  pose is collision-free in the default scene with the table plane
  present; the all-zero folded rest pose (a physically legal pose where
  the wrist doubles back near the upper arm) likewise; base-vs-plane
  contacts are absent (weld filtering, §6) while a reach-down pose does
  produce link-vs-plane contacts. If the primitive model reports
  penetration at a known-legal pose, that is a modeling bug to fix in the
  generator or thresholds, and these tests are what catch it.
- Checker: a fold-into-table pose and a cross-arm pose report collisions
  with the offending geom pair named; `penetration_threshold` boundary
  behavior; `table_height=None` removes table contacts; finger open
  extremes are sign-correct per side; init failures (bad XML, missing
  joint names) raise the documented errors; the colliding-start-pose
  refusal is an approver-constructor test.
- Approver: construction guard rejects EEF (10-D) and `joint_delta`
  semantics; `build_yam_guardrails` rejects `joints_are_delta=True` and
  resolves the clamped home-pose fallback when `home_pose` is `None`;
  pass-through preserves object identity; hold returns last-safe values
  with merged meta (core clamp flags stripped) and detail string; hold
  re-anchors
  `delta_limit:last` when present (and a companion test asserts the
  framework still uses that key name); strict mode aborts; non-finite
  action aborts; first action sweeps from `start_pose`; sweep catches a
  collision whose endpoints are both safe (pass through the other arm);
  resolution-derived substep count clamps at both bounds.
- Lazy import error path: inject a broken loader, assert the guided message.
- `build_yam_guardrails`: chain order, start-pose sourcing from YamConfig,
  and integration with the framework's `ChainApprover` (inspect-robots is
  already a base dependency).
- Perf regression (marker `perf`, generous bound like the existing one):
  bimanual check stays under 1 ms per configuration.

Anything genuinely unreachable without hardware stays out of this module by
construction; no `pragma: no cover` expected.

## 11. File tree

```
inspect-robots-yam/
├── plans/0011-collision-guardrail.md        (this document)
├── pyproject.toml                           (collision extra, dev += mujoco, mypy override)
├── uv.lock                                  (regenerated)
├── scripts/gen_collision_model.py           (new, dev-time)
├── src/inspect_robots_yam/
│   ├── assets/
│   │   ├── yam_collision.xml                (new, vendored, generated)
│   │   └── MENAGERIE_LICENSE                (new)
│   └── collision.py                         (new: CollisionConfig, CollisionChecker,
│                                             CollisionReport, CollisionApprover,
│                                             build_yam_guardrails)
├── tests/test_collision.py                  (new)
├── CLAUDE.md                                (safety-invariants paragraph: joint_delta
│                                             is now a declared control mode)
└── README.md                                (new "Collision guardrail" section)
```

## 12. Delivery

One PR closing #85: this plan, the generator script, the vendored asset, the
module, tests, packaging changes, README section, and the CLAUDE.md
correction. The README section follows the repo's public writing-style
rules (no em dashes in prose, bold discipline, headers with colons). Plan
lands first as a draft-PR commit; implementation follows on the same branch
after critique rounds converge.

## 13. Resolved questions

- Why Menagerie's model and not i2rt's own MJCF: i2rt's `yam.xml` has
  full-STL geoms with default collidability (crude convex hulls, slower,
  over-approximate); Menagerie has a curated visual/collision split with
  primitives. Both are MIT.
- Why hold instead of skip: the rollout has no skip concept; every step
  executes some action. Holding at the last safe pose is the only
  in-protocol rejection that keeps the trial alive and is honest in the
  transcript.
- Why not wire into `YAMEmbodiment.step()`: the embodiment's clamp is the
  last line of defense and stays dumb on purpose (safety invariant in
  CLAUDE.md). Prediction belongs in the approver layer the framework
  already provides, where it is optional, composable, and visible in
  transcripts. The corollary (§3): the approver layer only works where
  actions are already absolute joint positions, so EEF and delta modes are
  guarded out rather than half-supported.
- Why fingers-open for gripper qpos: conservative over-approximation and no
  coupling to the plan-0005 polarity mapping in v1; the handover tradeoff
  and its remedy are documented (§6).
- Why joint_pos only: see §3. The guard turns a silent wrong answer into a
  loud constructor error, which is the only acceptable failure mode for a
  safety component.
