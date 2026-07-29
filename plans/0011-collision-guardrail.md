# Plan 0011: Collision guardrail (MuJoCo CollisionChecker + CollisionApprover)

Issue: [#85](https://github.com/robocurve/inspect-robots-yam/issues/85)
Status: draft, pending adversarial critique rounds.

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

- Predictive collision checking for `yam_arms`: self-collision per arm,
  cross-arm collision, and arm-table collision, evaluated for every action
  before it reaches the driver.
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

- No motion planning, no collision-aware IK (mink's
  `CollisionAvoidanceLimit` in EEF mode is future work; noted in 0006).
- No feedback path into LLM policies. Surfacing "that would collide" as a
  correctable tool error lives in the inspect-robots agent plugin and is
  tracked as a separate issue there.
- No perception: obstacles are the modeled table plane and the arms
  themselves. Scene objects (props on the table) are out of scope for v1.
- No changes to EEF/IK mode internals; the approver sees final absolute
  joint-space actions regardless of which control interface produced them.
- No changes to `YAMEmbodiment` or the core framework.

## 4. Design overview

Three pieces, one new module (`src/inspect_robots_yam/collision.py`):

1. A vendored, collision-only MJCF of a single YAM arm derived from MuJoCo
   Menagerie's `i2rt_yam` model (MIT, i2rt robotics). Primitive capsule,
   sphere, and box collision geoms only: no meshes, no textures, no
   actuators. Small file, loads in milliseconds.
2. `CollisionChecker`: composes a bimanual scene at init (two arms attached
   at configured base offsets plus a table plane) via `mujoco.MjSpec`, then
   answers `check(q14) -> CollisionReport` queries with
   `mj_kinematics` + `mj_collision` on a scratch `MjData`.
3. `CollisionApprover`: implements the framework `Approver` protocol.
   Samples the path from the last approved pose to the incoming action's
   target, queries the checker, and either passes the action through
   untouched (identity preserved, so no spurious transcript events), replaces
   it with a hold at the last safe pose, or raises `SafetyAbort` in strict
   mode.

## 5. Vendored model and generation script

- `scripts/gen_collision_model.py` (dev-time only, not shipped): takes a path
  to a `mujoco_menagerie` checkout, loads `i2rt_yam/yam.xml` with `MjSpec`,
  deletes visual-class geoms, meshes, textures, materials, actuators, and
  keyframes, keeps joints, inertials, collision geoms, and the finger
  equality coupling, and writes
  `src/inspect_robots_yam/assets/yam_collision.xml` with a provenance header
  (menagerie commit hash, source path, license name, regeneration command).
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
  default 0.0.
- `table_height`: float or `None` (None removes the table plane), default
  `0.0` (arm bases sit on the table surface).
- `penetration_threshold`: float, default `1e-3` m. A contact counts as a
  collision only when `contact.dist < -penetration_threshold`. This is the
  same filtering recipe i2rt uses in
  `mujoco_control_interface._has_self_collision` and avoids flagging grazing
  or numerically-jittery contacts.
- `substeps`: int, default 4. Number of interpolation samples (endpoint
  inclusive) between the last approved pose and the incoming target.
- `gripper_qpos`: `"open"` (default) or `"command"`. `"open"` sets both
  finger joints to their widest range for every query: conservative (maximal
  footprint), and sidesteps mapping the normalized gripper command through
  the plan-0005 polarity contract. `"command"` is future work and rejected
  in v1 construction with `ValueError`.

Composition at `CollisionChecker.__init__`:

- `MjSpec` parent with a table plane geom (unless `table_height is None`),
  two `MjSpec.from_string(vendored_xml)` children attached at frames built
  from the configured base poses, prefixes `left_` and `right_`.
- Compile once; allocate one scratch `MjData`. The checker is not
  thread-safe and documents it (rollout is single-threaded).
- Fail fast at init with actionable errors: mujoco missing (guided install
  message, mirroring the `_i2rt.py` style), malformed vendored XML, or a
  compiled model whose joint count does not match the expected 2 x (6 + 2).

## 7. Action-to-qpos mapping

`yam_arms` actions are 14-D absolute `joint_pos`:
`[left j1..j6, left gripper, right j1..j6, right gripper]` (plan 0001).
The compiled scene has nq = 16: per arm 6 revolute joints + 2 finger slide
joints. Mapping:

- Arm joints: direct copy, left block then right block, using joint name
  lookup (`left_joint1` ...), never raw index arithmetic. A test locks the
  mapping against renames.
- Finger joints: from `gripper_qpos` policy (v1: fully open per the model's
  joint ranges).
- Non-finite values in the incoming action: raise `SafetyAbort`, consistent
  with the core approvers' NaN policy.

## 8. CollisionApprover semantics

```
CollisionApprover(checker, *, on_violation="hold")  # or "abort"
```

`review(action, store)`:

1. Read `last = store.get("yam_collision:last_safe")`.
2. First action of the trial (`last is None`): check the target alone. Safe:
   record it and pass through. Colliding: `SafetyAbort` regardless of mode,
   because no safe hold pose exists yet (this means the reset pose itself is
   in collision, which is a rig configuration error worth halting for).
3. Otherwise sample `substeps` points from `last` to the target (endpoint
   inclusive), query the checker on each, stop at the first hit.
4. All safe: update `last_safe`, return the original action object
   (identity preserved: the rollout logs no approval event).
5. Hit with `on_violation="hold"`: return a new `Action` whose values are
   `last_safe` and whose meta merges the original meta with
   `{"collision_blocked": True, "collision_detail": "<geom1>:<geom2>@<k>/<n>"}`.
   The rollout's identity check logs `approval_event(modified=True)`, so
   blocks are visible in the transcript and countable by scorers or the
   operator report.
6. Hit with `on_violation="abort"`: raise `SafetyAbort` with the same
   detail string.

Notes:

- `store` is the per-trial scratch dict the framework passes every approver;
  the key is namespaced `yam_collision:` to coexist with
  `DeltaLimitApprover`'s `delta_limit:last`.
- Chain position: after `ClampApprover`/`DeltaLimitApprover`, so the checker
  sees the post-clamp action that will actually be executed.
- Holding repeatedly is safe on hardware: the embodiment re-commands the
  held pose each step (its own clamp still applies), the arm stays put, and
  a stuck policy either recovers (proprioception shows no progress) or the
  trial times out via `max_steps` and gets scored as a failure. That is the
  entire point: reject the action, not the eval.
- A helper `build_yam_guardrails(action_space, config) -> ChainApprover`
  wires clamp + delta-limit + collision for programmatic `eval()` callers,
  mirroring what the framework CLI builds and appending the collision gate.
  CLI users cannot attach custom approvers today (core has no approver
  registry); the README documents the programmatic path and the limitation.

## 9. Dependencies, packaging, typing

- New optional extra: `collision = ["mujoco>=3.1"]`. The `dev` extra gains
  `mujoco>=3.1` so lint, mypy, and the coverage gate exercise the real
  library in CI (mujoco ships manylinux and macOS wheels; import cost is
  small and there is no GPU or GL context involved because nothing renders).
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

- Model asset: parses, provenance header present, expected joint names, all
  geoms are primitives (no mesh geoms), resolvable via `importlib.resources`.
- Checker: compose default scene; known-safe home pose reports no collision;
  a fold-into-table pose and a cross-arm pose report collisions with the
  offending geom pair named; `penetration_threshold` boundary behavior;
  `table_height=None` removes table contacts; init failures (bad XML, wrong
  joint count) raise the documented errors.
- Approver: pass-through preserves object identity; hold returns last-safe
  values with merged meta and detail string; first-action collision aborts;
  strict mode aborts; non-finite action aborts; store namespacing; substep
  interpolation catches a collision whose endpoints are both safe (sweep
  through the other arm).
- Lazy import error path: inject a broken loader, assert the guided message.
- `build_yam_guardrails`: chain order and integration with the framework's
  `ChainApprover` (inspect-robots is already a base dependency).
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
└── README.md                                (new "Collision guardrail" section)
```

## 12. Delivery

One PR closing #85: this plan, the generator script, the vendored asset, the
module, tests, packaging changes, README section. Plan lands first as a
draft-PR commit; implementation follows on the same branch after critique
rounds converge.

## 13. Resolved questions

- Why Menagerie's model and not i2rt's own MJCF: i2rt's `yam.xml` has
  full-STL geoms with default collidability (crude convex hulls, slower,
  over-approximate); Menagerie has a curated visual/collision split with
  primitives. Both are MIT.
- Why hold instead of skip: the rollout has no skip concept; every step
  executes some action. Holding at the last safe pose is the only in-protocol
  rejection that keeps the trial alive and is honest in the transcript.
- Why not wire into `YAMEmbodiment.step()`: the embodiment's clamp is the
  last line of defense and stays dumb on purpose (safety invariant in
  CLAUDE.md). Prediction belongs in the approver layer the framework already
  provides, where it is optional, composable, and visible in transcripts.
- Why fingers-open for gripper qpos: conservative over-approximation, avoids
  coupling to the plan-0005 polarity mapping in v1, and the error is a few
  centimeters of extra caution.
