# 0017 — Collision guardrail default-on

Issue: #93. Depends on: inspect-robots#232 / core plan 0034
(embodiment-contributed guardrails), released as core `0.31.0`.

## Problem

Plan 0011 delivered a complete MuJoCo collision guardrail — `CollisionChecker`
composes the measured bimanual scene, `CollisionApprover` sweeps commanded
targets and holds at the last safe pose, `build_yam_guardrails` chains it
behind clamp + delta-limit — and PR #86 merged it. Nothing calls it. Every
CLI run of a bimanual YAM rig, including two-arms-converging tasks the
guardrail was designed for, runs with only the generic clamp + delta-limit.
Collision protection must be the default, not a library API for callers that
do not exist.

## Design

Three pieces, all riding the new core seam: a config flag (default **on**),
the embodiment contribution, and a wizard toggle so `inspect-robots setup`
interviews it like `auto_start` (#89/#92).

### 1. `YamConfig.collision_guardrail: bool = True`, plus rig geometry

New field, default `True`: protection is opt-out. Plain-string coercion via
the existing `-E`/config.ini bool parsing; no validation interplay with other
fields (the flag gates *whether we try*, the contribution decides *whether we
can* — keeping mode checks out of `YamConfig.__post_init__` means an eef
config with the default flag stays constructible and degrades at contribution
time with a warning, rather than refusing to run).

**Geometry must be configurable, because default-on makes it load-bearing.**
`CollisionConfig`'s base poses are documented as "unverified defaults, not
physical facts" (plan 0011 §6: differing mounting makes cross-arm answers
"silently wrong in both directions"). Under plan 0011 that was fine — the
explicit caller passed a measured config. Under default-on there is no
caller, so `YamConfig` gains optional geometry overrides, `None` meaning
"library default":

```python
collision_left_base_pos: tuple[float, ...] | None = None   # x,y,z — comma string
collision_right_base_pos: tuple[float, ...] | None = None
collision_left_base_yaw: float | None = None
collision_right_base_yaw: float | None = None
collision_table: bool = True                               # False → no table plane
collision_table_height: float | None = None
collision_penetration_threshold: float | None = None
```

Tuple fields ride the existing `_FLOAT_TUPLE_FIELDS` comma-string mechanism;
value validation is delegated to `CollisionConfig` (the contribution builds
one via `replace(_DEFAULT_COLLISION_CONFIG, **set_fields)`), so the two
layers cannot disagree about what is valid. `table_height` and
`penetration_threshold` double as the documented remedies for the two known
false-positive classes (§Behavior changes).

`CollisionConfig.table_height` overloads `None` as "no table plane", so a
single `float | None` YamConfig field cannot express the off-state — and
core's `-E` parser turns the literal string `none` into Python `None`, which
a naive replace() would read as *unset*, silently handing an operator who
typed `collision_table_height=none` the exact table plane they meant to
remove. Hence the split: `collision_table=false` is the off-state (maps to
`table_height=None`), `collision_table_height` sets the height (`None` =
library default), and `from_kwargs` **rejects a parsed-`None`
`collision_table_height`** with an error naming `collision_table=false`.
Setting a height with `collision_table=false` is likewise rejected as
contradictory.

**Unmeasured-geometry warning — one condition, stated once:** the warning
stands **unless both `collision_left_base_pos` and `collision_right_base_pos`
are set** (yaws may legitimately be zero and don't lift it). Half-measured
rigs keep the warning; the text enumerates exactly the fields still at
library default. The contribution stays active regardless — same-arm
self-collision checking is geometry-independent (the arm model is the
packaged system-identified one) and is real protection even unmeasured; the
warning exists so the banner never claims measured *cross-arm* protection
that does not exist. Core 0034's `GuardrailContribution.warnings` docstring
phrases warnings as contributions "declined"; ask core (same PR pair) to
widen the phrasing to "declined or degraded" so a warning alongside an
active approver is within the documented contract.

### 2. `YAMEmbodiment.contribute_guardrails(action_space)`

Implements the core 0034 protocol. Decision ladder, each rung returning a
`GuardrailContribution`:

- flag `False` → empty contribution, **no warning** (deliberate operator
  opt-out is not a degradation).
- `control_interface != "joints"` or `joints_are_delta` → warning
  `"collision guardrail skipped: absolute joints mode only (plan 0011 v1)"`.
- MuJoCo absent → warning carrying the existing `_INSTALL_COMMAND`
  (`pip install "inspect-robots-yam[collision]"`). **Absence is probed with
  `importlib.util.find_spec("mujoco") is None` before any construction** —
  never by catching exceptions around checker construction, because
  `_load_mujoco` converts `ImportError` to `RuntimeError` and
  `CollisionChecker.__init__` raises `ValueError` for compose/compile and
  joint-name failures; a broad except would swallow exactly the
  malformed-model and start-pose bugs that must stay loud.
- Otherwise: `(("yam-collision", CollisionApprover(checker, start_pose,
  action_space=action_space, on_violation="hold")),)` — hold semantics, the
  #86 default: a predicted collision freezes at the last safe pose and
  annotates the recorded action (`collision_blocked`, `collision_detail`
  meta — visible in the transcript and logs; surfacing it to the *policy* as
  a correctable error is core #210, out of scope), plus the
  unmeasured-geometry warning per §1 when applicable.

Everything raised by checker construction and `CollisionApprover.__init__`
propagates. The start-pose-in-collision message in `CollisionApprover`
gains remedy text naming both `collision_guardrail=false` and the
`collision_*` geometry fields, since under default-on it can now fire
because the *model* is wrong rather than the rig.

`start_pose` derivation is exactly `build_yam_guardrails`': configured (or
default) home pose clipped to the config's joint bounds (safe seeding: core
rollout hands each trial a fresh approver store, and `YAMEmbodiment.reset()`
unconditionally ramps to the home pose every episode, so the seed equals the
actually-commanded start). Extract the shared assembly into a private
`_collision_approver(yam_config, action_space)` in `collision.py`, used by
both `build_yam_guardrails` and the contribution, so the two paths cannot
drift. `CollisionApprover.review` switches its delta-limiter rewind from the
duplicated private string literal to the public
`DeltaLimitApprover.rewind_reference` seam added by core plan 0034 §1b.

The checker is constructed inside the contribution (lazily, on the CLI's
single call at run setup), preserving collision.py's optional-dependency
boundary: importing the embodiment still never imports MuJoCo.

### 3. Wizard toggle

```python
OptionSlot(
    arg="collision_guardrail",
    label="Block predicted arm collisions before they happen (collision_guardrail)",
    default=True,
),
```

appended to `YAMEmbodiment.OPTION_SLOTS`. The wizard writes an explicit
`true`/`false` into `[embodiment.args]`; existing configs without the key get
the dataclass default (`True`) at construction, so *every* rig — wizard-run
or hand-written config — is protected after upgrading, and the wizard is
where an operator consciously declines.

### 4. Dependencies

- Core floor: `inspect-robots>=0.31` (imports `GuardrailContribution` and
  `DeltaLimitApprover.rewind_reference`; #225 precedent for floor
  discipline).
- `mujoco` stays in the `[collision]` extra. Default-on with the extra absent
  degrades to the install-command warning on every run — visible, actionable,
  and honest, without forcing a ~60 MB dependency on single-arm or
  camera-only installs. No wizard-completion nudge: there is no plugin seam
  for conditional post-setup hints (core's are static ClassVar-driven), and
  the run-time warning is the honest discovery mechanism; inventing a core
  hook for one hint is not worth the coupling.

## Behavior changes on upgrade

Flipping the default changes results, not just safety, and the release notes
must say so. Enumerated by run configuration:

- **joints-absolute rig, mujoco importable** (the common case — eef-capable
  installs already pull mujoco via the mink kinematics stack, so "importable"
  is much broader than "installed the `[collision]` extra"): the guardrail
  silently activates. True-positive holds are the feature. Two *known
  false-positive classes* (plan 0011 §6) change eval outcomes:
  - *Table-press grasps*: demo-derived targets often press slightly into the
    table; a grasp-moment hold freezes the arm. Remedy now exists in config:
    raise `collision_penetration_threshold` or lower
    `collision_table_height`.
  - *Bimanual close-quarters work* (handovers, clapping): fingers-held-open
    inflates the cross-arm footprint by centimeters exactly where these
    tasks operate. A policy that keeps re-commanding the blocked target
    livelocks at the held pose until `max_steps` and scores a failure —
    physically safe, but a scoring regression. Remedy: measured base
    geometry (shrinks the false margin), or per-rig opt-out.
  README and CHANGELOG call these out explicitly, with the remedies
  (tableless rigs: `collision_table=false`).
- **joints-absolute rig, mujoco importable, home pose in collision under the
  effective geometry**: the run **refuses to start** —
  `CollisionApprover.__init__` rejects a colliding start pose, and that
  error deliberately propagates (loud-bug stance). Under unmeasured default
  geometry this can hit a previously-working rig with non-default mounting
  or a custom home pose. The error message names the way out
  (`collision_guardrail=false`, or fixing the `collision_*` geometry); the
  CHANGELOG lists this alongside the false-positive classes as the harshest
  upgrade edge.
- **joints-absolute rig, no mujoco**: no behavior change beyond a new
  banner warning with the install command.
- **eef/delta-joints rigs**: no behavior change beyond the skip warning.
- **`collision_guardrail=false`**: exact pre-0017 behavior.

Hold (not abort) remains the right unattended default — an abort tears down
the episode a true positive just saved — but the livelock-until-`max_steps`
cost on false positives is acknowledged above and is the price of failing
safe. The README also notes the guardrail models *commanded* poses: arms can
sag away from checked waypoints — including under the default
`zero_gravity_mode=true`, so this is every default rig, not an edge
configuration — and clearance margins should not be shaved to zero.

## Not in scope

- EEF-mode collision checking (plan 0011 v1 limitation; tracked separately).
- Gripper-finger collision geometry (`gripper_qpos='command'` unsupported in
  v1; fingers held open).
- Surfacing collision blocks to the agent policy as a correctable tool error
  (`pre_check`, core #210) — composes later; the approver is the backstop
  either way.

## Tests

- Config: default is `True`; `-E collision_guardrail=false` parses; round-trip
  through the wizard writer; geometry fields parse from comma strings and
  reject malformed values via `CollisionConfig`'s own validation; parsed-
  `None` `collision_table_height` rejected naming `collision_table=false`;
  height-with-`collision_table=false` rejected as contradictory; an
  eef-mode and a delta-joints `YamConfig` with default `collision_guardrail`
  stay constructible (the no-`__post_init__`-interplay decision is
  load-bearing for upgrade safety — pin it).
- Contribution ladder: one test per rung (off → empty/no warning; eef mode →
  warning; delta joints → warning; MuJoCo absent (find_spec probe seam) →
  install-command warning; happy path → one named approver, hold mode).
- Loud-failure routing: mujoco present but model malformed (existing
  `model_xml` seam) → propagates through `contribute_guardrails`, not a
  warning; start pose in collision → propagates, message names the opt-out
  flag and geometry fields.
- Geometry: configured `collision_*` fields reach the built
  `CollisionConfig`; warning stands unless **both base positions** are set
  (one side set → warning persists, enumerating the defaulted fields; both
  set → no warning); `collision_table=false` reaches the checker as
  `table_height=None` (no table geom).
- Contribution approver blocks a cross-arm collision target and holds (reuse
  the #86 scenario fixtures through the contribution path); rewind goes
  through `DeltaLimitApprover.rewind_reference` (assert against the public
  seam, not a string literal).
- `build_yam_guardrails` and `contribute_guardrails` produce equivalent
  collision approvers (shared-assembly regression).
- Wizard (declarative, the auto_start precedent — the interview loop is
  core's, and core's generic wizard tests already cover interview/carry/
  write-back for any declared slot): pin the new OptionSlot's arg, label,
  and default-true, and its pairing with the `YamConfig` bool. Update the
  existing assertions the new slot breaks: `test_i2rt.py` pins
  `{option.arg} == {"auto_start"}` and single-slot-unpacks `OPTION_SLOTS`.
- CLAUDE.md rows for `collision.py`/`embodiment.py` updated; README guardrail
  section updated (false-positive classes + remedies, commanded-pose caveat);
  CHANGELOG flags the default flip as results-affecting.

## Release

Minor bump: `inspect-robots-yam 0.22.0`, after core `0.31.0` is on PyPI.
