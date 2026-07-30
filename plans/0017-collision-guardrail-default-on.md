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

### 1. `YamConfig.collision_guardrail: bool = True`

New field, default `True`: protection is opt-out. Plain-string coercion via
the existing `-E`/config.ini bool parsing; no validation interplay with other
fields (the flag gates *whether we try*, the contribution decides *whether we
can* — keeping mode checks out of `YamConfig.__post_init__` means an eef
config with the default flag stays constructible and degrades at contribution
time with a warning, rather than refusing to run).

### 2. `YAMEmbodiment.contribute_guardrails(action_space)`

Implements the core 0034 protocol. Decision ladder, each rung returning a
`GuardrailContribution`:

- flag `False` → empty contribution, **no warning** (deliberate operator
  opt-out is not a degradation).
- `control_interface != "joints"` or `joints_are_delta` → warning
  `"collision guardrail skipped: absolute joints mode only (plan 0011 v1)"`.
- MuJoCo import fails → warning carrying the existing `_INSTALL_COMMAND`
  (`pip install "inspect-robots-yam[collision]"`).
- Otherwise: `(("yam-collision", CollisionApprover(checker, start_pose,
  action_space=action_space, on_violation="hold")),)` — hold semantics, the
  #86 default: a predicted collision freezes at the last safe pose and
  annotates the action (`collision_blocked`, `collision_detail` meta), so an
  agent-policy run sees *why* nothing moved and can re-plan, and a scripted
  run fails safe instead of aborting mid-episode.

`start_pose` derivation is exactly `build_yam_guardrails`': configured (or
default) home pose clipped to the config's joint bounds. Extract the shared
assembly into a private `_collision_approver(yam_config, action_space)` in
`collision.py`, used by both `build_yam_guardrails` and the contribution, so
the two paths cannot drift. `CollisionApprover.__init__` already rejects a
start pose in collision; that error propagates — a mis-measured rig model is
a bug to surface at startup, not a warning to run past.

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

- Core floor: `inspect-robots>=0.31` (imports `GuardrailContribution`; #225
  precedent for floor discipline).
- `mujoco` stays in the `[collision]` extra. Default-on with the extra absent
  degrades to the install-command warning on every run — visible, actionable,
  and honest, without forcing a ~60 MB dependency on single-arm or
  camera-only installs. The wizard completion summary gains one line nudging
  the extra when the toggle was answered yes (same mechanism as existing
  post-setup hints).

## Not in scope

- EEF-mode collision checking (plan 0011 v1 limitation; tracked separately).
- Gripper-finger collision geometry (`gripper_qpos='command'` unsupported in
  v1; fingers held open).
- Surfacing collision blocks to the agent policy as a correctable tool error
  (`pre_check`, core #210) — composes later; the approver is the backstop
  either way.

## Tests

- Config: default is `True`; `-E collision_guardrail=false` parses; round-trip
  through the wizard writer.
- Contribution ladder: one test per rung (off → empty/no warning; eef mode →
  warning; delta joints → warning; MuJoCo missing (injected import failure) →
  install-command warning; happy path → one named approver, hold mode).
- Contribution approver blocks a cross-arm collision target and holds (reuse
  the #86 scenario fixtures through the contribution path).
- `build_yam_guardrails` and `contribute_guardrails` produce equivalent
  collision approvers (shared-assembly regression).
- Wizard: toggle interviewed with default yes; carried config value suggests
  the carried answer; explicit `false` written when declined.
- CLAUDE.md rows for `collision.py`/`embodiment.py` updated; README guardrail
  section updated; CHANGELOG.

## Release

Minor bump: `inspect-robots-yam 0.22.0`, after core `0.31.0` is on PyPI.
