# 0005 — Gripper normalization polarity: wire 1 = open

## Problem

The 14-D wire format's gripper slots (indices 6, 13) are normalized 0-1.
MolmoAct2 and the BimanualYAM training data use the i2rt convention:
**1 = fully open, 0 = fully closed** (i2rt normalizes the gripper joint via
`JointMapper` over auto-calibrated `[closed, open]` limits, and initializes
`_last_gripper_command_qpos = 1  # fully open`). Verified empirically: in
2,260 episodes sampled from `allenai/MolmoAct2-BimanualYAM-Dataset`, the
episode-initial gripper state is >= 0.978 (open) for the right arm in 100%
of episodes and >= 0.9 for the left in 99.9%.

The plugin maps the opposite way. `_denorm_grippers` sends wire 0 to
`cfg.gripper_open` and wire 1 to `cfg.gripper_closed`:

```python
span = self._cfg.gripper_closed - self._cfg.gripper_open
out[idx] = self._cfg.gripper_open + cmd[idx] * span
```

With the shipped defaults (`gripper_open=0.0`, `gripper_closed=1.0`) this is
the identity map, so on i2rt rigs the *actual behavior today is already
1 = open* — the driver's own convention passes straight through, and the
policy loop works. The bug is latent: the moment an operator sets *real*
calibration values — e.g. a stroke in raw radians, `gripper_open=0.72`,
`gripper_closed=0.0` — every gripper command and observation inverts relative
to the training data: the policy commands "open" (1.0) and the hardware
closes. So this change fixes documentation and calibrated-rig semantics, not
default hardware behavior. The misleading field semantics already caused a
real incident: a downstream rig copied a raw-units start pose into wire units
and ran episodes with the right gripper closed at start, a state absent from
the training set.

## Resolution

Flip the wire convention to match i2rt / MolmoAct2: **wire 1 = `gripper_open`,
wire 0 = `gripper_closed`**, and swap the defaults so the default calibration
stays the identity map.

## Changes

### `config.py`

- Defaults become `gripper_open: float = 1.0`, `gripper_closed: float = 0.0`.
  (Identity map preserved in both directions: wire x -> driver x and
  driver x -> wire x with defaults, same as today.)
- Field comment states the contract: `gripper_open`/`gripper_closed` are the
  *driver-native* positions at the two ends of the stroke; wire value 1 maps
  to `gripper_open`, 0 to `gripper_closed`.
- `rest_pose` / `home_pose` comments gain "(1 = open)" after "normalized 0-1".
- The delta-mode comment block (config.py:45-48, "the gripper can open as
  well as close ... clamp every negative gripper delta to zero") pairs
  "open" with "negative delta"; under the new convention a positive delta
  opens. Reword to keep the mechanical point (deltas need a symmetric box)
  without the inverted pairing.
- Validation (`gripper_open != gripper_closed`) unchanged.

### `embodiment.py`

- `_denorm_grippers`: `span = gripper_open - gripper_closed`;
  `out[idx] = gripper_closed + cmd[idx] * span`.
- `_norm_grippers`: `out[idx] = (physical[idx] - gripper_closed) / span`.
- Docstrings updated to name the convention (wire 1 = open).
- `cfg.low`/`cfg.high` clamp bounds are wire-unit `[0, 1]` on the gripper
  slots and `_send` clamps *before* de-normalizing, so the clamp is
  polarity-agnostic — no change (verified in critique round 1).

### `packing.py`

- No code change. Two normalized-gripper mention sites: the `STATE_KEY`
  comment block (lines 40-41, "the trailing gripper of each arm is
  normalized") and the `state_spec` docstring (line 47). Both gain
  "(1 = open)". (The module docstring does not mention normalization.)

### `plans/0001-yam-molmoact2-design.md` + `CLAUDE.md`

- Plan 0001 records the old convention as a binding resolution
  (`gripper_open=0.0, gripper_closed=1.0` at line ~114; "Gripper
  calibration" resolution at line ~256), and the plugin CLAUDE.md directs
  agents to read 0001 before changing the contract. Add a short
  supersession note at both 0001 sites pointing to this plan (0005), and
  mention 0004 next to the CLAUDE.md pointer. Do not rewrite 0001's
  history.

### `README.md`

- Units table (~line 286): "normalized 0-1 (0 = open, 1 = closed)" becomes
  "normalized 0-1 (1 = open, 0 = closed)".
- "Gripper polarity/trim" section (~lines 265-274): currently states the old
  defaults and recommends `gripper_open=1.0, gripper_closed=0.0` as the
  recipe for an inverted gripper — after the flip that recipe is the identity
  default. Rewrite the section: new defaults, what the two fields mean
  (driver-native stroke endpoints), and that an inverted or offset gripper is
  handled by setting the fields to the *measured* raw endpoints.
- Add a short compatibility note (see below) to the same section.

### Tests (TDD: write first, watch fail)

- Pin the convention with **asymmetric** calibration values: with
  `gripper_open=0.72, gripper_closed=0.04`, wire command `1.0` must reach the
  driver as `0.72`, wire `0.0` as `0.04`; a driver reading of `0.72` must
  observe as wire `1.0`. Midpoint values (0.5) are blind to polarity —
  `denorm(0.5)` is identical under both mappings — so direction-pinning tests
  MUST use off-center values.
- Pin the identity default: wire 0.35 -> driver 0.35 with default config
  (both directions).
- Known midpoint-blind existing tests to fix, not just re-expect:
  - `test_close_rest_pose_goes_through_clamp_and_denorm`
    (tests/test_embodiment.py:349-362) uses gripper 0.5 with open=10,
    closed=20 -> 15.0 under either polarity, and its comment states the old
    formula. Switch to an asymmetric gripper value (e.g. 0.3) and re-derive.
  - `test_close_init_pose_grippers_round_trip_through_normalized_units`
    (tests/test_embodiment.py:396-408) round-trips the exact midpoint 15.0;
    round-trips can't catch polarity by construction. Use an off-center
    physical value; keep it as a round-trip test but don't count it as a
    polarity guard.
- `test_gripper_inverted_polarity_round_trip` (tests/test_embodiment.py:167,
  open=20/closed=10): will fail numerically; its name and comments describe
  the old world ("negative span"/"inverted"). Re-derive: under the new
  formula open=20/closed=10 is a *normal* calibration with positive span.
  Rename/reword accordingly.
- `tests/test_embodiment.py:680` delta-mode comment ("the gripper can open
  (negative delta)") encodes the old direction and silently keeps passing —
  reword alongside the config.py comment.
- Loud-failing tests that hard-code the old formula (re-derive assertions
  AND comments to the new contract, not merely to pass):
  tests/test_embodiment.py:99-107 (asserts 11.0), :127-137 ("normalized
  gripper 0 -> open value"), :150-164 (asserts 13.0), :184-197 (asserts
  16.0; its re-derivation to 14.0 doubles as the delta-mode polarity pin).
- tests/test_embodiment.py:123's comment ("default identity (0..1) -> 1.0")
  silently keeps passing with a stale parenthetical — reword with the :680
  comment.
- README.md:226-229's `rest_pose` example (grippers at 1.0) becomes
  self-consistent after the flip (parks open, documented open); needs no
  edit — do not "fix" it.
- Audit remaining tests for old-direction assumptions and update them to the
  new contract (not merely to pass).

## Compatibility

Pre-1.0; no CHANGELOG file exists, so the compat note lands in the README
gripper section and the next GitHub release notes. Two config classes change
behavior, in opposite directions:

- A config that explicitly copied the old defaults
  (`gripper_open=0.0, gripper_closed=1.0`) now inverts its gripper.
- A config that followed the old README's inversion recipe
  (`gripper_open=1.0, gripper_closed=0.0`) silently *loses* its inversion —
  old code mapped wire x -> 1-x for it; new code maps it to identity.

No known deployment sets either field (the omen rig leaves defaults), so in
practice nothing changes; both classes are still called out in the README
note. Additionally, the gripper slots of every wire-unit config field
(`home_pose`, `rest_pose`, custom `joint_low`/`joint_high`) keep their
numeric behavior on identity-calibrated rigs but flip documented meaning
(e.g. a `joint_high` gripper of 0.9 read "can't fully close", now reads
"can't fully open") — documentation-semantics only, no code change.

## Core-framework divergence (companion change, separate PR)

`inspect_robots.spaces.CANONICAL_STATE_UNITS` (core repo, spaces.py:114)
documents `"gripper": "normalized",  # 0 (open) .. 1 (closed)` as the
encouraged convention. Nothing breaks mechanically (this plugin's state unit
string is `rad+normalized`), but after this plan the flagship hardware plugin
ships 1 = open while the core comment says 0 = open — and the ecosystem
evidence (i2rt, LeRobot-format datasets, MolmoAct2) says the core comment has
it backwards. File a core-repo issue + one-line comment PR flipping that
comment to `# 0 (closed) .. 1 (open)`; do not block this plugin PR on it.

## Out of scope

- Per-arm calibration pairs (left/right strokes differing in raw units).
  i2rt normalizes per-gripper upstream, so one pair suffices today; a future
  plan can add `left_gripper_open`/... if a non-i2rt driver needs it.
- Any change to joint (non-gripper) handling, clamping, or `step_limits`.
