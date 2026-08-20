# 0026: Full 6-DoF EEF targets (pitch + roll slots, pinned by default)

Issue: #133. Branch: `feat/eef-6dof`.

## Goal

The EEF action space grows from 5 to 7 slots per arm —
`(x, y, z, yaw, pitch, roll, gripper)`, 14 total — with pitch and roll
**pinned at zero by default** so every existing config, agent prompt, and
motion behaves byte-identically until an operator opens an axis with
`-E eef_low=…`/`-E eef_high=…`. The underlying i2rt IK already takes a full
4×4 pose; this plan is target construction, orientation extraction,
layout migration, and validation — no solver work.

## Orientation convention (the load-bearing decision)

All orientation slots are **relative to the trial's start orientation**
(the yaw slot already is). The commanded relative rotation, applied in the
arm-base frame, is

    R_delta = Rz(yaw) · Ry'(pitch) · Rx(roll),   target = R_delta · reference

where `Rz` is the existing vertical-axis rotation, `Rx` is a standard
rotation about base +x, and `Ry'` is rotation about base **−y**, chosen so
that at yaw 0 a positive pitch tips a gripper-down tool **forward** (+x,
radially outward) — matching how operators already talk about "pitching the
gripper 30–45° forward". Positive roll tips the tool toward the arm's left
(+y) at yaw 0. `yaw = pitch = roll = 0` keeps the start orientation, and
`pitch = roll = 0` reproduces today's `Rz(yaw) · reference` exactly.

**Extraction** inverts the same composition: from a measured pose,
`R_delta = R_measured · referenceᵀ`, decomposed as ZYX Euler (with the Ry'
sign convention). Because `R_delta` is near identity for any pose the
default box permits, this is well-conditioned everywhere it is reachable;
the decomposition's singularity sits at |pitch| = π/2, which config
validation excludes (below).

**This supersedes the `_yaw_axis` fallback machinery.** The current
extraction reads a column of the *absolute* rotation and needs
`_YAW_FALLBACK_THRESHOLD` because a near-vertical tool axis degenerates it.
The relative-rotation extraction never inspects the absolute pose, so
`_yaw_zero`, `_yaw_axis`, the `yaw_axis` property, and the fallback
threshold are deleted. For measured poses inside the yaw-only family the
two schemes return identical yaw (planar-rotation angle addition); for
real-hardware poses with small pitch/roll sag the new value is the proper
ZYX yaw rather than a column projection — an improvement, and a sanctioned
change to any test that asserted the old fallback behavior.
`capture_yaw_reference` keeps its name and its call site; it now stores
only the reference rotation.

## Changes

1. `config.py`
   - `EEF_DIM_LABELS`: per-arm parts become
     `(x, y, z, yaw, pitch, roll, gripper)` (14 labels).
   - `_EEF_ARM_LOW = (0.15, -0.25, 0.03, -π, 0.0, 0.0, 0.0)`,
     `_EEF_ARM_HIGH = (0.48, 0.25, 0.40, π, 0.0, 0.0, 1.0)` — pitch and
     roll pinned at (0, 0).
   - Bounds validation: the strict `eef_low >= eef_high` rejection relaxes
     to `eef_low > eef_high` — equality now means a pinned axis (the
     mechanism `action_box` already documents for pinned dimensions).
     Yaw (indices 3, 10) and roll (5, 12) bounds must stay within [-π, π];
     pitch (4, 11) bounds must stay strictly inside (-π/2, π/2), which
     keeps the ZYX decomposition away from its singularity by construction.
   - `action_semantics`: eef gripper `max_step` indices move from (4, 9)
     to (6, 13). `rotation_repr` stays `"none"`: it is compatibility-checked
     as a hard error against policy declarations in the framework, and the
     framework's accepted vocabulary is out of this repo's hands — changing
     it would break every existing policy pairing for zero enforcement
     benefit. Revisit upstream later if the framework grows a vocabulary
     for it.
   - `start_pose` comment: box-check sentence now says relative yaw, pitch,
     and roll 0.
2. `kinematics.py`
   - `solve()` takes 6 target values; target rotation built per the
     convention above (`_rotation_y_fwd`, `_rotation_x` join `_rotation_z`).
   - `observe()` returns 7 values `(x, y, z, yaw, pitch, roll, gripper)`
     via the relative-rotation ZYX extraction (`_wrap_yaw` reused for yaw
     and roll; pitch needs no wrap, asin range).
   - Delete `_yaw_zero`, `_yaw_axis`, `yaw_axis` property,
     `_YAW_FALLBACK_THRESHOLD`, `_extract_yaw`; `capture_yaw_reference`
     stores the reference rotation only.
3. `embodiment.py`
   - `_step_eef`: left target `action[:6]`, left gripper `action[6:7]`,
     right target `action[7:13]`, right gripper `action[13:14]`; docstring
     10-D → 14-D.
   - `_validate_eef_home`: home state per arm becomes
     `(*position, 0.0, 0.0, 0.0, gripper)` against bounds slices `0:7` /
     `7:14`.
   - `_DOCS_EEF_POS`: document the two new slots, their signs (positive
     pitch tips the tool forward at yaw 0; positive roll toward the arm's
     left), the relative-to-start convention, and that a pinned axis
     (low == high) is not commandable.
4. `README.md`: EEF section documents the 7-slot layout, the
   pinned-by-default pitch/roll, how to open an axis per run, and a
   warning that opening pitch/roll invalidates the z-floor's
   fingertips-down assumption — raise `eef_low` z accordingly.
5. `CHANGELOG.md` (Unreleased/Changed — this is a breaking layout change):
   the old→new slot mapping spelled out, with the "identical behavior at
   default bounds" guarantee.

## Verified before implementation

- Orientation math checked numerically (2000-sample roundtrip, worst error
  6e-14; `pitch=roll=0` reproduces `Rz(yaw)·reference` exactly; positive
  pitch tips a gripper-down tool axis toward +x and positive roll toward
  +y; old and new yaw extraction agree exactly for poses inside the
  yaw-only family away from the old scheme's degenerate case).
- The framework `Box` (inspect_robots/spaces.py) explicitly supports
  pinned dimensions: it rejects only `low > high` and rejects `max_step`
  declarations on `low == high` dims — the masking `action_box` already
  performs. A 14-D Box with pitch/roll pinned at (0, 0) constructs, and
  clipping into it pins commanded pitch/roll to exactly 0.

## Full consumer checklist (layout migration must touch all of these)

- `src/config.py`: EEF_DIM_LABELS (70), length validation via
  `len(EEF_DIM_LABELS)` (328-329 — self-updating), yaw indices (3, 8) →
  (3, 10) plus new pitch/roll index checks, `eef_low >= eef_high` (331),
  gripper max_step indices (4, 9) → (6, 13) at 618, observation_space
  docstring "10-D" (704).
- `src/embodiment.py`: `_step_eef` slices (1908, 1912, 1916, 1917),
  docstring "10-D" (1898), `_validate_eef_home` slices and state vector,
  `_DOCS_EEF_POS`.
- `src/kinematics.py`: solve/observe/capture/extraction per Changes 2.
- `tests/test_eef_embodiment.py`: shape (10,) at 121, eef_state tuples
  (131), action vectors, "expected a 10-D vector" match at 405,
  eef_low/eef_high overrides (5-tuples ×2 → 7-tuples ×2).
- `tests/test_config.py`: shapes at 252-253, 286, 342; max_step indices
  at 306; eef bounds validation tests.
- `tests/test_embodiment.py:1725`: eef gripper max_step indices (4, 9) →
  (6, 13).
- `tests/test_settle.py:307`: `np.full(10, 0.3)` action → 14 slots.
- `tests/test_eef_kinematics.py`: solve/observe shapes, yaw_axis test
  (220) superseded.
- `tests/test_agent_eef_integration.py`, `tests/test_embodiment_docs.py`:
  label-driven, mostly self-updating; eef_state fixtures widen.
- `README.md`: "10-D" at 468, 492-493, 713 and the eef_low/eef_high
  example lists.
- `tests/test_collision.py` constructs eef configs without dims — no
  change expected; verify by running.

## Retained invariants (restated at the point of modification)

- With pitch and roll at their default pinned bounds, every wire byte the
  driver sees is identical to today: `solve()` receives the same effective
  target family, the ramp/settle path is untouched, and the yaw slot's
  commanded semantics are unchanged.
- The per-step IK clamp, oscillation guard, resync, `_send` joint clamp,
  and EEF-home validation ordering (resolve → connect → validate → prompt
  → ramp) all carry over untouched.
- `_validate_eef_home` continues to pass for any home/start pose at
  default bounds: relative yaw/pitch/roll at arrival are 0 by construction
  and 0 is inside a pinned (0, 0) bound.
- Plan 0025's reconnect revalidation (`close()` clears
  `_eef_home_validated`) is untouched.

## Out of scope

- Pitch-coupled z-floor (the static Box cannot express coupled bounds;
  documented as an operator responsibility in the README warning).
- Any `rotation_repr` vocabulary change (framework-owned).
- EEF-space pose authoring.
- Exposing per-axis max_step for orientation slots.

## Tests (CI enforces 100% line+branch coverage)

New:
- `kinematics`: round-trip tests — for a grid of (yaw, pitch, roll) inside
  bounds, build `R_delta · reference` for a non-trivial reference, extract,
  and recover the commanded triple (the sign convention is locked by a
  test asserting positive pitch moves a gripper-down tool axis toward +x).
- `solve()` builds the expected 4×4 target for non-zero pitch/roll
  (recorded via the raw seam's `ik_calls`).
- `observe()` reports non-zero pitch/roll for a measured pose tipped out
  of the yaw family.
- Config: pinned axis accepted (low == high), `low > high` still rejected,
  pitch bounds at ±π/2 rejected, roll/yaw out of [-π, π] rejected.
- Embodiment: an action commanding pitch on default (pinned) bounds is
  clamped to 0 by `_send`/the action box (behavior-identity guard);
  `_validate_eef_home` failure message unchanged shape.

Sanctioned updates to pre-existing tests (layout migration, mechanical):
- `test_config.py`: EEF label list, eef tuple-length messages (10 → 14),
  yaw-index bound tests (indices 3, 8 → 3, 10), the `eef_low >= eef_high`
  rejection test becomes `low > high` rejected + equality accepted.
- `test_eef_embodiment.py`: action-space shape (10,) → (14,), all action
  vectors and `eef_state` expectations gain `0.0, 0.0` orientation slots
  per arm, `eef_low`/`eef_high` overrides become 7-tuples ×2, gripper
  action indices 4/9 → 6/13.
- `test_eef_kinematics.py`: `solve`/`observe` shapes; the `yaw_axis`
  fallback tests (e.g. the `kin.yaw_axis == 1` assertion) are superseded
  by the relative-extraction scheme and are replaced by the round-trip
  tests above.
- `test_agent_eef_integration.py` / `test_embodiment_docs.py`: label and
  docs-string updates.
No other pre-existing assertion may change; in particular every joint-mode
test is untouched.
