# 0027: Wizard path to open EEF pitch/roll + pinned-axis run warning

Issue: #140. Branch: `feat/eef-orientation-wizard`.

## Goal

Since plan 0026, `eef_pos` advertises 7 dims per arm but ships pitch and roll
pinned at `(0, 0)`, and the only way to open them is hand-editing two
14-tuples in config.ini — which the setup wizard neither offers nor surfaces,
and which nothing at run time flags. This plan adds the two missing pieces
from #140 (asks 2 and 3), entirely inside this repo:

1. **`eef_orientation` boolean option** (new `YamConfig` field + wizard
   `OptionSlot`): answering yes widens the default-pinned pitch/roll bounds
   to conservative real ranges.
2. **Run-header warnings** through the existing `contribute_guardrails`
   warnings channel, all in `eef_pos` mode: one when `eef_orientation` is
   on (it rewrites `0,0` pins, so the config text no longer matches the
   effective bounds), one naming any pinned orientation dims, and one per
   the z-floor hazard: an arm with open pitch/roll (by flag or by
   hand-tuned tuples) whose z low sits at or below the fingertips-down
   default.

Ask 1 of #140 (non-degenerate default bounds) is deliberately **not** taken:
changing `_EEF_ARM_LOW/_EEF_ARM_HIGH` would silently change motion limits on
every rig that re-runs setup or omits explicit bounds, and 0026's
byte-identical-default contract stays.

## Why a boolean, not a bounds interview

The core wizard's plugin surface is exactly `DEVICE_SLOTS` /
`OPTION_SLOTS(arg, label, default: bool)` / `RUNTIME_REQUIREMENTS`
(inspect-robots `conformance.py`); there is no free-text or numeric slot
kind. A numeric bounds interview would require a core wizard extension and a
cross-repo version dance. #140 itself scopes the wizard ask to
"enable pitch/roll? [y/N] → widen to defaults", which maps 1:1 onto the
existing boolean protocol with zero core changes. The core wizard carries
stored option values through re-runs and skips colliding keys, so the
setting survives `inspect-robots setup` like any other option (hand-edited
unmanaged args survive re-runs too via `_render_config`'s carry-through;
#140 is about discoverability, not loss). Operators who need custom
orientation ranges
still set `eef_low`/`eef_high` explicitly; the flag is the discoverable
80% path, not a replacement for the tuples.

## Semantics (the load-bearing decision)

`YamConfig.eef_orientation: bool = False`. When `True`, after `eef_low` /
`eef_high` are resolved (defaults or operator-supplied), every pitch/roll
dim whose bounds are **exactly `(0.0, 0.0)`** is widened to:

- pitch (indices 4, 11): `(-0.6, 0.6)` rad — well inside the `(-π/2, π/2)`
  validation bound that keeps the ZYX extraction non-singular (~34°, enough
  for the pitched-forward reach the rig docs value at 10–20 cm).
- roll (indices 5, 12): `(-π/2, π/2)` rad — half the `[-π, π]` validation
  range, enough for angled grasps without inverted-tool poses.

Widening applies to the *effective* bounds, not only the defaults, because
the fleet's real configs (tuned x/y/z boxes with pitch/roll still `0,0`,
per #140) carry explicit `eef_low`/`eef_high`; a flag that only swapped the
defaults would do nothing on exactly the rigs that motivated the issue.

Deliberate pins are respected: a pitch/roll dim pinned at a **nonzero**
value, or already widened, is left untouched. Yaw and position dims are
never touched.

**Re-pin footgun, documented explicitly:** before this change, writing
`0,0` in the tuples was the one documented way to pin a tilt axis. With
`eef_orientation=true` persisted in config.ini, a hand-written `0,0`
pitch/roll pin is silently re-widened (and the pinned-axis warning stays
quiet because nothing is pinned). The semantics stay (the fleet case
justifies them), but the mismatch between config text and effective bounds
must be visible at run time, not only in docs: a dedicated run-header
warning fires whenever the flag is on in eef mode (see Changes 2), and the
README section and the field's `#` comment (repo convention: fields take
comments, not docstrings) state the rule: with `eef_orientation=true`, a
`0,0` pitch/roll pin is widened; to re-pin, set `eef_orientation=false` or
pin at a nonzero epsilon.

In `joints` mode the flag is accepted and inert-but-applied: the tuples
are still rewritten, just unused (the eef
tuples are validated but unused there today; same behavior). The widening
happens in `YamConfig.__post_init__` via `object.__setattr__`, inserted
**after the eef tuple length checks** (config.py:336-338) and **before the
finiteness/range checks**, so a malformed short tuple still fails with the
clean length `ValueError` (never an `IndexError` from the widening indexing
4/5/11/12), `-E eef_orientation=true` works identically for CLI, config.ini,
and direct Python construction, and the widened values still pass through
the yaw/roll/pitch range checks.

**Type strictness:** no string path exists in practice (core `_parse_value`
converts `true`/`false` to Python bools before `from_kwargs` sees them),
and a truthiness check would make `-E eef_orientation=off` or `="false"`
(truthy strings) silently widen motion bounds. `eef_orientation` therefore
joins the existing bool-guard loop in `YamConfig.from_kwargs`
(config.py:289-296, currently guarding `collision_guardrail` et al.): any
non-bool raises `ValueError`. Because this flag widens motion bounds,
`__post_init__` additionally raises `ValueError` on a non-bool before the
widening (unlike the four precedent flags, which guard only in
`from_kwargs`): direct Python construction with a truthy non-bool must not
widen either.

## Changes

1. `config.py`
   - New frozen field `eef_orientation: bool = False`, **appended after the
     last existing field** (`park_before_grade`; the repo convention keeps
     positional construction stable), with a `#` comment stating the
     contract (repo convention: fields take comments, not docstrings):
     "open default-pinned pitch/roll axes to conservative ranges", including
     the re-pin note (see Semantics). Added to the `from_kwargs` bool-guard
     loop and guarded again in `__post_init__` (see Semantics: non-bool
     values raise).
   - Module constants for the widened ranges (e.g.
     `_EEF_ORIENTATION_PITCH = (-0.6, 0.6)`,
     `_EEF_ORIENTATION_ROLL = (-np.pi / 2, np.pi / 2)`) next to
     `_EEF_PITCH_INDICES` / `_EEF_ROLL_INDICES`.
   - `__post_init__`: after the eef length checks (config.py:336-338)
     and before the finiteness/range checks (see Semantics), if
     `eef_orientation` and `control_interface` is anything (inert-but-
     applied in joints mode is fine since the tuples are unused there;
     simpler than special-casing), rebuild `eef_low`/`eef_high` tuples with
     the widened entries for exactly-(0.0, 0.0) pitch/roll dims and
     `object.__setattr__` them. Validation then runs on the widened values.
   - New `YamConfig` method (a method, not a module-level function: no
     `__all__`/api-snapshot decision needed) returning the pinned
     orientation dim labels:
     `pinned_orientation_labels(self) -> tuple[str, ...]` — dims among
     yaw/pitch/roll indices where `low == high`, named via
     `EEF_DIM_LABELS`. Used by the warning and directly testable.

2. `embodiment.py`
   - `OPTION_SLOTS`: append
     `OptionSlot(arg="eef_orientation", label="Open EEF pitch/roll tilt axes
     (eef_orientation; eef_pos rigs only, raise the eef_low z floor after)",
     default=False)`. Default off in both wizard and runtime: opening tilt
     axes invalidates the fingertips-down z-floor assumption (0026), so it
     must be an explicit operator choice. Extend the OPTION_SLOTS comment
     block with this rationale.
   - `contribute_guardrails`: three new conditional warnings in the
     returned `GuardrailContribution`, all only when
     `control_interface == "eef_pos"` (this method is the run-header
     warnings channel; both eef-mode return paths — guardrail-disabled and
     joints-mode-only — must carry them):
     - when `eef_orientation` is `True` (unconditional on whether widening
       changed anything: `__post_init__` has already rewritten the tuples
       and the pre-image is unrecoverable):
       `"eef_orientation=true: pitch/roll bounds written as 0,0 are widened
       to +/-0.6 / +/-pi/2; set eef_orientation=false to re-pin"`. This is
       the run-time surface for the re-pin footgun (see Semantics).
     - when `pinned_orientation_labels()` is non-empty:
       `"eef_pos: action dims <labels> are pinned (low == high) and not
       commandable; widen eef_low/eef_high (eef_orientation=true opens
       only zero-pinned pitch/roll)"`. The remedy wording must stay honest
       for pinned yaw and nonzero pins, which the flag never opens.
       No warning when nothing is pinned (e.g. after `eef_orientation=true`).
     - z-floor warning, gated on the hazard **per arm** (not on the
       `eef_orientation` flag, so hand-tuned tuples trigger it too): fire
       when some arm has an open (not pinned) pitch/roll dim AND **that
       same arm's** z low is `<=` the shipped default (left: dims 4/5 vs
       index 2; right: dims 11/12 vs index 9; compare against the
       public `DEFAULT_EEF_LOW[2]`/`[9]` constants, not a `0.03` literal or
       the module-private `_EEF_ARM_LOW` — a lowered floor
       is strictly worse). An arm with open tilt and a raised z floor, or
       default z with pinned tilt, does not fire:
       `"eef pitch/roll are open but eef_low z is at or below the
       fingertips-down default; knuckles or the wrist camera can reach the
       table first; raise the z floor"` (semicolons, matching the existing
       warning strings; tests freeze the exact text). This is the live
       counterpart to the README's z-floor WARNING; raising that arm's z
       silences it.
     **Warning order is part of the spec** (tests assert exact tuples):
     the existing path warning (guardrail-disabled or joints-mode-only)
     first, then the eef_orientation widening notice, then the pinned-axis
     warning, then the z-floor warning.
     Reach caveat, acknowledged here and in the CHANGELOG entry: these
     warnings surface only on CLI-wired runs (`_build_and_announce_guardrails`);
     `--disable-guardrails` and direct `rollout()`/`eval()` API runs never
     call `contribute_guardrails`, so they see neither warning.

3. `README.md`
   - EEF section: document `eef_orientation` as the supported way to open
     pitch/roll, keep (and cross-reference) the existing z-floor WARNING,
     and state the re-pin rule (with `eef_orientation=true` a `0,0`
     pitch/roll pin is widened; re-pin by unsetting the flag or pinning at
     a nonzero epsilon). Also note the z-floor warning covers open axes
     only: a deliberate nonzero tilt pin has the same knuckles-first
     hazard and still requires the operator to raise z themselves.
   - Wizard/options documentation (wherever the other option slots are
     listed): add the new option.
   - Style rule applies: no em dashes in prose, bold only for `**term:**`
     lead-ins.

4. `CHANGELOG.md`: one entry under Unreleased → Added (Keep-a-Changelog
   format) — new `eef_orientation` config field + wizard option, and the
   three eef-mode run warnings (widening notice, pinned-axis, z-floor;
   CLI runs only) (#140).

5. `src/inspect_robots_yam/CLAUDE.md`: the module-map rows for `config.py`
   and `embodiment.py` enumerate notable flags and warnings; mention
   `eef_orientation` and the new eef-mode contribution warnings there.

## Tests (100% coverage, mypy strict, ruff)

- `tests/test_config.py`
  - `eef_orientation=True` with default bounds: pitch/roll widened to the
    constants, yaw/position/gripper untouched.
  - `eef_orientation=True` with explicit tuned bounds carrying `0,0`
    pitch/roll (the fleet case): widened.
  - Nonzero-pinned pitch (e.g. `0.1,0.1`) and already-widened dims: left
    alone.
  - `eef_orientation=False` (default): bounds byte-identical to today.
  - Bool guard: `from_kwargs(eef_orientation=True)` widens;
    `from_kwargs(eef_orientation="true")` (or any non-bool, e.g. `None`
    from `-E eef_orientation=none`) raises `ValueError`. No string
    acceptance: core `_parse_value` delivers real bools, and truthy strings
    like `"false"` or `"off"` must never widen motion bounds. Direct
    construction `YamConfig(eef_orientation="true")` also raises (the
    `__post_init__` guard).
  - `pinned_orientation_labels`: default config → the four pitch/roll
    labels; fully-open config → empty; pinned yaw included.
- `tests/test_collision.py` (where the existing `contribute_guardrails`
  tests live)
  - **Existing test to update, not weaken:**
    `test_contribution_ladder_skips_non_absolute_joint_modes`
    (tests/test_collision.py:545-561) is parametrized over `eef` **and**
    `delta-joints`; only the `eef` param gains the pinned-axis warning
    (default config: no flag, no open tilt, so exactly the path warning +
    pinned-axis warning, in the specified order), while `delta-joints`
    (joints mode) keeps its exact one-element assertion. Split the test or
    parametrize the expected warnings per id.
  - New: eef_pos + default bounds → pinned-axis warning present, names all
    four labels, on both the guardrail-disabled and guardrail-enabled
    (joints-mode-only) paths, in the specified order.
  - New: `eef_orientation=True` in eef mode → widening notice present
    (exact text, in order), no pinned-axis warning; joints mode or flag
    off → no notice.
  - New: z-floor warning per-arm coupling: flag on + default z lows →
    present; absent once both arms' z lows are raised; mixed config where
    the only arm with open tilt has raised z (other arm: pinned tilt,
    default z) → **absent** (no false positive); one arm with open tilt
    and default/lowered z → present.
  - New: z-floor warning also fires without the flag when pitch/roll are
    opened via explicit `eef_low`/`eef_high` tuples and that arm's z low
    is at or below the default (the hazard gate, not the flag gate).
  - New: joints mode → none of the three new warnings regardless of eef
    tuples.
- `tests/test_i2rt.py` (established cross-repo wizard seam)
  - **Existing test to update:** the exact option-arg set assertion
    (tests/test_i2rt.py:366-370, currently
    `{"auto_start", "collision_guardrail", "report_joint_eff"}`) gains
    `"eef_orientation"`.
  - New slot appears in `OPTION_SLOTS`, its `arg` is a real `YamConfig`
    field, wizard default (False) matches the runtime default.
  - Round-trip through core `_setup._options_section` + `_render_config` +
    `load_defaults`: answering yes writes `eef_orientation = true` and a
    constructed `YamConfig` from those args has widened pitch/roll.

## Out of scope

- Changing the shipped pinned defaults (ask 1 of #140): rejected above.
- A numeric bounds interview in the wizard: needs a core slot-kind
  extension; file upstream if the boolean proves insufficient.
- Automatic z-floor adjustment when opening tilt axes: the coupling is
  rig-geometry-dependent (0026 already documents it as an operator
  responsibility); the option label, README, and the new z-floor run
  warning carry the reminder instead.
- Per-axis orientation `max_step` (0026 follow-up, unchanged).
- Core `doctor` already emits a generic `zero_width` warning
  (conformance.py:205); giving it dim names is a core repo issue if wanted.
