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
2. **Run-header warning** naming any pinned orientation dims in `eef_pos`
   mode, through the existing `contribute_guardrails` warnings channel.

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
existing boolean protocol with zero core changes. The core wizard already
carries stored option values through re-runs and skips colliding keys, so
the setting survives `inspect-robots setup` — the exact failure mode #140
reports for hand-edited tuples. Operators who need custom orientation ranges
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
never touched. In `joints` mode the flag is accepted and inert (the eef
tuples are validated but unused there today; same behavior). The widening
happens in `YamConfig.__post_init__` via `object.__setattr__` **before**
bounds validation, so `-E eef_orientation=true` works identically for CLI,
config.ini, and direct Python construction, and the widened values still
pass through the yaw/roll/pitch range checks.

## Changes

1. `config.py`
   - New frozen field `eef_orientation: bool = False` with a docstring-level
     contract: "open default-pinned pitch/roll axes to conservative ranges".
   - Module constants for the widened ranges (e.g.
     `_EEF_ORIENTATION_PITCH = (-0.6, 0.6)`,
     `_EEF_ORIENTATION_ROLL = (-np.pi / 2, np.pi / 2)`) next to
     `_EEF_PITCH_INDICES` / `_EEF_ROLL_INDICES`.
   - `__post_init__`: before the existing eef bounds validation, if
     `eef_orientation` and `control_interface` is anything (inert-but-
     applied in joints mode is fine since the tuples are unused there;
     simpler than special-casing), rebuild `eef_low`/`eef_high` tuples with
     the widened entries for exactly-(0.0, 0.0) pitch/roll dims and
     `object.__setattr__` them. Validation then runs on the widened values.
   - New helper (module-level or `YamConfig` method) returning the pinned
     orientation dim labels for a config:
     `pinned_orientation_labels(...) -> tuple[str, ...]` — dims among
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
   - `contribute_guardrails`: when `control_interface == "eef_pos"` and
     `pinned_orientation_labels()` is non-empty, include one extra warning
     in the returned `GuardrailContribution` (this method is the run-header
     warnings channel; both eef-mode return paths — guardrail-disabled and
     joints-mode-only — must carry it):
     `"eef_pos: action dims <labels> are pinned (low == high) and not
     commandable; set eef_orientation=true or widen eef_low/eef_high"`.
     No warning when nothing is pinned (e.g. after `eef_orientation=true`).

3. `README.md`
   - EEF section: document `eef_orientation` as the supported way to open
     pitch/roll, keep (and cross-reference) the existing z-floor WARNING.
   - Wizard/options documentation (wherever the other option slots are
     listed): add the new option.
   - Style rule applies: no em dashes in prose, bold only for `**term:**`
     lead-ins.

4. `CHANGELOG.md`: one entry under Unreleased — new `eef_orientation`
   config field + wizard option, and the pinned-axis run warning (#140).

## Tests (100% coverage, mypy strict, ruff)

- `tests/test_config.py`
  - `eef_orientation=True` with default bounds: pitch/roll widened to the
    constants, yaw/position/gripper untouched.
  - `eef_orientation=True` with explicit tuned bounds carrying `0,0`
    pitch/roll (the fleet case): widened.
  - Nonzero-pinned pitch (e.g. `0.1,0.1`) and already-widened dims: left
    alone.
  - `eef_orientation=False` (default): bounds byte-identical to today.
  - String round-trip: `from_kwargs(eef_orientation="true")` behaves like
    the wizard-written config value (matching the existing boolean option
    args' parsing path).
  - `pinned_orientation_labels`: default config → the four pitch/roll
    labels; fully-open config → empty; pinned yaw included.
- `tests/test_embodiment.py` / `tests/test_eef_embodiment.py`
  - `contribute_guardrails` in eef_pos mode with default bounds: warning
    present, names all four labels, on both the guardrail-disabled and
    guardrail-enabled paths.
  - With `eef_orientation=True`: no pinned-axis warning.
  - Joints mode: no pinned-axis warning regardless of eef tuples.
- `tests/test_i2rt.py` (established cross-repo wizard seam)
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
  responsibility); the option label and README carry the reminder.
- Per-axis orientation `max_step` (0026 follow-up, unchanged).
- Core `doctor` surfacing of `zero_width` dim names (core repo issue if
  wanted).
