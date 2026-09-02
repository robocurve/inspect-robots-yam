# 0030: Wizard-settable motor_temp_limit (NUMBER_SLOTS)

Issue: #150 part 2 (with plan 0029). Branch: `feat/thermal-wizard-slot`.
Blocked on the inspect-robots release that ships core plan 0081
(NUMBER_SLOTS, inspect-robots#432) — expected 0.58. **The implementer must
re-verify the shipped `NumberSlot` API against the released core before
writing anything; the snippet below transcribes the core plan and may have
drifted.**

## Design

Declare beside `OPTION_SLOTS`:

```python
NUMBER_SLOTS: ClassVar[tuple[NumberSlot, ...]] = (
    NumberSlot(
        arg="motor_temp_limit",
        label="Motor temperature soft limit in degrees C "
        "(motor_temp_limit; none disables the thermal guardrail)",
        default=70,
        minimum=1,
        allow_none=True,
    ),
)
```

`NumberSlot` imports from `inspect_robots.conformance` like `OptionSlot`.
The wizard suggestion (70) deliberately diverges from the YamConfig default
(`None` = off): a fresh wizard-driven setup should arm the guardrail, while
the library default stays non-breaking. `allow_none=True` lets an operator
disable it at the prompt. Extend the OPTION_SLOTS rationale comment
(`embodiment.py:1169-1181`, currently "yes/no questions") to cover number
slots and this divergence.

Dependency floor: `inspect-robots>=0.58` in pyproject.toml (update the
`:23` comment's rationale too) plus `uv lock`, mirroring the plan-0028 bump.
The module-level `from inspect_robots.conformance import ...` makes the old
floor an ImportError that would drop the whole plugin at entry-point load, so
the bump is required, not courtesy.

## Tests

- A NUMBER_SLOTS pin beside the OPTION_SLOTS pin (`tests/test_i2rt.py:355-372`):
  exact arg set `{"motor_temp_limit"}`, every number arg is a real YamConfig
  field, and the values pinned.
- The divergence pin, matching the existing idiom
  (`test_auto_start_wizard_default_diverges_from_config_default`,
  `tests/test_i2rt.py:374-389`): slot default is `70` AND
  `YamConfig().motor_temp_limit is None`.
- The existing OPTION_SLOTS pinned set stays untouched.

## Docs/meta

- README: the thermal guardrail paragraph gains "the setup wizard offers
  motor_temp_limit (suggested 70; answer none to leave it off)".
- CHANGELOG `## Unreleased`: `### Added` (wizard slot) and `### Changed`
  (floor raised to 0.58) prose entries referencing (#150).
- src/inspect_robots_yam/CLAUDE.md: embodiment.py row mentions NUMBER_SLOTS.

## Out of scope

- No wizard slot for `motor_temp_warn_margin` (one number to decide at setup
  is enough; the margin default is fine).
