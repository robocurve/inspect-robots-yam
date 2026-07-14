# Consume core `bind_task`: countdown horizon without config duplication

Date: 2026-07-14
Status: approved (critique round 2 verdict OK; property-member protocol, FutureWarning,
warn-after-validate, close() clears the bound horizon, README + test-site
enumeration)

## Problem

The operator countdown only shows the episode horizon when `max_steps_hint`
is set in the embodiment config, a value the framework already owns
(robocurve/inspect-robots-yam#41). Core now ships the fix on its side
(robocurve/inspect-robots#80, plan 0013): `eval()` calls an optional
duck-typed `bind_task(envelope)` on the embodiment before the compatibility
check, where `envelope` is a frozen `TaskEnvelope` with `name` and
`max_steps`.

## Design

### `embodiment.py`

- A local structural protocol, so the module keeps importing and
  type-checking against every core this package supports
  (`inspect-robots>=0.8`, which predates `TaskEnvelope`):

  ```python
  @runtime_checkable
  class TaskEnvelopeLike(Protocol):
      """Structural mirror of ``inspect_robots.task.TaskEnvelope``."""

      @property
      def name(self) -> str:
          """The task's registry/display name."""
          ...

      @property
      def max_steps(self) -> int:
          """The rollout horizon the framework will enforce."""
          ...
  ```

  (Property docstrings are required — ruff's D102 applies to `src`, and the
  `BimanualDriver` precedent docstrings every member.)

  The members MUST be read-only properties, not plain attributes: a Protocol
  attribute member demands a settable variable under mypy strict, which
  statically rejects the frozen `TaskEnvelope` dataclass this protocol
  exists to mirror (the package ships `py.typed`, so a typed downstream
  caller passing `task.envelope` would get an arg-type error). Property
  members accept frozen dataclasses statically and still pass runtime
  `isinstance` under `@runtime_checkable` (both `Protocol` and
  `runtime_checkable` are already imported; the house style precedent is
  `BimanualDriver`). When the package later bumps its core floor past the
  release that ships `TaskEnvelope`, this alias can be replaced by the real
  import.

- `YAMEmbodiment.bind_task(envelope: TaskEnvelopeLike) -> None` stores the
  horizon: `self._bound_max_steps = int(envelope.max_steps)` (initialized to
  `None` in `__init__`). Nothing else: binding must stay hardware-free (it
  runs before `reset()` connects the driver).

- Horizon resolution becomes a tiny helper used by `_horizon_secs()`:
  the bound value when `bind_task` has fired, else the deprecated
  `cfg.max_steps_hint`, else `None`. `_horizon_secs()` keeps dividing by
  `cfg.control_hz` — our own rate is the truth because this embodiment is
  `SELF_PACED` and `_pace()` sleeps to it.

- The adapter contract from core plan 0013 applies: the hook is optional
  input. On direct `rollout()` calls or older cores it never fires and the
  behavior is exactly today's (hint if set, otherwise elapsed-only
  countdown). Re-binds (one per `eval()`) overwrite: latest envelope wins.
- `close()` also clears `_bound_max_steps`, unconditionally and as its FIRST
  statement — before the existing `self._driver is None` early return.
  Otherwise a bound-but-never-reset instance (e.g. `eval()` aborting at
  `assert_compatible`, which core runs after `bind_task`) would keep a stale
  horizon through close(). A closed instance later driven via direct
  `rollout()` must fall back, not display the previous task's horizon;
  close() stays idempotent.

### `config.py`

- `max_steps_hint` stays but is deprecated: when a non-None value is
  configured, emit a `FutureWarning` at the END of `__post_init__`, after
  ALL validations (an invalid config — bad hint, bad cameras, bad step
  limits — raises without ever warning). `FutureWarning`, not
  `DeprecationWarning`: the audience is
  operators running the `inspect-robots` console script, and Python's
  default filters hide `DeprecationWarning` raised from library code — they
  would never see it. Message: the framework now supplies the horizon via
  `bind_task`; the hint is only a fallback for runs where the hook never
  fires. Deleting the field outright would `TypeError` on existing
  `config.ini` files (`from_kwargs` rejects unknown keys); actual removal
  waits for a later release.
- The field docstring is rewritten to say "Deprecated fallback" instead of
  presenting the knob as the way to get a countdown.

### `README.md`

- The `max_steps_hint` entry in the `YamConfig` field listing is marked
  deprecated (fallback only), and the attended-flow section gains a line
  noting the countdown shows the real horizon with zero configuration under
  framework-driven runs.

### User-visible effect

With any core that ships the hook, `inspect-robots "instruction"` on yam
shows `t = 42s / 120s | any key ends the episode` and
`Running: ... Max 120s.` with zero configuration, and the number always
matches the real `--max-steps` / `config set max_steps` horizon.

## Testing

TDD; gates: yam CI's 100% coverage, mypy strict, ruff.

- `bind_task` stores the horizon and the status line shows
  `elapsed / total` (drive `step()` via the injected driver/camera/clock
  seams as existing status tests do; envelopes in tests are a tiny local
  dataclass — the protocol is structural).
- Precedence: bound envelope wins over a configured `max_steps_hint`; the
  hint still works when nothing was bound; neither → elapsed-only line.
- A second `bind_task` call overwrites the first (latest wins).
- `reset()`'s "Max Ns." line reflects the bound horizon.
- Configuring `max_steps_hint` warns `FutureWarning`. Four existing
  construction sites start warning and get `pytest.warns` at their own
  `YamConfig(...)` constructions (the `_build_with_status` helper's internal
  default construction never warns): the three in `test_embodiment.py`
  (`test_reset_announces_run_instructions`,
  `test_status_line_updates_once_per_second_with_horizon`,
  `test_unattended_runs_emit_no_status`) and the valid
  `YamConfig(max_steps_hint=1200)` construction in `test_config.py`. The
  `max_steps_hint=0` case there keeps raising `ValueError` without warning
  (validation precedes the warning) — assert that explicitly.
- After `close()`, the bound horizon is gone (fallback behavior returns).
- `TaskEnvelopeLike` accepts the real shape at runtime: a frozen local
  dataclass with `name`/`max_steps` passes `isinstance` (the
  `@runtime_checkable` structural check; tests are outside mypy's scope in
  this repo, so the static acceptance is enforced by the property-member
  form above rather than a test).

## Out of scope

- Removing `max_steps_hint` (a later, breaking release).
- Bumping the `inspect-robots` floor: not needed — the hook is duck-typed
  and this package must keep working on cores that never call it.
