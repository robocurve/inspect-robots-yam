# Consume core `bind_task`: countdown horizon without config duplication

Date: 2026-07-14
Status: draft (headed for subagent critique loop)

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

      name: str
      max_steps: int
  ```

  (`typing.Protocol` is already imported; no core import is added. When the
  package later bumps its core floor past the release that ships
  `TaskEnvelope`, this alias can be replaced by the real import.)

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

### `config.py`

- `max_steps_hint` stays but is deprecated: when a non-None value is
  configured, emit a `DeprecationWarning` from `__post_init__` (where the
  field is already validated) saying the framework now supplies the horizon
  via `bind_task` and the hint is only a fallback for runs where the hook
  never fires. Deleting the field outright would `TypeError` on existing
  `config.ini` files (`from_kwargs` rejects unknown keys); actual removal
  waits for a later release.
- The field docstring is rewritten to say "Deprecated fallback" instead of
  presenting the knob as the way to get a countdown.

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
- Configuring `max_steps_hint` warns `DeprecationWarning` (and existing
  tests that set it are updated to expect it).
- `TaskEnvelopeLike` accepts the real shape: a frozen local dataclass with
  `name`/`max_steps` satisfies it (structural check).

## Out of scope

- Removing `max_steps_hint` (a later, breaking release).
- Bumping the `inspect-robots` floor: not needed — the hook is duck-typed
  and this package must keep working on cores that never call it.
