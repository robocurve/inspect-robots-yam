# 0023: observe_parked — clear-view final frames for grading

Closes #126. Companion to inspect-robots plan 0076 (#399): the framework
calls a duck-typed `embodiment.observe_parked()` right before grading a
scored trial; the embodiment parks itself and returns one fresh
`Observation`, or `None` to decline. Duck-typed on both sides, so this
plugin change ships independently of the core release: an old core simply
never calls the method.

## Changes

### `config.py`

New frozen field `park_before_grade: bool = True` beside the other opt-in
runtime flags, with a docstring comment: parking before grading gives the
grader an unobstructed final view, but it moves the arms, so a rig running
tasks whose success state is the gripper holding an object must set it
false (the framework then grades the last step's frames as before).

### `embodiment.py`

New public method on `YAMEmbodiment`:

```python
def observe_parked(self) -> Observation | None:
```

Behavior, reusing existing primitives only:

1. Return `None` when `self._cfg.park_before_grade` is false, when
   `self._driver is None` (never connected), or when `self._init_pose is
   None` (connected but never reset — no park target exists).
2. Park target: identical rule to `close()` — the configured
   `rest_pose`, else the reset-captured `_init_pose`. Restate the
   invariant here: `close()` keeps its own ramp (a later close still parks
   from wherever the arms are; ramping twice to the same target is a
   no-op ramp).
3. Attended status line around the ramp (`"parking for grading: ramping
   arms clear"`), closed in a `finally` exactly like the `close()` ramp's
   status handling, and skipped when `unattended`.
4. `self._ramp_to(target)`, then `return self._observe(None)`
   (instruction `None`: the observation is for the grader, not a policy).
5. No exception handling: the framework's call site catches everything and
   degrades to last-step frames with a stderr note. A fault mid-ramp
   surfaces there, and the later `close()` still runs its own park in its
   own `try/finally`.

Docstring states the contract (framework calls it pre-grading; motion
side effect; decline conditions) and cross-references the config flag.

## Tests (100% coverage, injected seams, no hardware)

- Flag false → `None`, no driver call, no ramp.
- Never connected → `None`; connected-but-never-reset → `None`.
- Happy path with injected driver and camera reader: ramp target equals
  the configured `rest_pose` (and equals the captured init pose when
  `rest_pose=None`), `_observe`'s observation is returned, status line
  emitted and closed when attended, absent when unattended.
- A ramp fault propagates (no swallowing) and the status line still
  closes.
- `from_kwargs` round-trips `park_before_grade` like the other bool flags.

## Docs

- README/config reference: document `park_before_grade` beside
  `report_joint_eff` and friends, including the hold-task caveat.

## Out of scope

- Core-side changes (inspect-robots plan 0076).
- Wizard prompts for the flag (config default plus docs suffice; the
  wizard passes unknown keys through like other booleans).
