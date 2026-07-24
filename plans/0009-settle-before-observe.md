# 0009 — Settle before observe: make the observation reflect the commanded pose

Closes robocurve/inspect-robots-yam#62.

Revised across two adversarial critique rounds. "Revision note" blocks record
choices that were wrong and why, because several were wrong in ways worth not
repeating.

## Problem

`YAMEmbodiment.step()` commands, paces, then observes (`embodiment.py:475-480`):

```python
if self._cfg.control_interface == "joints":
    self._send(cmd)
self._pace()
self._emit_status()
obs = self._observe(self._instruction)
```

`_send()` calls `driver.command_joint_pos()`, which sets a position target and
returns (`embodiment.py:707-712`). `BimanualDriver` exposes no blocking command
(`embodiment.py:128-142`). The observation is captured one control period after
the target is set, whatever the arm is doing then. Nothing checks arrival.

For a VLA running closed loop at `control_hz` this is defensible: the next
observation lands 100 ms later and the policy was trained on that cadence.

It breaks for chunked policies. The `agent` (LLM) policy interpolates one tool
call into up to 100 actions (`_tools.py:80`) at 1% of joint range per step
(`_tools.py:468`), and `DefaultController` calls `policy.act()` only when its
buffer empties. The LLM sees one frame per tool call: the final interpolated
step's observation, captured before the arm has necessarily arrived.

The window is not fixed either. `_pace()` computes `elapsed` from the end of the
previous `_pace()` (`embodiment.py:737`), spanning `_emit_status`, `_observe`,
`_poll_end`, the approver, the log sink, **and the policy's `act()` call**. On
the first step after an LLM inference, `_pace()` sleeps zero and that step gets
no settle window at all.

## What this does and does not guarantee

**Guarantee: the observation reflects the last pose the embodiment commanded.**

Not "the pose the policy asked for". In `eef_pos` mode the sent vector routinely
diverges from the request: osc hold re-sends `previous` (`kinematics.py:181-185`),
non-finite IK returns `previous` (`:206-207`), the `ik_step_joint_limit` rate
clamp truncates (`:209-214`), an osc trip holds (`:227-230`), and `_send` clamps
to `joint_low/high` regardless. Settling then succeeds immediately while the arm
sits far from the requested target. Correct behavior for this feature, and a
real limit on what it buys. The README says so in those words.

**It does not guarantee the camera image postdates the motion.** #63 covers
stale V4L2 frames: `cap.read()` can return a frame the driver captured earlier
and queued, so a settled arm can still be photographed mid-motion. #62 fixes
when we ask for the frame; #63 fixes which frame comes back. Both are needed for
the end-to-end property.

j1 and j2 have physical lower hard stops at 0 (`embodiment.py:66-70`) while the
default `joint_low` is `-π` (`config.py:42`). A policy can command below the
stop, `_send` clamps only to config limits, and the arm sits against the stop
forever. Handled by the budget below, not by pretending it cannot happen.

## Design

Poll `get_joint_pos()` after commanding until every **arm** joint is within
tolerance of the vector just sent, or the attempt is abandoned.

### Config

| Field | Default | Meaning |
|-------|---------|---------|
| `settle_tolerance` | `None` | Radians. `None` disables settling. |
| `settle_timeout_s` | `1.0` | Upper bound on one step's wait. |
| `settle_timeout_budget` | `20` | Timeouts per trial before settling self-disables. |

Validation mirrors `config.py:223-226`: `settle_tolerance`, when not `None`,
finite and `> 0`; `settle_timeout_s` finite and `> 0`; `settle_timeout_budget` a
non-bool `int` and `>= 1`.

Poll spacing is a module constant `_SETTLE_POLL_S = 0.01`, consumed through the
injected `sleep_fn`.

**Revision note (poll cost).** Each `get_joint_pos()` is two CAN round trips
(`embodiment.py:169-170`). A 10 ms poll implies up to 200 extra reads per step,
about 2000/s at `control_hz=10` against today's 20/s. The loop will be read-bound
in practice, so `_SETTLE_POLL_S` is a floor, not the actual spacing. Real bus
load needs a rig measurement.

### The loop

Read and check **before** sleeping, so an already-converged step costs zero
sleeps and adds nothing to the `o + s <= 1/hz` budget below. This is the common
case: `EchoDriver` in tests, and every osc-hold step on hardware.

```
start = clock(); polls = 0
loop:
    read joint positions
    if all arm joints within tolerance: settled
    polls += 1
    if polls >= max_polls: timed out          # bound A
    if clock() - start >= settle_timeout_s: timed out   # bound B
    sleep(_SETTLE_POLL_S)
```

**The enabled/disabled check lives in exactly one place, at the top of
`_settle`**, never duplicated at the two call sites. Duplicating it makes the
reset-site "disabled" arc unreachable once settle state clears at reset entry,
which is a partial branch and fails the coverage gate.

`max_polls = max(1, ceil(settle_timeout_s / _SETTLE_POLL_S))`. Bound A protects
against a frozen or non-monotonic clock, on hardware and in the frozen-clock
test fixtures alike. Bound B is the real-time guarantee.

**Revision note (bounds must not be coincident).** The first revision specified
both bounds without noticing they fire on the same iteration whenever
`sleep_fn` advances the clock by exactly `_SETTLE_POLL_S`: after `k` polls,
`elapsed == k * _SETTLE_POLL_S`, so A and B trip together at `k = 100`. With
`branch = true` and `fail_under = 100` (`pyproject.toml:115-120`) that is either
an unreachable second branch, which fails the build, or a compound condition in
which bound B is never the sole exit and is silently untested. The fixtures below
are deliberately built so each bound can be the sole exit.

### Choosing a tolerance: this is the sharp edge

`hold_check.py` already measured this rig: the steady-state control offset right
after a command is **~0.012-0.015 rad on YAM joint 3** (`hold_check.py:79-89`),
and `DEFAULT_SETTLE_RAD = 0.05` (`hold_check.py:39`) is the accept threshold it
ships. A tolerance at or below the rig's own steady-state offset can never be
met, so every step burns the full `settle_timeout_s`.

Every documented example uses `0.05`, framed as "must exceed the settle figure
`inspect-robots-yam-holdcheck` reports for your rig, with margin for gravity
loading", not as a value to copy blindly.

**Settling presumes a position-holding servo, and the default mode is not one.**
`zero_gravity_mode` defaults to `True` (`config.py:163`), a gravity-compensated
compliant mode that may drift rather than hold, which is exactly why
`hold_check.py` exists and why `README.md:316-318` tells operators to test both
modes and fall back to `-E zero_gravity_mode=false` when compliant drifts. The
holdcheck figure must be taken **in the mode the trial will run**, and
`zero_gravity_mode=false` is the expected pairing for settling. Settling on a
drifting compliant rig will exhaust the budget on every trial.

**Revision note.** The first draft used `0.02` in its operator example, below the
measured offset of at least one joint, and never connected `hold_check` to the
mode that script exists to discriminate. That combination would have shipped a
documented configuration guaranteed to stall on a default rig.

### The budget, and why not a consecutive-timeout breaker

After `settle_timeout_budget` timeouts **in a trial**, settling disables itself
for the remainder and warns once with the offending joint index and residual.

**Revision note.** The previous revision used a consecutive-timeout breaker.
It cannot trip in `eef_pos` mode: an osc trip sets `_hold_counter` to
`osc_hold_steps` (default 10, `config.py:147`), during which `solve()` returns
`previous` (`kinematics.py:181-185`), the arm is already at that pose, and
settling succeeds instantly. A trial with unreachable targets alternates real
motion (timeout) with holds (instant success), resetting the streak forever. The
breaker would never fire in precisely the scenario it existed for.

A total budget has no streak to reset, so it is immune to that. It also bounds
the quantity actually at stake: worst-case wasted wall clock per trial is
`settle_timeout_budget * settle_timeout_s`, 20 s at the defaults. A sliding
window (the `osc_reversals`/`osc_window` shape, `config.py:145-146`) would bound
a rate instead, which is the right invariant for oscillation but not for "do not
waste the operator's afternoon".

Timeouts remain non-fatal. Slow convergence is a fact about hardware, not an
eval failure.

### Why not raise `EmbodimentFault` instead

Considered, and rejected. Per the framework contract, `EmbodimentFault` halts
the **entire eval**, not just the trial. Budget exhaustion has at least two
causes that are not hardware faults: a tolerance below the rig's steady-state
offset, and a compliant `zero_gravity_mode` rig. Killing a multi-scene benchmark
because an operator picked a tight tolerance is disproportionate. Settling is an
observation-quality feature, not a safety gate; clamping and the approver chain
remain untouched either way.

The cost of that choice is real and is mitigated, not waved away: once the budget
trips, later steps in the trial return pre-settle observations while earlier ones
did not, and `scorer.py:169` judges `record.steps[-1]`. So a scorer can score an
un-converged final pose in a trial that otherwise looks healthy.

Mitigation is an unambiguous, persistent marker:

- Settling disabled by config: **no settle keys at all** in `info`.
- Settling disabled by budget: `{"settle_disabled": True, ...}` on **every**
  subsequent step.

A scorer can therefore segment a trial, and the two states can never be
confused. The tripping step is itself un-settled and carries `settled: False`,
so the marker starting on the *following* step loses nothing.

**For yam's primary workflow the marker is close to unreadable, and the warning
is the real signal.** Success here comes from the operator's y/N at
`embodiment.py:486-490`, and a human cannot read `info`. Combined with the
persistence gap below, that means the practical notification is the warning
emitted when the budget trips. It therefore names the scene id, so an operator
reviewing a multi-scene run knows which trial degraded, with the joint index and
residual appended to otherwise stable message text (Python's default filter
dedupes per location, and a varying message would print once per trial across a
100-scene eval).

### Reporting

**Revision note.** The first draft reported the count in the terminal
`StepResult.info`. That path is nearly unreachable: `info` is only built when
`not unattended and _poll_end()` (`embodiment.py:483-491`), the README prescribes
`unattended=True` for agent runs (`README.md:213`), and horizon exhaustion
truncates in `rollout` with no terminated `StepResult`.

Every `StepResult` therefore carries, when settling is configured:

```python
{"settled": bool, "settle_residual": float, "settle_timeouts": int}
```

plus `"settle_disabled": True` once the budget trips, merged into the terminal
dict on the terminating path.

**Where these numbers can actually be read.** `rollout` keeps `result` on every
`StepRecord` and offers it to sinks, but no shipped sink persists `info`:
`JsonLogSink` ignores `log_step`, `RerunSink` builds its payload from
images/state/action/reward/termination and drops `info`, and `EvalLog` persists
only `SceneResult` aggregates with no `StepRecord`s at all. So these values are
reachable from a **scorer** (`scorer.py:200` reads `s.result.info[...]` across
`record.steps`) or a custom in-process sink, and are not in the JSON log an
operator opens after a run.

**Revision note.** The previous revision claimed per-step `info` "is already
logged" and rested the every-step-vs-final-only decision on it. That was wrong
about the persistence path. The operator-visible signals are the budget warning
and the terminal `info`; extracting per-step residuals needs a scorer or sink.

### Placement

Settle goes immediately after the command, before `_pace()`, so settle time is
absorbed by the control period instead of added to it.

`_t_last` bookkeeping is safe: it is unconditionally re-based to `self._clock()`
after every pace (`embodiment.py:739`), so an overrunning settle yields
`sleep(0)` and re-bases, with no accumulating debt and no catch-up burst. With
`o` = post-pace overhead and `s` = settle time, step period is `max(1/hz, o + s)`.

**Revision note.** The first draft claimed an enabled settle "costs nothing extra
whenever the arm keeps up". The correct predicate is `o + s <= 1/hz`. Three V4L2
reads plus resize can consume most of a 100 ms budget alone, so `o` is not
negligible.

### `reset()` has the same defect

`_ramp_to` sends its last waypoint, sleeps one period, and returns
(`embodiment.py:555-556`). `reset()` then immediately reads the driver and calls
`capture_yaw_reference(measured)` (`:445-449`), pinning the whole trial's yaw
zero to a possibly mid-motion pose, before returning the first observation the
policy ever sees (`:458`). `wait_ready()` does not help: it runs *after* the
capture and is skipped entirely when unattended.

Settling runs at the **`reset()` call site**, inside the existing
`try/finally` that owns the homing status line (`:434-438`), before
`self._status(None)` closes it, so the operator sees a `settling` status rather
than up to `settle_timeout_s` of silence while standing at the e-stop. That
status emission is guarded by `if not self._cfg.unattended`, like every other
status call, or `test_unattended_runs_emit_no_status`
(`tests/test_embodiment.py:818-825`) breaks.

**Settle state clears at `reset()` entry, before the ramp**, not alongside
`self.num_steps = 0` at `:456`.

**Revision note.** Clearing with `num_steps` is the natural place and is wrong:
`:456` runs *after* the ramp block, so a trial that exhausted its budget leaves
`_settle_disabled` set when the next trial's reset settle consults it. That
settle would skip, `capture_yaw_reference` would pin the yaw zero to a
mid-motion pose again, and the reset fix would be silently dead for every
remaining trial of the eval. A test that only asserts "reset clears the
counters" does not catch it, because the bug is ordering.

**Not inside `_ramp_to`.** That helper is also called from `close()` (`:524`),
where a settle would delay teardown by up to `settle_timeout_s` inside a
`finally` whose purpose is to release handles promptly, and where waiting is
pointless because torque drops immediately after. `close()` does not settle.

The stand-clear gate (`:427-431`) is already confirmed before the ramp, so no
motion is unconsented; the wait is bounded by the same timeout, so reset cannot
hang; `_init_pose` is captured before the ramp (`:419-422`) and is unaffected.

Preflight is unaffected: `preflight.py:36` constructs the embodiment and reads
only `.info`, and the settle fields touch none of the spaces or `control_hz`. A
green preflight implies nothing about settle configuration.

### Both control interfaces

Settle against the vector actually sent, post-clamp. `_send()` returns the
clamped command; `_step_eef()` sends internally and currently returns `None`
(`embodiment.py:654`), so it changes to return the sent vector. One call site
(`:466`). mypy strict will require the sent vector definitely-assigned across
the interface split.

In `joints_are_delta` mode this is also a small latent fix: `:473` builds the
absolute target from the *measured* base, which today can lag mid-motion and
compound across steps. Settling makes that base a converged pose.

### Grippers are excluded

A gripper closing on an object never reaches its commanded position; including
slots 6 and 13 would time out on every grasp. The mask is built from
`packing.ARM_DOF` and `packing.ARM_WIDTH` as `_denorm_grippers` does
(`embodiment.py:718, 730`), never from literals.

The exclusion is also what makes the comparison sound: `get_joint_pos()` returns
driver-native gripper units while the sent vector is normalized, so the two are
commensurable only on the arm slots. A comment says so, to stop a later reader
"fixing" it by including the grippers.

## Files

| File | Change |
|------|--------|
| `src/inspect_robots_yam/config.py` | Three fields plus validation |
| `src/inspect_robots_yam/embodiment.py` | `_settle()`, calls in `step()` and `reset()`, `_step_eef` returns sent vector, counters, per-step `info`, docstring caveats |
| `tests/conftest.py` | New: shared clock fixtures and the settle driver |
| `tests/test_settle.py` | New: settle behavior, including the eef target |
| `tests/test_config.py` | Defaults and validation rejections |
| `README.md` | Config entries, tolerance and mode guidance, the guarantee's limits |
| `CLAUDE.md` | Timing note, since step duration is a contract |

## Test fixtures

`tests/` has no `conftest.py` today, and `test_embodiment.py` and
`test_eef_embodiment.py` have independent build helpers that both inject
`clock=lambda: 0.0`. The three clock behaviors go in a new shared `conftest.py`
rather than being duplicated:

| Fixture | Clock behavior | Makes this the sole exit |
|---|---|---|
| frozen | constant `0.0` (today's) | bound A, the iteration cap |
| sleep-advancing | `sleep_fn` adds its argument | neither alone; used for `_pace` arithmetic |
| read-advancing | `get_joint_pos()` advances the clock by `r` | bound B, elapsed time |

The read-advancing fixture is physically justified, not contrived: each
`get_joint_pos()` is two CAN round trips (`embodiment.py:169-170`), so real
per-poll elapsed time exceeds `_SETTLE_POLL_S`.

**`r` must be strictly greater than `_SETTLE_POLL_S`; the fixture uses
`r = 0.05`.** `start` is stamped before the first read, so after `k` iterations
`elapsed == k * r`. Bound B fires at `k = ceil(settle_timeout_s / r)` and bound A
at `k = max_polls`. With `r = 0.05` and the defaults that is 20 against 100. Pick
`r = _SETTLE_POLL_S` and both fire at `k = 100` again, reinstating exactly the
coincidence bound A and B were separated to avoid, this time hidden inside a
fixture that looks like it solved the problem.

**Tests observe which bound fired by counting driver reads.** The fake driver
records `get_joint_pos()` calls; test 5 asserts exactly `max_polls` reads, test 6
asserts strictly fewer. Without that counter "sole exit" is not an assertable
property.

**The existing `_build` helpers in `test_embodiment.py` and
`test_eef_embodiment.py` keep their hardcoded `clock=lambda: 0.0`.** The conftest
fixtures are additive, for new tests only. Refactoring the old helpers onto a
read-advancing clock looks like tidy-up and would break roughly 40 tests: an
`_observe` read (`embodiment.py:746`) would inflate `_pace`'s `elapsed` (`:737`),
so `test_pacing_sleeps_to_control_rate` (`:322-326`, asserts `sleeps[-1] == 0.1`)
fails immediately.

## Tests

1. Default config: no settle; sleep sequence byte-identical to today's.
2. Enabled and converging: polls until within tolerance, then observes.
3. Already converged: zero sleeps (check precedes sleep).
4. Per-step `info` reports `settled`, residual, and running count.
5. Timeout under the frozen fixture: bound A is the sole exit.
6. Timeout under the read-advancing fixture: bound B is the sole exit.
7. Budget exhaustion disables settling and emits `settle_disabled` on every
   later step; instant successes in between do **not** replenish the budget.
8. Gripper divergence alone does not block settling.
9. `eef_pos` settles against the clamped joint vector `_step_eef` sent.
10. `reset()` settles before capturing the yaw reference.
11. `close()` does not settle.
12. `reset()` clears counters, the disabled flag, and re-arms the warning **at
    entry**: a trial that exhausts the budget is followed by a `reset()` that
    still settles before capturing the yaw reference. Asserts the ordering, not
    just the final state.
13. Settling enabled with `unattended=True` emits no status lines, exercising
    the taken arc of the `settling` status guard.
14. Config validation rejections for all three fields.

The budget warning is asserted with `pytest.warns` (precedent at
`tests/test_embodiment.py:773`); `pytest` sets no `filterwarnings = "error"`
(`pyproject.toml:106-113`), so it will not disturb unrelated tests. There is one
warning mechanism, fired only when the budget trips, to avoid a warned/tripped
state matrix with unreachable corners against `fail_under = 100`.

## Known limitations

**Operator time displays understate real time when settling is enabled.**
`_emit_status` computes `elapsed = self.num_steps / hz` (`embodiment.py:696`) and
`_horizon_secs()` divides by `control_hz`. With settling on, `control_hz` is a
floor on step duration, so the running counter and the `Max 120s.` limit
(`:452-454`) become lower bounds. Both gain docstring caveats here; wall-clock
rework is #64.

**The budget is absolute, not proportional to trial length.** 20 timeouts across
a 1200-step trial is 1.7% of steps, yet it ends the guarantee for the remaining
800. The consecutive breaker was too slow to trip in eef mode; a fixed budget is
too quick to trip on long horizons. The embodiment does know the horizon
(`_bound_max_steps`, `embodiment.py:394`), so scaling the default against it is
available if this bites. Left absolute here because it bounds the quantity the
operator actually feels, wasted wall clock, and because a proportional default
is harder to reason about at a glance. Raise `settle_timeout_budget` for long
horizons.

**Tolerance is scalar.** Gravity loading differs per joint and `hold_check` found
j3 worst, so a scalar forces sizing for the worst joint. `_FLOAT_TUPLE_FIELDS`
(`config.py:116-126`) already supports comma-separated 14-vectors if this proves
too coarse.

## Alternatives considered

**Settle only on the chunk's final action.** Materially better if per-step
settling proves expensive, and cheaper to plumb than first assumed: `Action.meta`
is set policy-side and `rollout` already honors `meta["request_stop"]`, so the
agent plugin could tag its last interpolant with **no core contract change**.

Which is better is unknown without a rig measurement. 1% of joint range is 1% of
±π, about **0.063 rad per step**, 4-5x the rig's measured steady-state offset, so
each interpolant is a real motion. Settling every step could turn a smooth 10 s
sweep into 100 converge-and-wait moves and make the agent tool's advertised
playout duration false. Whether it does depends on how fast the servo closes
0.063 rad against a 100 ms period, which nobody here has measured.

This PR ships every-step because it is self-contained in one repo and cannot
regress default behavior. Final-only tagging is filed as
robocurve/inspect-robots#167.

**Blocking driver command.** Not available; `BimanualDriver` has none and i2rt is
out of our control.

**Fixed sleep instead of a convergence poll.** Either wastes time when the arm is
already there or under-waits when it is not.

## Out of scope

#63 (stale V4L2 frames) defeats the end-to-end image guarantee just as hard, but
its fix sits in `# pragma: no cover` camera code this suite cannot verify, and
the viable remedy branches on a measurement: if `CAP_PROP_BUFFERSIZE` is honored
it is one line, and if not it needs a per-camera grabber thread. That thread
would also move camera reads off the control thread, shrinking the `o` term
above, which is why the measurement is worth having before this plan's timing
story is treated as final.

Generalizing to the other five embodiment plugins, which share the same ordering
(`franka:270-271`, `so101:181-183`, `widowx:222-223`, `unitree-g1`,
`agibot-a2`), is follow-up once this has run on hardware.
