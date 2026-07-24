# 0010 — Fresh frames: drain the V4L2 queue so observations are not stale

Closes robocurve/inspect-robots-yam#63.

The issue asked for a measurement before a fix, because the remedy branched on
what the driver does. The measurement was taken on the rig, and it took three
attempts to build an instrument that did not lie. The first two produced a
confident wrong answer. That history is kept in the "Revision note" blocks below,
because the wrong answer was wrong in a way worth not repeating.

## Problem

`_opencv_camera_reader` (`embodiment.py:233-283`) opens each camera and reads
with a plain `cap.read()`. It never caps `cv2.CAP_PROP_BUFFERSIZE` and never
drains the queue.

`cv2.VideoCapture` does not take a picture when you call `read()`. The driver
free-runs at 30 fps into a ring of buffers; `read()` dequeues one. The rollout
consumes at `control_hz` (10 Hz), so the ring saturates and every `read()`
returns a frame captured well before the moment it was asked for.

## Measurement

`scripts/measure_camera_latency.py`, run on the rig against the three RealSense
color nodes (`/dev/cam_context` D435, `/dev/cam_wrist_left` and
`/dev/cam_wrist_right` D405), cv2 5.0.0. It reads `/dev/video*` only: no CAN, no
torque, no motion, safe with the arms powered down.

Age comes from `CAP_PROP_POS_MSEC`, which on this stack carries the kernel's
capture timestamp for the buffer just dequeued. Its epoch is unrelated to
`perf_counter`, so the offset is recovered once per camera by draining the queue
in a tight loop, where the frame is as fresh as the pipeline allows, and taking
that residual as zero. Every number below is therefore staleness **above the
floor**: the part a fix can actually remove. It needs no scene change, perturbs
nothing, and resolves to microseconds.

Median staleness above the floor, three cameras, three control rates:

| `control_hz` | period | default | `buffersize=1` | grabber |
|---|---|---|---|---|
| 5 Hz | 200 ms | 747-792 ms | 145-191 ms | 3-27 ms |
| 10 Hz | 100 ms | 356-381 ms | 50-71 ms | 4-28 ms |
| 20 Hz | 50 ms | 153-177 ms | 3-26 ms | 15-17 ms |

Supporting readings, consistent across all three rates:

- `CAP_PROP_BUFFERSIZE` reads back as **4** by default and **1** when set, so
  the property is honored on this stack and the fix is not silently a no-op.
- Backlog (instant reads before the first blocking one) is **4-5** by default
  and **1** under the cap, corroborating the age signal by a different route.
- One `read()` costs **0.11-0.20 ms** in every configuration, and a sweep of all
  three cameras costs **0.43-0.47 ms** of a 100 ms control period.
- Every camera holds **30.0 fps** in all three configurations, so the three
  streams are not starving each other for USB bandwidth and the comparisons are
  between like regimes.

### What the numbers say

**1. The bug is far worse than the issue estimated, and it scales with
`control_hz`, not with fps.** The issue predicted "roughly 130 ms" from four
queued frames times a 33 ms frame interval. That arithmetic does not hold for a
saturated ring. Each `read()` frees exactly one slot and the driver refills it
immediately, so the FIFO advances one frame per **consumer** period: staleness
is `capacity x 1/control_hz`, measured at ~3.7x the period against a
`BUFFERSIZE` readback of 4. Lowering `control_hz` makes it worse, which is the
opposite of the intuition the issue encoded. At 5 Hz the observation is most of
a second old.

**2. The one-line cap helps a lot and is not enough.** `BUFFERSIZE=1` removes
the ring but not the mechanism: the single buffer is refilled right after each
read and then holds that frame until the next one, so staleness stays
proportional to the control period (~0.6-0.85x). At the default 10 Hz that is
50-71 ms, most of a control period.

**3. A drain thread makes staleness independent of the control rate.** The
grabber's frame is at most one frame interval old, 3-28 ms measured, flat across
a 4x span of control rates. Against the cap alone it saves ~40 ms at 10 Hz and
~150 ms at 5 Hz. This is the fix.

**Revision note (two wrong instruments).** The first harness measured queue
*depth* by timing reads, and multiplied depth by the frame interval. That is the
issue's own arithmetic and it is wrong, as above. The second measured age with a
UVC brightness step, timing the first frame that showed it. That instrument had
two defects that happened to point the same way: it sampled on the consumer
period, so every reading quantized to a multiple of it, and it took its baseline
from an extra `read()` outside the paced loop, which on a 1-deep buffer empties
it completely and makes the next read a capture-on-demand the rollout never
performs. Together those reported `BUFFERSIZE=1` at 32 ms, i.e. at the floor,
and the grabber at exactly one sampling bin at every rate. The conclusion drawn
was that the cap was sufficient and the grabber measurably worse. Both were
artifacts. The tell was an internal contradiction the plan asserted without
noticing: a 0.3 ms sweep cost means reads return instantly, which is
incompatible with a 32 ms age. The lesson is not "brightness stepping is bad",
it is that an instrument whose resolution equals the quantity being compared
cannot distinguish the candidates, and that any read outside the measured regime
perturbs the thing being measured.

**Rejected: doing nothing beyond the cap.** Tempting, since the cap is one line
and gets ~85% of the way at 20 Hz. But the residual scales with the control
period, and `control_hz` is a config knob operators are expected to turn. A fix
whose effectiveness silently degrades when a rig slows its control rate is the
kind of thing this issue exists to stop.

### What the numbers refute

Two of the issue's stated arguments *for* the threaded grabber do not survive,
and should not be used to justify it:

- **Inter-camera skew.** Reading all three cameras sequentially costs 0.43-0.47
  ms. Read ordering contributes essentially nothing. What skew remains is three
  free-running sensors with unsynchronized 30 Hz phases, which no threading
  fixes: they are not genlocked. The grabber's per-camera spread (3-28 ms) is
  exactly that residual.
- **Control-budget pressure.** The same 0.45 ms is the `o` term in #62's
  `o + s <= 1/hz` budget, against a 100 ms period. There was never anything to
  reclaim, in any configuration.

The grabber is worth its cost for one reason only: staleness bounded by the
frame interval instead of the control period.

## Design

### Shape

A class rather than the current closure, because the fix introduces state with a
lifetime (threads and captures) and the closure has no way to end it.

```python
class _OpenCVCameraReader:
    """Builtin V4L2 reader: one drain thread per camera, newest frame wins."""

    def __call__(self, cfg: YamConfig) -> ImageMap: ...
    def close(self) -> None: ...
```

`_opencv_camera_reader(cfg)` keeps its signature and returns an instance, so
`CameraReader` (a callable protocol) is still satisfied and nothing upstream
changes. Captures open lazily on the first call, as today.

Per camera: `BUFFERSIZE=1` at open, then a daemon thread looping `cap.read()`
and publishing the frame into a slot under a lock. `__call__` takes the newest
published frame per camera and converts it. The cap is kept even though the
thread makes it nearly irrelevant: it bounds the worst case if a thread is
descheduled, and it costs one line.

**Why publish frames, not `grab()`/`retrieve()`.** The issue suggested a thread
calling `grab()` with the consumer calling `retrieve()`. Those two share the
capture's internal buffer state, so splitting them across threads races on it.
The thread owns the capture exclusively; the consumer only ever touches the slot.

### Lifecycle

The threads must stop, or a `close()` then `reset()` cycle leaks one set per
cycle and the old ones keep reading. `YAMEmbodiment.close()` gains a camera
release alongside the driver release, guarded so a custom `camera_reader` with
no `close` is unaffected:

```python
close = getattr(self._camera_reader, "close", None)
if callable(close):
    close()
```

In the `finally` that already guarantees driver release, so a fault mid-park
cannot strand threads.

**This fixes a second, unfiled bug.** Today captures are never released at all:
`caps` lives in the closure for the process lifetime, so after `close()` the
devices stay open, and the first observation after a subsequent `reset()` comes
from a queue filled before the homing ramp. That is the same staleness class
this issue is about, at a much larger magnitude. Worth stating in the PR.

### Testability

The issue's step 3, and the part that is more than the fix itself. Today the
whole reader body is `# pragma: no cover - real cameras`, which covers two
different things at once: talking to hardware, and how a capture is configured
and a frame becomes an observation. The second is ordinary logic that the repo's
100% gate should be seeing. A regression dropping the `BUFFERSIZE` line, or
publishing a stale slot, would today be caught by nothing.

The cv2 module becomes a constructor argument, defaulting to a lazily imported
real one, following the injection pattern `YAMEmbodiment.__init__` already uses
for the driver, clock, and operator. Tests pass a fake module and fake captures.
Only the default factory that does `import cv2` keeps a pragma.

Behavior held constant: lazy `cv2` import so the package still imports without
OpenCV, lazy device open on first read so construction stays inert, the existing
10-attempt warm-up **including its fall-through when the warm-up never yields a
frame** (see below), BGR→RGB, resize to `cam_width` x `cam_height`, `uint8`, and
the `RuntimeError` messages naming both camera and device path.

**Revision note (a behavior change nearly smuggled in).** An earlier draft had
the open helper raise when the warm-up exhausted its 10 attempts. Current code
falls through and lets the per-read retry try again, so a camera slow to start on
a cold USB bus works today and would have started failing, inside `reset()`,
after the arms were already homed. Not in scope for a staleness fix.

**Latent bug that is in scope,** because the extraction has to decide it either
way. The read loop keeps the last `frame` from `ok, frame = cap.read()` and then
decides on `if frame is None`, so a capture returning `ok=False` with a non-`None`
frame on every attempt passes the guard and a failed read reaches the policy as
an observation. The drain thread tracks `ok` instead.

**Typing.** The injected module is typed `Any`, which loses cv2's shipped stubs
for these calls. Mitigated by a fake in the tests that defines exactly the
constants used, so a misspelled constant raises `AttributeError` there rather
than passing silently. The alternative, a `Protocol` for the module surface,
means structurally matching a module object plus the nested
`cv2.VideoWriter.fourcc`, which is more machinery than the risk warrants. Note:
returning a value derived from an `Any` call trips mypy strict's
`no-any-return`; the conversion ends in an explicitly typed
`npt.NDArray[np.uint8]` local.

### Tests

Against a fake cv2 module and fake captures, in `tests/test_embodiment.py`:

1. The full recorded `set` sequence matches expectation: `BUFFERSIZE=1` present,
   first, before `FOURCC`, the frame size, and both timeouts. One assertion
   covering presence, value, and order for every property, so it is a regression
   guard for the fix and for the settings the fix must not disturb. Order is
   load-bearing: OpenCV's V4L2 backend rejects `BUFFERSIZE` once streaming has
   started.
2. The warm-up reads happen after every `set`, for the same reason.
3. Devices open once and are reused across calls, not reopened per read.
4. The drain thread publishes, and `__call__` returns the newest published frame
   rather than an older one.
5. A frame arriving between two `__call__`s replaces the slot.
6. `__call__` raises `RuntimeError` naming camera and device when a camera has
   never published.
7. The drain loop skips frames where `ok` is false, including `ok=False` with a
   non-`None` frame, the latent bug above.
8. Conversion: BGR→RGB, resized to the configured dimensions, `uint8`.
9. `close()` stops every thread and releases every capture, and is idempotent.
10. `close()` before any call is a no-op, since captures open lazily.
11. `YAMEmbodiment.close()` calls the reader's `close`, and tolerates a custom
    `camera_reader` without one.

Threads make timing-dependent tests a hazard. The drain loop's structure is
tested by driving one iteration directly rather than by sleeping and hoping, and
the fake capture uses an `Event` the test sets, so no test depends on a race
resolving a particular way.

### Not in scope

- **Sibling plugins.** `franka`, `so101`, `widowx`, `unitree-g1`, `agibot-a2`
  have their own readers with the same gap. Prove it here, port after, exactly
  as #62 is doing for settle.
- **A `cam_buffersize` config knob.** The drain thread makes the property
  nearly irrelevant, so an escape hatch for it would be a knob for a number that
  no longer matters.
- **The `--rerun-connect` visual acceptance in the issue's step 4.** It checks
  #62 and #63 composing, which needs #62 merged. It belongs to whichever lands
  second.

## Interaction with #62

Both touch the observation path, in different functions: #62 is in `step()` and
`_observe`, this is in the reader. No conflict expected.

The measurement hands #62 one number it should have: reads cost 0.45 ms for all
three cameras in every configuration, so `_pace()`'s shrinking-settle-window
concern is dominated by policy inference, not camera reads. Posted on #62.

Worth noting for the pair: `_pace()` runs before `_observe()` and stamps
`_t_last` at its own end, so camera cost sits outside the declared period. It
stays a rounding error here only because reads never block in any configuration
measured. A future reader that blocks would make the embodiment run below its
declared `control_hz` while `EmbodimentInfo` kept claiming otherwise. Not this
PR's to fix; worth a line in #62's timing analysis.

## Acceptance

1. `ruff check .`, `ruff format --check .`, `mypy`, `pytest --cov` at 100%.
2. A `--via-reader` mode in the harness that measures staleness through the
   package's own `_OpenCVCameraReader` instead of captures the script opened
   itself. Without it the script only proves things about configurations it
   constructed: the three rows would stay green if the fix were reverted. The
   class makes this straightforward, since the script can hold the instance and
   call `close()`. Run on the rig; the reader must land on the `grabber` row.
3. Re-run the harness after any OpenCV or kernel upgrade. "The driver honors
   `BUFFERSIZE`" and "`POS_MSEC` carries a capture timestamp" are facts about
   this stack, not guarantees. The unit tests pin the configuration so CI catches
   a regression without a rig, but they cannot notice the driver changing.
4. Docs: README note on observation freshness, and the pragma inventory in
   `src/inspect_robots_yam/CLAUDE.md` updated for the seams this moves.
