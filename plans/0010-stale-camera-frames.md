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

Median staleness above the floor, three cameras, three control rates, with the
model below alongside each measured range:

| `control_hz` | period | default | `4/hz - 1/fps` | `buffersize=1` | `1/hz - 1/fps` | grabber |
|---|---|---|---|---|---|---|
| 5 Hz | 200 ms | 747-792 ms | 767 ms | 145-191 ms | 167 ms | 3-27 ms |
| 10 Hz | 100 ms | 356-381 ms | 367 ms | 50-71 ms | 67 ms | 4-28 ms |
| 20 Hz | 50 ms | 153-177 ms | 167 ms | 3-26 ms | 17 ms | 15-17 ms |

One closed form predicts all six queue readings, every one inside the measured
per-camera spread:

```
staleness ~ N / control_hz - 1 / fps        (N = buffer count: 4 default, 1 capped)
```

The mechanism it encodes: a saturated ring hands back the frame captured one
frame interval after your **previous** read, N reads ago. The grabber is not in
the same family. Its frame is at most one frame interval old whatever the
control rate, and the measured 3-28 ms is a half-interval mean plus the spread
of three unsynchronized capture phases.

That the model holds across a 4x span of control rates, for two different buffer
counts, is the evidence that this is the mechanism rather than a coincidence
fitted to one operating point. It is also what makes the decision below
predictive rather than local: the residual after the one-line cap is
`1/control_hz - 1/fps`, which only vanishes as the control rate approaches the
frame rate.

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

**1. The bug is far worse than the issue estimated, and it grows as `control_hz`
falls.** The issue predicted "roughly 130 ms" from four queued frames times a
33 ms frame interval. That arithmetic does not hold for a saturated ring. Each
`read()` frees exactly one slot and the driver refills it one frame interval
later, so the FIFO advances one frame per **consumer** period and staleness is
`N/control_hz - 1/fps` against a `BUFFERSIZE` readback of 4. Lowering
`control_hz` makes it worse, the opposite of the intuition the issue encoded. At
5 Hz the observation is most of a second old.

**2. The one-line cap helps a lot and is not enough.** `BUFFERSIZE=1` takes `N`
from 4 to 1 but leaves the mechanism: the single buffer is refilled one frame
interval after each read and then holds that frame until the next one, so the
residual is `1/control_hz - 1/fps`. At the default 10 Hz that is 50-71 ms, most
of a control period, and at 5 Hz it is 145-191 ms. It shrinks only as the
control rate approaches 30 Hz, and `control_hz` is a knob operators turn down.

**3. A drain thread makes staleness independent of the control rate.** The
grabber's frame is at most one frame interval old, flat across a 4x span of
control rates. Against the cap alone it saves ~40 ms at 10 Hz and ~150 ms at
5 Hz. This is the fix.

**Where the grabber does not win.** The model puts the crossover at
`control_hz ~ fps/1.5`: above it the cap's residual is already inside one frame
interval and the thread buys nothing. At 20 Hz the two are indistinguishable
(cap 3-26 ms, grabber 15-17 ms). The decision rests on the 10 Hz default and
below, which is where this rig runs, not on a uniform advantage. Stating this
matters after two instrument failures: the case is real but bounded.

**Two consequences of the model that are stronger than the 5 Hz row.** The
residual is not `1/control_hz - 1/fps` but `(interval between reads) - 1/fps`,
and the control period is only a lower bound on that interval.

- **#62 makes the one-liner strictly worse.** With settle merged, the interval
  between reads is `max(1/hz, settle + observe)`, and `settle_timeout_s`
  defaults to 1.0 s. A rig that settles slowly gets a cap-only residual
  approaching a second per step, exactly on the steps where the arm moved
  furthest and the frame matters most.
- **The cap does not bound the first frame of an episode at all.** Across the
  `reset()` gap (the homing ramp plus `operator.wait_ready()`, which waits on a
  human pressing Enter) `BUFFERSIZE=1` still holds the frame captured one frame
  interval after the *last* read. The first observation of every episode is as
  old as the operator took to walk back from the e-stop. The drain thread bounds
  it at one frame interval; nothing else here does.

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
lifetime (threads and captures) and a closure has no way to end it.

```python
class _OpenCVCameraReader:
    """Builtin V4L2 reader: one drain thread per camera, newest frame wins."""

    def __call__(self, cfg: YamConfig) -> ImageMap: ...
    def close(self) -> None: ...
```

`_opencv_camera_reader(cfg)` keeps its signature and returns an instance, so the
`CameraReader` alias (`Callable[[YamConfig], ImageMap]`) is still satisfied,
`reset()`'s `callable()` fail-fast still passes, the class stays private, and
`__all__` / `test_api_snapshot.py` are untouched. Captures open lazily on the
first call, so construction stays inert.

Per camera: `BUFFERSIZE=1` at open, then a daemon thread looping `cap.read()` and
publishing into a one-deep slot under a lock. `__call__` takes the newest
published frame per camera and converts it.

**Why publish frames, not `grab()`/`retrieve()`.** The issue suggested a thread
calling `grab()` with the consumer calling `retrieve()`. Those share the
capture's internal buffer state, so splitting them across threads races on it.
The thread owns the capture exclusively; the consumer only ever touches the slot.
Nothing else may touch the capture either, including property reads.

**The slot holds a copy, not the frame OpenCV returned.** `retrieve` may hand
back an array viewing the capture's own buffer, so the thread's next read would
tear a frame the consumer is still converting, and a read after `release()` would
be a use-after-free. The thread publishes `frame.copy()`.

**Conversion stays on the consumer thread.** Publishing raw and converting in
`__call__` keeps `cvtColor`/`resize` running at `control_hz` rather than at
30 fps, which is the cheaper side of the trade (a copy is a memcpy; the
conversion is not), and keeps the resize bound to the *call-time* `cfg` exactly
as today. The slot costs 3 x 640x480x3, about 2.8 MB.

### Failing loudly, which the naive version does not

Today the reader raises `RuntimeError` naming camera and device after 10 failed
attempts. A drain thread deletes that property by construction: if the thread
dies on an exception, or the device stops yielding frames, the slot simply stops
advancing and `__call__` serves the same frame forever. That is the bug this
issue exists to fix, in a worse form, because nothing reports it.

So the slot carries a publish timestamp and the thread carries an error latch:

- Any exception in the drain loop is recorded and ends that thread.
- `__call__` re-raises a latched exception, naming camera and device.
- `__call__` raises when the newest frame is older than `_MAX_FRAME_AGE_S`
  (0.5 s, the same budget as today's 10 x 50 ms retry loop).

Raise rather than warn-and-serve: a silently stale observation is what the issue
is about, and `EmbodimentFault` handling already exists for a camera that dies
mid-episode.

**First frame.** `reset()` observes immediately after the devices open, so a
thread that has not yet published would make `reset()` raise *after the arms have
homed* — the same failure mode the warm-up note below exists to prevent, through
a different door. The successful warm-up read seeds the slot synchronously before
the thread starts, so a `__call__` immediately after open always has a frame. If
the warm-up never yields one, `__call__` waits out the same 10 x 50 ms budget
before raising, preserving today's behavior.

### Lifecycle

Threads must stop, or `close()` then `reset()` leaks a set per cycle and the old
ones keep reading. Worse than today: a live daemon thread is a GC root, so a
missed `close()` leaks the devices *and* three threads spinning at 30 fps for the
process lifetime, where today dropping the embodiment lets refcounting release
the captures.

`close()` stops every thread, **joins before releasing**, and on a join timeout
logs and skips the release. With `READ_TIMEOUT_MSEC=1000` a thread can sit inside
`cap.read()` for a second; releasing the capture underneath it segfaults a process
that is holding torque-enabled arms. Leaking a device is the better failure.
Stops are set on all cameras first, then joined, so teardown costs one timeout
rather than three. `close()` is idempotent.

`YAMEmbodiment.close()` gains the release, guarded so a custom `camera_reader`
without one is unaffected:

```python
close_cameras = getattr(self._camera_reader, "close", None)
if callable(close_cameras):
    try:
        close_cameras()
    except Exception:
        logger.exception("camera release failed")
```

Placed with the unconditional `kinematics.clear()` at the top of `close()`, not
in the driver `finally`: `close()` early-returns on `if self._driver is None`
before reaching that `try`, so a release placed there is unreachable on the
never-reset path and on a second `close()`. The exception guard matters because
that placement puts camera release *ahead* of the park ramp and the `finally`
that guarantees `driver.close()`; an escaping error would strand the arms
torque-on with handles held.

**Correcting a claim from the previous draft.** It said captures are never
released today. They are: `caps` lives in a closure owned by one embodiment
instance, so dropping the instance releases them. The real defect is narrower:
queues are reused across `close()` / `reset()` on the *same* instance, so the
first observation of a later episode comes from a queue filled before the homing
ramp. That survives, and the drain thread plus release fixes it.

### Testability

The issue's step 3, and the part that is more than the fix itself. Today the
whole reader body is `# pragma: no cover - real cameras`, covering two different
things: talking to hardware, and how a capture is configured and a frame becomes
an observation. The second is ordinary logic the 100% gate should be seeing. A
regression dropping the `BUFFERSIZE` line, or publishing a stale slot, would
today be caught by nothing.

Injected, following the pattern `YAMEmbodiment.__init__` already uses for the
driver, clock, and operator: the `cv2` module, `sleep_fn`, and `clock`. The cv2
default is a `None` sentinel resolved at first open, never a factory called in
`__init__`, so construction stays inert and the `import-hygiene` CI job keeps
passing. Only the sentinel resolution keeps a pragma.

Behavior held constant: lazy `cv2` import, lazy device open, the 10-attempt
warm-up **including its fall-through when no frame arrives**, BGR->RGB, resize to
`cam_width` x `cam_height`, `uint8`, and the `RuntimeError` messages naming both
camera and device.

**Revision note (a behavior change nearly smuggled in).** An earlier draft had
the open helper raise when the warm-up exhausted its attempts. Current code falls
through and lets the per-read retry try again, so a camera slow to start on a
cold USB bus works today and would have started failing inside `reset()`, after
the arms were homed. Not in scope for a staleness fix.

**Two latent bugs, in scope because the rewrite has to decide them.**

1. The read loop keeps the last `frame` from `ok, frame = cap.read()` and decides
   on `if frame is None`, so a capture returning `ok=False` with a non-`None`
   frame on every attempt passes the guard and a failed read reaches the policy.
   The drain loop tracks `ok`.
2. If the second camera fails to open, `caps` already holds the first, so
   `if not caps:` is false on every later call and the remaining devices are
   never opened. `_observe` validates the shape of whatever keys arrive but never
   the key set, so the rollout proceeds with one camera, silently. Captures are
   built into a local dict and published only on full success; a partial failure
   releases what it opened.

**Typing.** The injected module is `Any`, which loses cv2's shipped stubs for
these calls. Mitigated by a fake defining exactly the constants used, so a
misspelled constant raises `AttributeError` in tests rather than passing. A
`Protocol` for the module surface would mean structurally matching a module
object plus the nested `cv2.VideoWriter.fourcc`, which is more machinery than the
risk warrants. mypy strict's `warn_return_any` is the one trap; the conversion
ends in an explicitly typed `npt.NDArray[np.uint8]` local.

### Tests

Against a fake cv2 module and fake captures, in `tests/test_embodiment.py`:

1. The full recorded `set` sequence: `BUFFERSIZE=1` first, then `FOURCC`, frame
   size, and both timeouts. One assertion covering presence, value, and order for
   every property, guarding the fix and the settings it must not disturb.
2. Warm-up reads happen after every `set`. `BUFFERSIZE` must precede the first
   grab, since OpenCV's V4L2 backend rejects it once streaming has started.
3. Devices open once and are reused, not reopened per call.
4. A partial open failure releases what it opened and leaves nothing cached, so
   the next call retries every device.
5. The drain loop publishes; `__call__` returns the newest frame, not an older.
6. The drain loop skips `ok=False`, including `ok=False` with a non-`None` frame.
7. A drain-loop exception is latched and re-raised by `__call__`, naming camera
   and device.
8. `__call__` raises when the newest frame is older than `_MAX_FRAME_AGE_S`.
9. The warm-up seeds the slot, so the first `__call__` succeeds with no thread
   iteration; and when the warm-up yields nothing, `__call__` waits the retry
   budget before raising.
10. Conversion: BGR->RGB, resized to the configured dimensions, `uint8`.
11. `close()` stops threads, releases captures, and is idempotent; `close()`
    before any call is a no-op.
12. `close()` skips the release when a join times out.
13. `YAMEmbodiment.close()` calls the reader's `close`, tolerates a
    `camera_reader` without one, and does not let a failing `close` stop the park
    ramp or the driver release.

**No test starts a thread.** The drain loop is a method run *synchronously on the
test thread*, with a fake capture whose Nth `read()` sets the stop event. That
covers the loop-taken arc, the loop-exit arc, both `ok` arcs, and the exception
arc, deterministically and with no sleeping. Only tests 11 and 12 involve real
threads, and 12 uses a fake capture blocking on a test-controlled `Event` so the
join-timeout branch is covered without a pragma. `sleep_fn` injection keeps the
warm-up and retry tests off the wall clock.

### Not in scope

- **Sibling plugins.** `franka`, `so101`, `widowx`, `unitree-g1`, `agibot-a2`
  have their own readers with the same gap. Prove it here, port after, as #62
  did for settle.
- **A `cam_buffersize` config knob.** The drain thread makes the property nearly
  irrelevant, so it would be a knob for a number that no longer matters.

## Interaction with #62

#62 merged as `9d184c0` while this was in review, so this branch rebases onto it.
The two touch the observation path in different places: #62 restructured `step()`
and added `_settle`, this changes the reader and adds a camera release to
`close()`. The rebase was clean.

They compose into the property neither has alone. #62 makes `step()` wait for the
arm to reach its commanded pose before observing, which fixes **when** the frame
is asked for. This fixes **which** frame comes back. Without both, a settled arm
is still photographed mid-motion, from up to 380 ms earlier.

The measurement also hands #62's timing analysis one number: reads cost 0.45 ms
for all three cameras in every configuration measured, so `_pace()`'s
shrinking-settle-window concern is dominated by policy inference, not camera
reads. Posted on #62 before it merged.

One note for a future reader of `plans/0009`: `_pace()` runs before `_observe()`
and stamps `_t_last` at its own end, so camera cost sits outside the declared
period. That stays a rounding error only because reads never block in any
configuration measured here, including the one this PR ships. A reader that
blocked would make the embodiment run below its declared `control_hz` while
`EmbodimentInfo` kept claiming otherwise.

### The visual acceptance, now inherited

The issue's step 4 asks for a Rerun timeline check that #62 and #63 compose:
scrub and confirm the frame at each step matches the joint state at that step.
It was deferred to whichever landed second, which is this. It stays a manual,
operator-run check rather than an acceptance gate: the rig is headless, so it
needs `--rerun-connect` over an SSH reverse tunnel, and "the frames look right
when scrubbed" is not a pass/fail a CI job or a script can assert. The
`--via-reader` run in Acceptance is the automatable evidence for the same
property; the Rerun check is documented in the PR for whoever next sits at the
rig.

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
