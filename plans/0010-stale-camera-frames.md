# 0010 — Fresh frames: cap the V4L2 queue so observations are not stale

Closes robocurve/inspect-robots-yam#63.

The issue asked for a measurement before a fix, because the remedy branched on
what the driver does. The measurement was taken on the rig and it decided the
branch, corrected the issue's own estimate of the damage, and ruled out the
larger of the two candidate fixes. This plan records the numbers first, because
every design choice below follows from them.

## Problem

`_opencv_camera_reader` (`embodiment.py:233-283`) opens each camera and reads
with a plain `cap.read()`. It never caps `cv2.CAP_PROP_BUFFERSIZE`.

`cv2.VideoCapture` does not take a picture when you call `read()`. The driver
free-runs at 30 fps into a queue of buffers; `read()` dequeues. The rollout
consumes at `control_hz` (10 Hz), so the queue saturates and every `read()`
returns a frame captured well before the moment it was asked for.

## Measurement

`scripts/measure_camera_latency.py`, run on the rig against the three RealSense
color nodes (`/dev/cam_context` D435, `/dev/cam_wrist_left` and
`/dev/cam_wrist_right` D405), cv2 5.0.0, consumer 10 Hz, 5 trials per camera.
Full output is attached to the PR.

Two signals, because the obvious one is misleading alone.

**Backlog** is queue depth, timed: a queued frame returns in microseconds, an
empty queue blocks for a frame interval, so the count of instant reads before
the first blocking one is the depth. Measured 3-4 frames deep.

**Age** is what actually matters. A UVC brightness step is a scene change with a
known `t0`; the wall time of the first frame that shows it is the end-to-end
latency. It includes a fixed per-camera sensor delay for applying the control,
so the script also measures an **instrument floor** with a tight-loop consumer
that never lets a queue form. Staleness is the excess over that floor.

| Configuration | top_cam age | left/right age | worst excess over floor |
|---|---|---|---|
| floor (tight loop) | 66.4 ms | 33.3 ms | — |
| **default (today)** | **301.1 ms** | **301.3 ms** | **268.0 ms** |
| `BUFFERSIZE=1` | 100.7 ms | 32.4 ms | 34.3 ms |
| `BUFFERSIZE=1` + grabber thread | 100.7 ms | 100.6 ms | 67.3 ms |

### The control-rate sweep

Both of the interpretive claims below (that latency is set by the consumer
period, and that the grabber's number is an instrument artifact) are predictions
about how the numbers must move when `control_hz` changes. So the harness was
re-run at 5 Hz and 20 Hz to check them, rather than leaving them as arguments.

| `control_hz` | period | default | grabber | `BUFFERSIZE=1` (wrists / context) | floor (wrists / context) |
|---|---|---|---|---|---|
| 5 Hz | 200 ms | 601.3 ms | 200.8 ms | 31.5 / 62.5 ms | 33.5 / 66.7 ms |
| 10 Hz | 100 ms | 301.1 ms | 100.6 ms | 32.3 / 100.7 ms | 33.3 / 66.4 ms |
| 20 Hz | 50 ms | 151.0 ms | 50.7 ms | 32.2 / 65.2 ms | 33.3 / 66.8 ms |

Both predictions hold, exactly:

- **Default is `3 x period` at every rate** (601 / 301 / 151 against 600 / 300 /
  150). The mechanism is measured across a 4x span, not inferred from one point.
- **The grabber is `1 x period` at every rate** (200.8 / 100.6 / 50.7). A number
  that tracks the sampling bin and nothing else is measuring the sampler. This
  is what an instrument-limited reading looks like, and it is why the grabber's
  apparent deficit at 10 Hz is not evidence about the grabber.
- **`BUFFERSIZE=1` sits at the floor at 5 Hz and 20 Hz for every camera**,
  context camera included. The one number that did not (top_cam's 100.7 ms at
  10 Hz) is the same 100 ms bin artifact, and the 5 Hz and 20 Hz runs resolve it
  below the bin: 62.5 ms and 65.2 ms against a 66.7 ms floor.

That last row is what makes the decision safe rather than merely unrefuted. With
`BUFFERSIZE=1` there is no measurable staleness left above the floor, so there
is nothing for a grabber thread to remove. A fix cannot beat the floor.

Three findings, in order of how much they change the plan.

**1. The damage is roughly twice the issue's estimate, and it scales with
`control_hz`, not with fps.** The issue predicted "roughly 130 ms" from four
queued frames times a 33 ms frame interval. That arithmetic is wrong for a
saturated ring. Each `read()` frees exactly one slot and the driver refills it
immediately, so the FIFO advances one frame per **consumer** period: latency is
`capacity x 1/control_hz`, about 300 ms here. Lowering `control_hz` makes it
worse, which is the opposite of the intuition the issue encoded.

**2. `CAP_PROP_BUFFERSIZE=1` is honored on this stack.** Every camera lands on
the tight-loop floor once the sweep above resolves the 10 Hz bin artifact. That
is the first row of the issue's fix table, and it is a one-line change.

**3. The grabber thread has nothing left to remove, and its two supporting
arguments do not survive measurement.** Its 67 ms apparent deficit at 10 Hz is
an artifact, confirmed by the sweep: slot reads return instantly, so the sample
at `t0` always predates the control applying and the step is always caught on
the next one, pinning the reading to exactly one sampling bin at every rate. So
it is not evidence of a regression. But `BUFFERSIZE=1` already measures at the
floor, and no fix beats the floor. Meanwhile:

- **Inter-camera skew.** Reading all three cameras sequentially costs **0.3-0.4
  ms**, not the ~200 ms the issue feared. Read ordering contributes essentially
  nothing to skew. What skew remains comes from three free-running sensors with
  unsynchronized 30 Hz phases, which no threading fixes: they are not genlocked.
- **Control-budget pressure.** The same 0.3-0.4 ms is the `o` term in #62's
  `o + s <= 1/hz` budget, against a 100 ms period. There is nothing to reclaim.

So the threaded drainer buys no measured latency, no measured skew, and no
measured budget, in exchange for three daemon threads, a lifecycle to get right
across `close()`, and a race surface on a machine that drives real arms. It is
rejected on the evidence. If a future rig shows a driver that ignores
`BUFFERSIZE`, the script re-runs and the decision reopens with new numbers.

## Design

### The fix

One line in `_opencv_camera_reader`, set **before** the format negotiation
because that is where a driver that honors the property wants it:

```python
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
```

Residual staleness is one frame interval on the D435 and under one on the D405s.
That is the capture pipeline itself, not a queue, and no software change removes
it.

### Testability

The issue's step 3, and the part that is more than one line. Today the entire
reader closure sits under `# pragma: no cover - real cameras`, which covers two
different things at once: talking to hardware, and how a capture is configured
and a frame turned into an observation. The second is ordinary logic and the
repo's 100% gate should be seeing it. A regression that silently drops the
`BUFFERSIZE` line would today be caught by nothing.

Split it, following the injection pattern `YAMEmbodiment.__init__` already uses
for the driver, clock, and operator. Three module-level helpers, each taking the
`cv2` module as a parameter so a fake stands in:

| Helper | Contract |
|---|---|
| `_configure_capture(cv2, cap)` | Cap the queue at one frame, then negotiate YUYV 640x480 and the open/read timeouts. |
| `_open_capture(cv2, device, name)` | Construct, configure, warm up; raise `RuntimeError` naming the camera if it will not open or never yields a frame. |
| `_read_rgb(cv2, cap, name, device, cfg)` | Read past transient empty frames, then BGR→RGB, resize to `cam_width` x `cam_height`, `uint8`. Raise `RuntimeError` naming the camera when every attempt fails. |

The extraction also fixes a latent bug in the current retry loop. It keeps the
last `frame` from `ok, frame = cap.read()` and then decides on `if frame is
None`, so a capture returning `ok=False` with a non-`None` frame ten times in a
row passes that guard, and a failed read reaches the policy as an observation.
`_read_rgb` tracks the success flag rather than inferring it from the frame,
which is also what makes the failure path testable at all.

The closure keeps the pragma and shrinks to the two things that genuinely need
hardware: the lazy `import cv2` and the wiring.

```python
def reader(cfg: YamConfig) -> ImageMap:  # pragma: no cover - real cameras
    import cv2

    if not caps:
        for name, device in devices.items():
            caps[name] = _open_capture(cv2, device, name)
    return {name: _read_rgb(cv2, cap, name, devices[name], cfg) for name, cap in caps.items()}
```

Behavior held constant through the refactor: lazy `cv2` import so the package
still imports without OpenCV, lazy device open on first read so construction
stays inert, the existing 10-attempt warm-up and 10-attempt read retries, and
the `RuntimeError` messages naming both camera and device path.

### Tests

Against a fake `cv2` module and fake captures, in `tests/test_embodiment.py`:

1. `_configure_capture` requests `BUFFERSIZE=1`. The regression guard for the
   actual fix.
2. It requests it **before** `FOURCC` and the frame size. Order is a real part
   of the contract, and asserting the recorded call sequence is what makes it
   one.
3. `_open_capture` raises `RuntimeError` naming the camera when `isOpened()` is
   false.
4. It retries the warm-up and succeeds when an early frame is empty.
5. It raises when the warm-up never yields a frame.
6. `_read_rgb` retries past empty frames and returns the first good one.
6b. It raises when every attempt reports `ok=False` while still handing back a
   frame object, the latent bug above.
7. It raises `RuntimeError` naming camera and device when every attempt fails.
8. It converts colour, resizes to the configured dimensions, and returns
   `uint8`.

The fake cv2 records `set` calls as `(prop, value)` pairs, which is what makes
tests 1 and 2 assertions about the contract rather than about the implementation.

The sleeps in the retry paths are the module's `time.sleep`; tests monkeypatch
it so the suite does not spend real seconds on the retry cases.

### Not in scope

- **A grabber thread.** Rejected above on measurement.
- **Sibling plugins.** `franka`, `so101`, `widowx`, `unitree-g1`, and `agibot-a2`
  have their own camera readers with the same gap. Fix proven here first, ported
  after, exactly as #62 is doing for settle.
- **The `--rerun-connect` visual acceptance in the issue's step 4.** That checks
  #62 and #63 composing, which needs #62 merged. It belongs to whichever lands
  second.

## Interaction with #62

Both touch the observation path. #62 fixes **when** the frame is asked for (the
arm has arrived); this fixes **which** frame comes back. The measurement hands
#62 one number it should have: the observation's camera cost is 0.3-0.4 ms, so
`_pace()`'s shrinking-settle-window concern is dominated by policy inference
time, not by camera reads. Post that on #62 rather than waiting for a merge
order; the two changes touch different functions and should not conflict.

## Acceptance

1. `ruff check .`, `ruff format --check .`, `mypy`, `pytest --cov` at 100%.
2. A `--via-reader` mode in the script, which measures age through the package's
   own `_opencv_camera_reader` instead of through captures the script opened
   itself. Without it the script only ever proves things about configurations it
   constructed, never about the code that ships: the three configuration rows
   would stay green even if the fix were reverted. Run it on the rig after the
   change and confirm the reader now measures at the `BUFFERSIZE=1` row rather
   than the `default` row. Attach the log to the PR.
3. The unit tests pin the reader to that configuration, so CI catches a
   regression without a rig. Re-run the script after any OpenCV or kernel
   upgrade, since "the driver honors `BUFFERSIZE`" is a fact about this stack
   rather than a guarantee.
