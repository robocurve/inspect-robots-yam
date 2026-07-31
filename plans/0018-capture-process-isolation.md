# 0018 — RealSense capture process isolation

**Issue:** #95 · **Branch:** `fix/capture-process-isolation` · **Status:** draft (pre-critique)

## Problem

With all three camera slots on `{slot}_depth_serial`, both CAN chains report
per-motor `loss communication` at first-observation time and the i2rt robot
servers exit — two consecutive on-rig reproductions (2026-07-30, different
victim motors each time). USB, CAN integrity, bandwidth, and hardware were
ruled out at the kernel/bus level; standalone streaming of the same three
cameras at full rate alongside motor reads stayed healthy **when the cameras
ran in a separate process**.

Mechanism: the i2rt DM control loops are plain Python threads in the same
interpreter as the capture stack. The librealsense drain threads
(`_RealsenseCameraReader._drain` → `align.process` → two `np.asarray(...).copy()`
per frame per camera) plus first-observation work (resize, PNG encodes, wire
capture, rerun) contend for the GIL. A stall longer than the motor reply
deadline reads as "loss communication" — the driver can't tell "motor went
silent" from "nobody was listening." The V4L2 path has never shown this; it
does strictly less Python-side work per frame.

Fix direction (per #95): move RealSense capture out of the control process
entirely, so no camera or image work can hold the GIL the motor threads need.

## Design

### 1. New module `_capture_proc.py`: a spawn-context capture child

One child process owns **all** configured RealSense pipelines (not one child
per camera — a single child halves the process-management surface and the
per-frame work in the child is C++-dominated; it has no motor threads to
starve).

- `multiprocessing.get_context("spawn")` only. Fork is unsafe under the
  parent's live threads, and spawn is the portable behavior across the CI
  matrix (ubuntu + macos).
- The child entry does the **lazy imports** (`pyrealsense2`, nothing else
  heavy). The module's top level imports only stdlib + numpy, keeping the
  `import-hygiene` CI job green.
- Child responsibilities, mirroring `_RealsenseCameraReader` semantics
  exactly: serial resolution via `rs.context().query_devices()` matching
  `serial_number` *and* `asic_serial_number`, duplicate-device rejection,
  all-or-nothing open, per-camera 640×480 colour+depth streams,
  `rs.align(color)`, `depth_scale` read once, warm-up frames, then a drain
  loop publishing into shared memory.
- Open handshake: the parent sends the resolved config over a `Pipe`; the
  child replies `("ready", meta)` or `("error", message)` — parent raises
  the same open-failure errors it does today, within a deadline
  (`OPEN_TIMEOUT_S`, generous: real opens take seconds).
- Shutdown: parent sets a stop `Event`, joins with the existing 2.0 s
  budget, then escalates `terminate()` → `kill()`. This *improves* on the
  in-process reader, which can only leak a wedged pipeline; a wedged child
  is killable without endangering the torque-holding parent.

### 2. Shared-memory frame slots with a seqlock

Per camera, one `multiprocessing.shared_memory.SharedMemory` block laid out
as: header (`seq: uint64`, `published_s: float64`, `generation: uint64`,
`depth_scale: float64`), colour `(480, 640, 3) uint8`, depth
`(480, 640) uint16`, intrinsics `(3, 3) float32`.

- **Seqlock, no cross-process locks**: writer bumps `seq` to odd, writes
  payload, bumps to even; reader snapshots `seq`, copies, re-reads `seq`,
  retries on mismatch/odd. Bounded retries (the writer publishes at most at
  camera rate; a handful of attempts suffices), then falls through to the
  staleness path. No lock means a killed child can never leave a reader
  deadlocked.
- `published_s` is child-stamped `time.monotonic()` — on Linux and macOS
  that clock is machine-wide, so the parent's existing staleness comparison
  (`MAX_FRAME_AGE_S = 0.5`, 10 × 50 ms retry loop) carries over unchanged.
- Blocks are created, owned, and unlinked by the parent; names passed to
  the child at spawn.

### 3. Parent-side `_ProcessRealsenseCameraReader`

Drop-in replacement constructed where `_RealsenseCameraReader` is built
today (`YAMEmbodiment.__init__`), same public surface:

- `__call__(cfg)` → colour from shm, `cv2.resize` to `cam_{width,height}`
  (unchanged semantics; cv2 stays a lazy parent-side import).
- `extra(cfg)` → `{name}_intrinsics` as a plain rescaled array;
  `{name}_depth` as a **thunk** that re-reads the shm slot at resolution
  time — preserving the documented "fresh conversion of the newest captured
  frames" contract, staleness check included. Thunks carry the generation
  and raise the existing "resolved after camera close or reopen" error on
  mismatch.
- `_latest` equivalent additionally checks `child.is_alive()`; a dead child
  surfaces as the existing `frame read failed for {name}` fault shape.
- Reopen (`_ensure_open` after `close`) restarts the child and bumps the
  generation.
- The in-process `_RealsenseCameraReader` **remains** and stays the
  implementation of record for the capture semantics; the process reader
  reuses its constants (`MAX_FRAME_AGE_S`, `JOIN_TIMEOUT_S`) rather than
  redefining them.

### 4. Config: `realsense_capture` mode + `depth_fps`

- `realsense_capture: str = "process"` — `"process"` (new default) or
  `"inline"` (the current in-process reader, kept as the debugging escape
  hatch and for environments where spawn is unworkable). Validated in
  `__post_init__` with a guided error naming both values.
- `depth_fps: int = 30` — passed to both `enable_stream` calls (colour and
  depth stay locked to the same rate). Range-validated (1–90, int); invalid
  device combinations still fail at open with the librealsense error, which
  the handshake now propagates cleanly. Capture *resolution* stays 640×480
  and out of scope (as plan 0011 already decided); the K-rescale
  denominators become named constants shared by `_open_one` and `extra()`
  so a future resolution change is one edit.
- **Adjacent papercut from #95**: `{slot}_depth_serial` values arriving as
  `int` (config.ini digit strings are int-coerced by core `_parse_value`)
  are now coerced to `str` in `from_kwargs` instead of crashing with
  `AttributeError: 'int' object has no attribute 'strip'`. Serials are
  opaque digits; silent coercion is strictly friendlier than a guided
  error, and `health.py`'s `_RAW_STRING_KEYS` workaround can be retired in
  a follow-up.

### 5. Out of scope

- V4L2 path: untouched (never implicated).
- `health.py` / `watch.py` RealSense coverage: they are V4L2-only today;
  teaching them the process reader is follow-up work (noted in #95).
- Capture resolution overrides (per plan 0011's existing exclusion).
- i2rt-side retry tolerance (upstream hardening, separate conversation).
- Core `_parse_value` changes (its quoting escape hatch is documented core
  behavior; we only stop crashing on its output).

## Behavior changes on upgrade

- RGB-only rigs: none (no RealSense slots → no child process).
- RGB-D rigs: capture runs in a child process by default; observation
  contract, staleness budget, error shapes, and thunk semantics unchanged.
  `-E realsense_capture=inline` restores the old wiring exactly.
- New process appears in `ps` (`yam-capture`, via `Process(name=...)`).

## Tests

All hardware-free, preserving the 100% branch-coverage gate; real-rs lines
in the child entry take `# pragma: no cover` exactly like existing hardware
seams.

1. **Seqlock + layout unit tests**: writer/reader round-trip over a real
   `SharedMemory` block; torn-read retry (odd seq); bounded-retry
   exhaustion → staleness path.
2. **Capture-loop logic in-process**: the child's drain/publish loop is a
   plain function taking `rs_module=` — driven synchronously with the
   existing `FakeRs` family (promoted from `test_depth_reader.py` to
   `conftest.py`), covering serial resolution, dup rejection,
   all-or-nothing open, warm-up, publish, stop.
3. **True-subprocess lifecycle tests** with a **fake child entry** (test
   helper writing synthetic frames into shm): parent open handshake (ready
   / error / timeout), staleness raise on silent child, dead-child
   detection, close/join/terminate escalation, reopen generation bump.
   (Precedent for subprocess-in-tests: `test_collision.py`.)
4. **Reader-parity tests**: `_ProcessRealsenseCameraReader` produces
   identical images/extra shapes, K rescale, thunk freshness/generation
   errors as the inline reader given the same synthetic frames.
5. **Config tests**: mode validation, `depth_fps` bounds, int-serial
   coercion (the #95 crash becomes a passing test).
6. **On-rig acceptance (operator, post-merge)**: the #95 repro — 3-camera
   depth episode survives homing + first observation; re-measure plan
   0011's contention baseline.

## Implementation tasks

Ordered; each lands with pytest + mypy strict + ruff + 100% coverage green.

1. Config fields (`realsense_capture`, `depth_fps`) + int-serial coercion
   + validation tests.
2. Shared-memory slot + seqlock (`_capture_proc.py` data plane) + unit
   tests.
3. Child capture loop (rs-injectable function) + in-process tests with
   FakeRs; promote fakes to `conftest.py`.
4. Child entry + parent handshake/lifecycle + fake-entry subprocess tests.
5. `_ProcessRealsenseCameraReader` + parity tests + `YAMEmbodiment` wiring
   behind `realsense_capture`.
6. Docs: config comment blocks, CLAUDE.md capture-architecture note,
   CHANGELOG entry.

## Release

Minor bump (0.23.0). CHANGELOG under Unreleased referencing #95. No
config migration; existing profiles work unchanged (and int serials stop
crashing).
