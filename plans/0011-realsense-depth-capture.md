# 0011 — RealSense depth: librealsense replaces V4L2 per depth-enabled camera

Closes robocurve/inspect-robots-yam#70. Supersedes the design on PR #71's first
revision, which ran a librealsense depth pipeline *beside* the V4L2 colour
reader on the same physical camera. On-rig measurement (below) shows that
coexistence is impossible on the cameras that matter, so for a depth-enabled
camera the librealsense pipeline must own the device outright and serve both
colour and depth. The PR's optional-dependency structure, drain-thread
architecture, and test harness are kept.

## Problem

The rig's three RealSense cameras (D435 top, D405 left/right wrists) measure
depth in hardware, but the embodiment reads only their colour UVC nodes via
cv2/V4L2 (`_OpenCVCameraReader`). Agent policies get no metric scene
information; in five `stack_the_bowls` trials this cost ~15 LLM calls per
trial on manual scale calibration and 0/5 successes.

PR #71 rev 1 added `_RealsenseDepthReader`, which opens the same physical
camera through librealsense with colour+depth streams enabled (colour is
required by `rs.align(rs.stream.color)`), while `_OpenCVCameraReader` keeps
streaming the colour node. Both stacks drive the same `/dev/video*` nodes.

## Measurement

Contention test on the rig, 2026-07-27, librealsense 2.58.3, cv2 5.0.0, cv2
capture configured exactly as `_open_one` does (CAP_V4L2, YUYV, 640x480,
continuous drain thread). All camera nodes verified idle before each scenario.

| Scenario | D405 (wrist) | D435 (top) |
|---|---|---|
| cv2 streaming → `pipeline.start()` colour+depth (PR rev 1) | FAIL: `xioctl(VIDIOC_S_FMT) failed, errno=5 Input/output error` | FAIL: `errno=16 Device or resource busy` |
| cv2 streaming → `pipeline.start()` depth-only | FAIL: same errno=5 | OK: depth frames flow, cv2 unaffected |
| librealsense colour+depth first → cv2 open | rs streams; cv2 cannot open the node | same |

Three consequences:

1. **PR rev 1 cannot start on this rig.** The first `_observe()` with depth
   serials configured raises out of `pipeline.start()`, `_open_all`'s
   all-or-nothing rollback fires, and the trial dies.
2. **Depth-only streaming is not a fallback.** It survives on the D435, whose
   depth rides a separate USB interface, but fails on the D405, where every
   stream muxes through the depth-sensor module cv2 already holds. The D405
   wrist cameras are where depth has manipulation value, so a design that
   only works on the D435 solves the wrong camera.
3. **Whoever opens second loses**, cleanly. cv2 keeps its 30 fps through every
   failed librealsense attempt, so there is no corruption to worry about —
   only exclusivity.

A fourth finding, from the same session: librealsense identifies devices by a
*device* serial (`rs.camera_info.serial_number`, e.g. `260322271536`) that is
**not** the USB serial shown in `/dev/v4l/by-id` paths. The by-id serial is
what librealsense calls `asic_serial_number` (e.g. `255323074044`). A user
configuring depth from the same by-id path they used for `*_cam_device` — the
only serial they have ever seen — gets a device-not-found failure from
`rs.config.enable_device()` before contention even enters the picture. Rig
mapping for reference: left `255323074044→260322271536`, right
`255323074024→260322275861`, top `310323023943→261222078836`.

## Design

For each camera slot, exactly one capture stack owns the device:

- `*_cam_device` set → `_OpenCVCameraReader` (unchanged, colour only).
- `*_depth_serial` set → a new `_RealsenseCameraReader` owns the camera via
  one librealsense pipeline and serves **both** the colour frame for
  `Observation.images` and the depth+intrinsics for `Observation.extra`.
- Setting both for one camera is a config error (details under Config).

The all-or-none rule on `*_depth_serial` is dropped and replaced by the rule
under Config below. The intended rig config is D405 wrists on librealsense,
D435 top staying on cv2 — the mixed case is the primary case, not an
afterthought.

### Builtin vs injected readers — precedence, spelled out

Today (`embodiment.py:800-823`) the builtin cv2 reader is built only when
`camera_reader` is not injected, but the depth reader is built from serials
*regardless* of injection. Under this design the rs reader owns a camera's
images too, so that combination no longer has a coherent meaning: a builtin
rs reader beside an injected cv2-backed reader recreates the very V4L2
contention this plan exists to kill. New precedence:

- No `camera_reader` injected: builtin readers are constructed from config —
  cv2 for device slots, rs for serial slots, composite when mixed.
- `camera_reader` injected: **no builtin reader of either kind is built.** If
  any `*_depth_serial` is set alongside an injected `camera_reader`,
  `YAMEmbodiment.__init__` raises `ValueError` naming the conflict and the
  way out ("configured depth serials drive the builtin capture path; with a
  custom camera_reader, supply depth via depth_reader instead"). Silent
  no-depth is not an option: the serials express intent the constructor
  cannot honor.
- `depth_reader` injected (any `cfg -> dict` callable): merged into
  `Observation.extra` exactly as in rev 1, works with either camera path.
  Injecting both `camera_reader` and `depth_reader` remains the fully-custom
  escape hatch.

The existing test `tests/test_depth_reader.py`
`test_emb_auto_constructs_depth_reader_from_config` (injected `_cameras` +
all three serials, asserts a builtin depth reader appears) pins the old
semantics; it is reworked into (a) injected-reader-plus-serials asserts the
new `ValueError`, and (b) a serial-only config with **no** injected reader
asserts the builtin rs reader is constructed and observations carry depth.

### Serial resolution

`_RealsenseCameraReader._open_all` enumerates `rs.context().query_devices()`
once and matches each configured serial against **both** `serial_number` and
`asic_serial_number`. `enable_device()` is then called with the matched
device's `serial_number`. No match → `RuntimeError` listing every visible
device as `name / serial / asic_serial`, so the fix is copy-paste.

### Streams, alignment, resolution

Per camera: colour `640x480 rgb8 @30` + depth `640x480 z16 @30`, both proven
on-rig for D405 and D435 (scenario C above streamed exactly this profile).
The stream format is `rs.format.rgb8` — **not** rev 1's `bgr8`
(`embodiment.py:666`) — and the colour array is published without channel
conversion; a test pins channel order end-to-end (fake colour frame with
distinct per-channel values surviving `__call__` unswapped). Depth is
aligned to colour (`rs.align(rs.stream.color)`) in the drain thread.

Published formats, all sized to the images contract so downstream masks align
pixel-for-pixel (capx segments on `observation.images` and ships the mask
alongside this depth to GraspNet):

- image: `(cam_height, cam_width, 3) uint8` RGB — same contract as the cv2
  reader (`_convert`). The colour resize uses `cv2.resize` (default linear,
  matching the cv2 reader so image quality is uniform across mixed-rig
  cameras): the rs reader gains an optional `cv2_module` injection seam
  beside `rs_module`, falling back to the lazy `_import_cv2` — no new
  top-level import (the import-hygiene CI job requires camera modules to
  stay lazy), and the fake-namespace tests inject a fake cv2 exposing
  `resize`. Depth's nearest-neighbour resize is pure numpy (integer index
  striding), needing no cv2.
- `extra["{cam}_depth"]`: a **zero-arg callable** returning
  `(cam_height, cam_width) float32`, metres, aligned to the colour frame,
  resized with **nearest-neighbour** (interpolating metres across an
  occlusion boundary invents surfaces). The callable, not an array, is the
  published value — see Thunks below.
- `extra["{cam}_intrinsics"]`: `(3, 3) float32` K as a **plain array**
  (constant per trial, 36 bytes, and it must survive capx's
  `np.asarray → np.save(allow_pickle=False)` path), **scaled to the
  published resolution**: `fx' = fx * cam_width/640`,
  `cx' = cx * cam_width/640`, the y-row by `cam_height/480`. Distortion
  coefficients are dropped and the docs say so (D405/D435 at VGA are
  near-rectilinear).

### Thunks — semantics, spelled out

`{cam}_depth` is published as a zero-arg callable closing over
`(reader, name)`: capx's `_TurnView` resolves callables at consumption time
precisely so trial records stay small, and per-step float32 arrays would
otherwise balloon every eval log. Two consequences are part of the contract
and the docs:

- **Resolve promptly.** Each call returns a *fresh* conversion of the newest
  captured pair, so a consumer that delays resolution gets depth captured
  later than the colour image it plans against. capx resolves at
  turn-construction time, immediately after observe — that is the justifying
  assumption. The `info.docs` text states it: "resolve the depth callable
  when the observation is received."
- **Resolution after `close()` fails loudly — by generation, not by
  emptiness.** Each thunk captures the reader's generation counter at
  publication; resolution checks it against the current generation and
  raises `RuntimeError("depth for {cam} resolved after camera close")` on
  mismatch — not the stale-frame message, which would misread as a hardware
  fault. Generation capture rather than an is-closed check matters because
  the readers are deliberately reopenable (lazy open, generation counter,
  pinned for cv2 by `tests/test_camera_reader.py:351-366`): a thunk held
  across `close()` → next-trial reopen must raise, not silently resolve
  against the new trial's depth. Tests cover both the closed and the
  closed-then-reopened case.

### Failure handling

- `wait_for_frames` is replaced by `try_wait_for_frames(timeout_ms=1000)`
  **everywhere, including `_open_one`'s warm-up loop**
  (`embodiment.py:672-679`, currently a bare `except Exception: sleep`) —
  the warm-up becomes `not ok → sleep/retry`, so the fake pipeline
  implements one API and the retry branch is a plain value test rather
  than a covered except. In the drain, `not ok` → `continue`; staleness
  detection in `_latest` (`MAX_FRAME_AGE_S = 0.5`) reports a real stall
  with the device identity in the message. Genuine
  exceptions latch into `_faults` and re-raise from `_latest`, exactly like
  the cv2 reader. This deletes rev 1's substring-matching on exception text.
- The drain thread stores the newest raw pair only: z16 depth (copied out of
  the librealsense buffer — its backing store is recycled), rgb8 colour, K,
  depth scale, timestamp. Float conversion, metre scaling, and both resizes
  happen at observation rate (~10 Hz), not at 30 fps x 2 wrists.
- Lifecycle (generation counter, all-or-nothing `_open_all` with rollback,
  fresh stop event per cycle, join-before-stop close that leaks rather than
  crashes under torque) is kept verbatim from rev 1 — it is a faithful port
  of the cv2 reader's already-debugged pattern.
- No new `# pragma: no cover` on logic branches. Rev 1's pragmas on the
  timeout-retry path, fault latch, generation checks, and close-with-alive
  thread come off; the close-with-alive-thread branch is testable with a
  blocking fake wait exactly as `tests/test_camera_reader.py`'s
  `FakeCapture(block=...)` does for cv2. Pragmas stay only on
  real-dependency seams (`_import_rs`, `_import_cv2` bodies), per the
  module-header convention. CI enforces 100%.

### Config (`config.py`)

Fields keep their rev 1 names (`top/left/right_depth_serial: str | None`).
`__post_init__` replaces both existing camera rules — devices all-or-none
(`config.py:288-293`) and serials all-or-none (`config.py:294-301`) — with
two rules that preserve construction-time failure for partial rigs:

1. **Per-slot exclusivity:**

```python
for slot in ("top", "left", "right"):
    if getattr(self, f"{slot}_cam_device") is not None and (
        getattr(self, f"{slot}_depth_serial") is not None
    ):
        raise ValueError(
            f"{slot}_cam_device and {slot}_depth_serial are mutually "
            f"exclusive: a RealSense camera opened through librealsense "
            f"cannot also be opened through V4L2 (one streamer per node)"
        )
```

2. **All slots sourced, or none:** each slot's source is
   `device XOR serial`; either every slot has exactly one source (builtin
   capture, cv2/rs/mixed) or no slot has any (injected `camera_reader`).
   A config with only `top_cam_device` set — valid under a naive relaxation
   — still raises at construction with a ValueError naming the unsourced
   slots, rather than falling through to `_default_camera_reader` and dying
   at `reset()` (`embodiment.py:905-912`) with a message written for the
   nothing-configured case.

Tests pinning the old error strings are reworked in the same commit:
`tests/test_config.py:311-312` and `tests/test_depth_reader.py:364-366`
(all-or-none messages), and `tests/test_embodiment.py:385,911` (reset
fail-fast message text, which the wiring task updates to mention both
sources).

### Wiring (`YAMEmbodiment`)

- `__init__` builds, absent an injected `camera_reader`: cv2 reader over
  device slots, rs reader over serial slots, composite `CameraReader`
  merging their `ImageMap`s when both exist. The rev 1 block at
  `embodiment.py:810-823` (including its
  `assert left is not None and right is not None`, whose "validated by
  YamConfig" premise the new rules break) is replaced, and the cv2 gate no
  longer keys on `top_cam_device` alone.
- `_observe` merges the rs reader's `extra(cfg)` (depth thunks + K) and any
  injected `depth_reader`'s dict into `Observation.extra`, in that order:
  an injected `depth_reader` returning the same keys overrides the builtin
  values — explicit injection is user intent. The precedence is stated in
  the `depth_reader` parameter docs.
- `_release_cameras` keeps its duck-typed contract exactly
  (`embodiment.py:1075`: call `close()` on `self._camera_reader` if it has
  one, likewise the depth reader — the injected-reader close and interrupt
  behavior pinned by `tests/test_camera_reader.py:421-490` is untouched).
  To fit that path, the builtin composite itself exposes `close()`, which
  closes both wrapped readers. The embodiment keeps a reference to the
  builtin rs reader for the `extra()` merge, but device release flows only
  through `self._camera_reader.close()` (the composite or sole reader).
- The `info.docs` depth paragraph's gate at `embodiment.py:847-854` —
  currently `self._depth_reader is not None`, which serial-configured rigs
  would no longer set — becomes "builtin rs reader built **or**
  `depth_reader` injected". A docs-presence test covers the
  serial-configured case, mirroring the docs assertion at
  `tests/test_depth_reader.py:482`.
- The reset fail-fast message (`embodiment.py:905-912`) is rewritten to name
  both config sources (`*_cam_device` for V4L2 colour, `*_depth_serial` for
  RealSense colour+depth) alongside the injected-reader option.
- `DEVICE_SLOTS` (`embodiment.py:768-778`): the comment "Cameras mirror
  YamConfig's all-three-or-none validation" is updated to describe the
  exactly-one-source rule. The setup wizard (inspect-robots#61) interviews
  the three v4l2 slots as a group and would fill all three `*_cam_device`;
  that remains correct for cv2-only rigs, and wizard support for depth
  serials (which have no probe-able `/dev` slot kind) is explicitly out of
  scope — mixed rigs hand-edit config, and the README example shows how.

### Docs

- `info.docs`: rewrite the depth paragraph to match the published reality —
  `{cam}_depth` is a zero-arg callable returning H x W float32 metres **at
  the same resolution as, and aligned to, that camera's image**, to be
  resolved when the observation is received; `{cam}_intrinsics` is a plain
  3x3 K matrix valid for that resolution. This text is what the LLM policy
  reads; rev 1's text described a dict that was never published.
- Class docstring: "RealSense" not "D405-only"; the two serial namespaces
  and that either is accepted.
- `config.py:190-194` field comments: currently describe the superseded
  beside-V4L2 design ("alongside the V4L2 colour frames", "Set all three to
  enable") — rewritten for ownership semantics and per-slot rules. These
  comments are what `-E` CLI users read.
- README: install extra, a config example for the mixed rig (wrists via
  serial, top via device), where to find serials (`rs-enumerate-devices` or
  the ASIC serial embedded in `/dev/v4l/by-id` names), first-`reset()`
  warm-up cost, the one-streamer-per-node constraint. (The repo keeps no
  CHANGELOG — release notes live in GitHub releases; nothing to add there
  in this PR.)

### Out of scope

Publishing depth to `/act` VLA servers (wire ships `images`+`state` only),
point-cloud helpers, D435 IR streams, per-camera resolution overrides,
setup-wizard interviews for depth serials.

## Implementation tasks

Ordered; each lands with its tests green (`uv run pytest`, mypy strict, both
ruff jobs, 100% coverage) and commits separately on the PR branch. Config
validation and embodiment wiring land **in one commit** (task 2): relaxing
validation first would leave `embodiment.py:810-813`'s assert and the
`top_cam_device`-keyed gate accepting configs they mishandle, with no test
catching the gap between commits.

1. **`_RealsenseCameraReader`**: rework `_RealsenseDepthReader` — rename;
   serial resolution against both namespaces with the enumerated error;
   `rs.format.rgb8` streams; `try_wait_for_frames` drain; raw-pair slot
   (rgb8 colour, z16 depth, K, scale, ts); `__call__(cfg) -> ImageMap`
   doing the cv2 colour resize (with the `cv2_module` seam);
   `extra(cfg) -> dict[str, Any]` returning generation-capturing depth
   thunks + scaled plain-array K; pragmas off logic branches (blocking-fake
   for close-with-alive-thread). **In the same commit, the embodiment's
   rev 1 auto-construction block (`embodiment.py:810-823`) is deleted**:
   serial-configured rigs get no builtin depth from this intermediate
   commit (task 2 restores it with the new wiring), because the renamed
   class returning an `ImageMap` from `__call__` no longer fits the
   `depth_reader` slot that block feeds — leaving it would merge uint8
   images into `extra`. `test_emb_auto_constructs_depth_reader_from_config`
   (tests/test_depth_reader.py:463-483) is removed in this commit too;
   task 2 introduces its two successors. Injected `depth_reader` behavior
   is untouched. The fake `rs` namespace grows `context()` with
   `query_devices` (devices exposing both serials), `try_wait_for_frames`,
   colour frames with `get_data`; a fake cv2 exposing `resize` is injected.
   Channel-order test. Rev 1's `_PublishedDepth(intrinsics={})` test
   construction disappears with the raw-pair slot rewrite. Tests for:
   happy path images+extra, either-serial matching, no-match error listing,
   K scaling math, nearest-neighbour depth resize, thunk freshness (two
   calls, two framesets), post-close and close-then-reopen thunk errors,
   timeout retry, fault latch, generation gating, rollback, close
   idempotence.
2. **Config + wiring (one commit)**: the two new validation rules replacing
   both all-or-none rules; embodiment precedence (injected `camera_reader`
   suppresses builtins, serials+injected-reader ValueError); composite
   reader; `_observe` extra merge; `_release_cameras`; docs gate; reset
   fail-fast message; `DEVICE_SLOTS` comment. Rework the pinned tests named
   above (config error strings, `test_emb_auto_constructs_depth_reader_from_config`
   split, reset-message matchers). New tests: mixed-rig observe returns all
   three images + wrist depth thunks; serial-only rig; docs presence on
   serial-configured rig; both-set per-slot error; partial-rig
   construction-time error; injected-reader-plus-serials error; injected
   `depth_reader` still merged on both camera paths, and its keys override
   the builtin's. Test seam for these embodiment-level tests (the builtin
   readers construct with `cv2_module=None`/`rs_module=None` and would hit
   real imports in CI): monkeypatch `embodiment._import_cv2` and
   `embodiment._import_rs` to return the fake namespaces — the readers
   fall back to those functions exactly when the module attribute is None.
3. **Docs + lock**: `info.docs` text, class docstring, config field
   comments, README; revert the unrelated `exceptiongroup`
   marker churn in `uv.lock` (re-lock minimally so the diff touches only
   the `depth` extra).
4. **Hardware validation on the rig** (maintainer, not CI): mixed config
   with both D405 serials + top via device; one full observe cycle
   delivering three images + two depth maps with plausible metric values at
   known distances; repeat with all three cameras on librealsense to
   qualify the D435 path; confirm no EBUSY/errno=5 and cv2 top coexists
   with librealsense wrists. Attach results to the PR before re-review.

## Revision notes

Rev 1's `_drain` classified timeouts by substring-matching exception text.
It happened to work, but `try_wait_for_frames` exists precisely to make the
timeout a value instead of an exception, and the substring test would have
silently reclassified a fatal error whose message mentioned "timeout" as
retryable, spinning the drain thread on a dead device until staleness fired
with no cause attached. The API change also lets the fake `rs` express
timeouts without manufacturing exception strings, which is why the tests get
simpler rather than longer.

Plan rev 2, after fresh-context critique: injected-reader precedence made
explicit (was undefined, and an existing test pinned the old semantics);
`{cam}_depth`-as-thunk carried consistently into the published-formats
section and `info.docs` (rev 1 of this plan contradicted itself); thunk
temporal and post-close semantics specified; partial-rig configs kept
failing at construction (a naive relaxation deferred them to a misleading
reset()-time error); config validation and embodiment wiring fused into one
commit (separately, the intermediate tree mishandles newly-valid configs);
rgb8-vs-bgr8 pinned with a channel-order test; config field comments,
`DEVICE_SLOTS`, and the setup-wizard story brought into the docs scope.

Plan rev 3, after second critique: task 1's intermediate tree made coherent
(the rev 1 auto-construction block and its pinning test go in task 1's
commit — the renamed reader no longer fits the slot that block feeds);
post-close thunk detection pinned to generation capture (an is-closed check
would let a thunk held across close→reopen silently resolve against the
next trial's depth); builtin-vs-injected `extra` merge order specified
(injected wins); colour-resize seam specified (`cv2_module` injection,
lazy import preserved); dropped a wrong mypy-on-tests justification (mypy's
scope is src only, pyproject.toml:101).

Plan rev 4, after third critique (no blockers remained): dropped the
CHANGELOG line item (the repo keeps none — release notes live in GitHub
releases); named the embodiment-level test seam (monkeypatch
`_import_cv2`/`_import_rs`); warm-up loop switched to `try_wait_for_frames`
alongside the drain; `_release_cameras` disposition fixed as
composite-exposes-`close()` so the duck-typed contract and its pinned tests
stay untouched.
