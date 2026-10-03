# Per-camera capture size and V4L2 size verification

Issue: [#165](https://github.com/robocurve/inspect-robots-yam/issues/165).
Follows plan 0032 (#158), which is merged but not yet released.

## Problem

1. `capture_width` / `capture_height` and `depth_capture_*` are rig-wide. The
   reference rig mixes a D435 top camera (V4L2) with two D405 wrists
   (librealsense). The D405 offers at most 1280 x 720, so the README's own
   full-resolution example (1920 x 1080) fails at RealSense pipeline start for
   the wrists. Plan 0032's use case (AprilTag detection from the top camera)
   needs only the top camera at full size.
2. `_OpenCVCameraReader` requests a size with `cap.set(...)` and never reads
   back what V4L2 delivered. An unsupported request silently falls back to
   another mode; the frame is then resized to `cam_width x cam_height`, so the
   policy gets less detail than configured, stretched if the aspect differs.

## Design

### Configuration (`config.py`)

Per-slot overrides in the existing per-slot style (`top_cam_device`,
`top_depth_serial`), plain ints so `-E` and config.ini carry them:

- `{slot}_capture_width`, `{slot}_capture_height` for `slot` in
  `top`, `left`, `right`: default `None`, meaning "use the rig-wide
  `capture_width` / `capture_height`".
- `{slot}_depth_capture_width`, `{slot}_depth_capture_height`: default `None`,
  meaning "use the rig-wide `depth_capture_*`", which itself defaults to the
  slot's colour capture size.

Validation in `__post_init__`, matching the rig-wide fields: each pair is set
both-or-neither; values are ints >= 16, bools rejected.

Accessors used everywhere instead of reading the fields directly:

- `capture_size_for(camera) -> tuple[int, int]`
- `depth_capture_size_for(camera) -> tuple[int, int] | None`

`camera` is the reader-facing name (`"top_cam"`, `"left_cam"`, `"right_cam"`).
The slot is the prefix before `_cam`. An unknown name raises `ValueError`. The
existing `depth_capture_size` property stays for compatibility and returns the
rig-wide value.

New fields are added wherever the config's field lists are enumerated (e.g.
setup wizard and docs tables, if they list capture fields); the implementation
greps for `capture_width` to find them.

### Readers (`embodiment.py`, `_capture_proc.py`)

Readers take per-camera maps instead of one size. Each keeps its existing
single-size keyword for backward compatibility with direct constructors and
tests, used as the default for cameras missing from the map:

- `_OpenCVCameraReader(devices, capture_size=(640, 480), capture_sizes=None)`:
  each device opens at `capture_sizes.get(name, capture_size)`.
- `_RealsenseCameraReader(..., capture_sizes=None, depth_capture_sizes=None)`:
  `_open_one` uses the per-name colour and depth sizes.
- `_ProcessRealsenseCameraReader` / `_CaptureProcess(..., capture_sizes=None,
  depth_capture_sizes=None)`: each shared-memory slot is created at its
  camera's colour size (the child already enables colour at the slot's size).
  `_CaptureSpec` gains `depth_sizes: tuple[tuple[str, tuple[int, int] | None], ...]`
  (picklable); the child passes each camera's own depth size to
  `_open_child_pipeline`. The existing `depth_size` field is kept as the
  fallback for cameras absent from `depth_sizes`.
- Intrinsics scaling in `_RealsenseCameraReader.extra` (and the process
  reader's equivalent) divides by the camera's own capture size, not the
  rig-wide one.
- `_opencv_camera_reader(cfg)` and the RealSense construction in
  `YamEmbodiment` build the maps with `cfg.capture_size_for(name)` and
  `cfg.depth_capture_size_for(name)` for each configured camera.

### V4L2 delivered-size check

In `_OpenCVCameraReader`, after opening and setting the size, read back
`CAP_PROP_FRAME_WIDTH` / `CAP_PROP_FRAME_HEIGHT`. If they differ from the
requested size, release the capture and raise `RuntimeError` naming the camera,
device, requested and delivered sizes, with a fix hint (`v4l2-ctl
--list-formats-ext -d <device>` to list supported sizes; set
`{slot}_capture_width/height` to one of them). Failing loudly matches the
RealSense path, which already fails at pipeline start on an unsupported
profile. The default 640 x 480 is universally supported, so existing rigs are
unaffected.

### Health and watch

`health._reader_factory_for` binds each device to
`cfg.capture_size_for(name)`. The health check therefore probes each camera at
its own size and surfaces the delivered-size error per camera. Watch mode uses
the same factory, so it follows.

## Docs

- README capture-resolution section: per-camera overrides with the mixed-rig
  example (`top_capture_width = 1920`, `top_capture_height = 1080`, wrists at
  the default or up to 1280 x 720); note the V4L2 delivered-size check.
- CHANGELOG entry under Unreleased (this repo still uses `CHANGELOG.md`).
- Module docstrings / `src/inspect_robots_yam/CLAUDE.md` if they describe
  capture size.

## Tests

- Config: per-slot pairs both-or-neither, int >= 16, bool rejected;
  `capture_size_for` / `depth_capture_size_for` fall back correctly (slot over
  rig-wide over default; depth falls back to the slot's colour size when
  nothing is set); unknown camera name raises.
- OpenCV reader: per-camera sizes passed to `cap.set`; delivered-size mismatch
  raises and releases the capture; matching size passes.
- RealSense inline reader: `_open_one` enables each camera's own colour and
  depth size; intrinsics scale by the camera's own capture size.
- Capture process: slots created at per-camera sizes; spec carries per-camera
  depth sizes; child opens each pipeline with its own depth size.
- Embodiment wiring: a mixed config (top override, wrists default) builds
  readers with the right maps.
- Health: each probe uses its camera's own size.
- Gates: ruff, ruff format, strict mypy, pytest at 100% coverage.

## Files

```
plans/0033-per-camera-capture-size.md
src/inspect_robots_yam/config.py
src/inspect_robots_yam/embodiment.py
src/inspect_robots_yam/_capture_proc.py
src/inspect_robots_yam/health.py
README.md
CHANGELOG.md
tests/test_config.py
tests/test_camera_reader.py
tests/test_capture_proc.py
tests/test_depth_reader.py
tests/test_health.py
```
