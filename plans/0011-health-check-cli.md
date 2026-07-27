# 0011: `inspect-robots-yam-health` — one-shot rig health check (issue #72)

## Motivation

There is no quick "is the rig alive?" command. `inspect-robots-yam-preflight`
is static (contract check, no hardware); `inspect-robots-yam-holdcheck`
measures hold drift only, per arm, and takes tens of seconds. Before launching
an eval (or from cron) we want a single command that proves in a few seconds:
all three cameras deliver fresh frames, and every joint on both arms reads a
sane position.

## CLI

New console script, third entry in `[project.scripts]`:

```
inspect-robots-yam-health = "inspect_robots_yam.health:main"
```

Usage:

```
inspect-robots-yam-health \
  --top-cam /dev/v4l/by-id/...-top --left-cam ... --right-cam ... \
  [--out health.jpg] [--json] [-E key=value ...]
```

- Camera devices: all three or none (reuses `YamConfig.__post_init__`
  validation). If none are given the camera check is SKIPPED and reported as
  such (some rigs configure cameras only in eval configs); skipping is not a
  failure, but a note is printed so a typo'd flag cannot silently pass.
- `-E key=value` scalars are forwarded to `YamConfig.from_kwargs` (same
  mechanism the framework CLI uses), so `left_channel`, `zero_gravity_mode`,
  etc. stay reachable without dedicated flags.
- `--out` (default `health.jpg`): montage destination. Written only when the
  camera check runs.
- `--json`: machine-readable report instead of the human table.
- Exit code: 0 iff every executed check passed (skipped camera check does not
  fail the run; any FAULT joint or dead camera exits 1, hardware/driver errors
  exit 2 with the exception message).

## Checks

### Cameras (when devices are configured)

Reuse `_OpenCVCameraReader` from `embodiment.py` (same package, private is
fine): construct with a `YamConfig` carrying the three devices, call it once,
require all three names (`top_cam`, `left_cam`, `right_cam` from
`config.camera_order`) present with nonzero-variance frames (an all-black or
all-identical frame means a dead sensor behind a live V4L2 node). The reader
already enforces freshness (`MAX_FRAME_AGE_S`) and raises on a stopped camera;
we surface that per camera rather than aborting the other two: each camera is
reported OK / FAULT(reason) independently.

Montage: horizontal concat of the three frames in `camera_order`, each labeled
with its name via `cv2.putText`, written with `cv2.imwrite` (RGB→BGR swap).
The reader's `close()` is always called (drain threads hold the devices).

### Motors

Connect via the injected driver factory (default
`embodiment._default_driver_factory`), one `get_joint_pos()` call, then
`driver.close()` in a `finally`. Per-slot verdicts over the packed 14-D vector
using `packing` labels:

- 12 arm joints: FAULT if non-finite or outside `[joint_low, joint_high]`
  from the config (the same bounds `step()` clamps to).
- 2 gripper slots: FAULT only if non-finite. `get_joint_pos()` reports
  driver-native gripper units, not normalized ones, so range checks are not
  commensurable (mirrors `_ARM_SLOTS` reasoning in `embodiment.py`).

No motion is commanded. Connecting with the config's default
`zero_gravity_mode=True` keeps the arms compliant throughout.

## Module design (`src/inspect_robots_yam/health.py`)

Follow the `hold_check.py` pattern: everything injectable, real hardware
behind `# pragma: no cover` defaults.

```python
@dataclass(frozen=True)
class CheckResult:
    name: str            # "top_cam", "left/j3", "right/gripper", ...
    ok: bool
    detail: str          # "" when ok; reason otherwise

@dataclass(frozen=True)
class HealthReport:
    cameras: tuple[CheckResult, ...]   # empty when skipped
    cameras_skipped: bool
    joints: tuple[CheckResult, ...]
    montage_path: str | None

    @property
    def ok(self) -> bool: ...

def run_health(
    cfg: YamConfig,
    *,
    out_path: str | None,
    camera_reader_factory: ... = _default_camera_reader_factory,  # pragma'd
    driver_factory: DriverFactory = _default_driver_factory,       # pragma'd
    imwrite: ... = _default_imwrite,                               # pragma'd
) -> HealthReport: ...

def main(argv: list[str] | None = None, *, run: ... = run_health) -> int: ...
```

- `_format_human(report)` prints a compact table (module, ✅/❌, detail) plus
  the montage path; `--json` emits `dataclasses.asdict`.
- Joint labels come from `packing` (left/right split at `ARM_WIDTH`,
  `j0..j5` + `gripper`), so the table stays truthful if packing ever changes.

## Tests (`tests/test_health.py`)

Fakes for reader (dict of arrays; raising variants), driver (canned vectors:
all-good, NaN slot, out-of-range slot), and imwrite (records path + array).
Cover: all-pass exit 0; each FAULT class exits 1; driver exception exits 2;
cameras skipped when unconfigured (and noted in output); partial camera
config rejected; montage written only when cameras run; `--json` output
parses and round-trips verdicts; reader `close()` called even on failure.
100% line coverage, mypy strict, D1 docstrings, like every other module.

## Docs

- README: add the new CLI beside preflight/holdcheck in the tools section
  (respect the repo writing-style rules: no em dashes, minimal bold).
- `src/inspect_robots_yam/CLAUDE.md` module table: one row for `health.py`.

## Out of scope (v1)

`--watch` live refresh, Rerun streaming, camera latency measurement
(`scripts/measure_camera_latency.py` exists), hold-drift (holdcheck's job),
and any motion.
