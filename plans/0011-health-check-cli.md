# 0011: `inspect-robots-yam-health` — one-shot rig health check (issue #72)

Revision 5: round 1 reworked per-camera reader construction, warm-up settle,
torque-off documentation, `-E` coercion, exit code taxonomy, packing labels,
cv2 seam. Round 2 fixed the exit-2 routing for config errors, narrowed the
montage pragma to the cv2 import only, and tightened skip/settle semantics.
Round 3 unified the zero-checks rule, pinned the driver-failure report shape,
stderr routing, and pragma scope. Round 4 (no majors) pinned the uniform
threshold, FAULT detail rule, and safety-precondition wording.

## Motivation

There is no quick "is the rig alive?" command. `inspect-robots-yam-preflight`
is static (contract check, no hardware); `inspect-robots-yam-holdcheck`
measures hold drift only, per arm, and takes tens of seconds. Before launching
an eval we want a single command that proves in a few seconds: all three
cameras deliver fresh frames, and every joint on both arms reads a sane
position.

This is a pre-launch gate, run by the operator with the rig idle. It is NOT
safe to run concurrently with an eval (it opens the same V4L2 devices and
enables a second torque controller on the same CAN buses), so it must not be
pitched as a cron job in docs.

## CLI

New console script, third entry in `[project.scripts]`:

```
inspect-robots-yam-health = "inspect_robots_yam.health:main"
```

Usage:

```
inspect-robots-yam-health \
  --top-cam /dev/v4l/by-id/...-top --left-cam ... --right-cam ... \
  [--out health.jpg] [--json] [--skip-cameras] [--skip-motors] \
  [--settle-s 1.0] [--joint-epsilon 0.02] [-E key=value ...]
```

- Camera devices: all three or none (reuses `YamConfig.__post_init__`
  validation). If none are given the camera check is SKIPPED and reported as
  such; skipping is not a failure, but a note is printed so a typo'd flag
  cannot silently pass.
- `--skip-cameras` / `--skip-motors`: run half the check (e.g. motors only
  while debugging camera cabling). Skipped sections are reported as skipped,
  never as passed. The uniform rule: an invocation that would execute ZERO
  checks is a usage error (exit 2) — that covers `--skip-cameras
  --skip-motors` explicitly and equally `--skip-motors` with no cameras
  configured (auto-skip). A gate that checks nothing must not exit 0.
  `--skip-cameras` combined with a configured camera device via either route
  (`--*-cam` flag or `-E *_cam_device=`) is likewise a usage error
  (contradictory intent).
- `-E key=value` scalars are forwarded to `YamConfig.from_kwargs` after local
  coercion: `from_kwargs` does not parse strings into bools/ints/floats
  (`config.py` coerces only the float-tuple fields), and the framework's
  `parse_value` lives in a private module we will not import. `health.py`
  vendors a ~10-line `_parse_scalar` (`true`/`false` → bool, int, float,
  else string), documented as intentionally simpler than the framework's.
- Camera devices may come from `--top-cam/--left-cam/--right-cam` or
  `-E top_cam_device=...`, but not both: a key set by both routes is a usage
  error (exit 2), not a silent precedence.
- `--out` (default `health.jpg`): montage destination, written only when the
  camera check runs and at least one frame was captured.
- `--settle-s` (default 1.0, floor 0.2): auto-exposure settle between the two
  reads per camera (see below). Values below the floor are a usage error: at
  or under a frame interval (~33 ms) the drain thread may not have
  republished, both reads would return the same published frame, and a
  healthy camera would FAULT as "frozen". 0.2 s spans several frame intervals
  at any plausible fps.
- `--joint-epsilon` (default 0.02 rad): tolerance on joint-bound checks.
- `--json`: machine-readable report instead of the human table.

Exit codes: `0` all executed checks passed; `1` any FAULT (dead/uniform/
frozen camera, out-of-range or non-finite joint, driver or camera exception
during a check — the exception message becomes the FAULT detail); `2` usage
and config errors (bad flags, partial camera set, unknown `-E` key,
flag/`-E` conflict, both skips, sub-floor `--settle-s`). Config errors are
NOT argparse-native: partial cameras raise `ValueError` from
`YamConfig.__post_init__` and an unknown `-E` key raises `TypeError` from
`from_kwargs`, so `main` must wrap config construction in try/except and
route those through `parser.error()` (which exits 2). Exceptions inside the
two check bodies are caught and become FAULTs (exit 1); anything else is a
bug and may traceback. The siblings also exit 2 via argparse on bad flags;
this tool only adds new *reasons* for exit 2, which the module docstring
spells out.

## Checks

### Cameras (when devices are configured and not skipped)

`_OpenCVCameraReader.__call__` is all-or-nothing: one dead device aborts
opening every camera, and one stale camera raises out of the dict
comprehension before the others are read. Per-camera verdicts therefore
require one reader per camera: construct three `_OpenCVCameraReader`
instances, each with a single-entry `devices` mapping (the constructor
signature takes `Mapping[str, str]`; camera names come from
`config.DEFAULT_CAMERAS` — note `camera_order` is the policy-side
`ActServerConfig` field, not ours). Each camera's open/read runs inside its
own try/except, so one dead V4L2 node yields `1 FAULT / 2 OK`, not a global
error.

Per camera, two reads bracket a settle delay (injectable `sleep_fn`):

1. First `__call__` — triggers device open; the published frame is the warm-up
   read, typically dark before auto-exposure converges. Discard.
2. `sleep_fn(settle_s)`, then second `__call__` — score this frame:
   - FAULT "uniform frame" if `frame.std() < UNIFORM_STD_MAX`, a module
     constant pinned at 1.0 (8-bit pixel units): dead sensor behind a live
     node. No flag; a rig that trips this on a real scene should re-aim the
     camera, not tune the gate.
   - FAULT "frozen stream" if byte-identical to the first frame across the
     delay: the drain thread restamps identical frames as fresh, so
     `MAX_FRAME_AGE_S` alone cannot catch frozen content, while sensor noise
     makes byte-identical live frames practically impossible.
   - Reader raises (device failed to open, no fresh frame, latched drain
     fault) are caught per camera and reported FAULT with the exception
     message as the detail, the same rule the exit-code section pins. No
     re-labeling: the reader's own messages already distinguish the cases.

Every reader's `close()` is called in a `finally` (drain threads hold the
devices open).

Montage: horizontal concat of the captured frames in `DEFAULT_CAMERAS` order
(FAULTed cameras contribute a black placeholder tile so positions stay
recognizable), each labeled with its name, written to `--out`. The montage
function takes the cv2 module as a parameter defaulting through the existing
`_import_cv2` seam, exactly like `_OpenCVCameraReader`: labeling, shape
normalization, concat, and the RGB→BGR swap (`frame[..., ::-1]`, no cv2
needed) are all real logic and are tested against a fake cv2 (hoist
`FakeCv2` from `tests/test_camera_reader.py` into `tests/conftest.py`
rather than importing across test modules); only the `import cv2` itself
stays pragma'd, per the seam rule in `src/inspect_robots_yam/CLAUDE.md`.
Order constraint: label (`putText`) BEFORE the BGR slice — the swap yields a
non-contiguous view that cv2 in-place draw ops reject while `imwrite`
accepts it.

### Motors (when not skipped)

Connect via the injected driver factory (default
`embodiment._default_driver_factory`), one `get_joint_pos()` call, then
`driver.close()` in a `finally`. Per-slot verdicts over the packed 14-D
vector, labeled with `packing.DIM_LABELS` verbatim (`left_j3`,
`right_gripper`, ...):

- 12 arm joints: FAULT if non-finite or outside
  `[joint_low - eps, joint_high + eps]` with `eps = --joint-epsilon`. The
  config bounds clamp *commands*; readings from an arm resting against a hard
  stop can sit epsilon outside them, and a strict test would intermittently
  FAULT healthy parked arms.
- 2 gripper slots: FAULT only if non-finite. `get_joint_pos()` reports
  driver-native gripper units, not normalized ones, so range checks are not
  commensurable (mirrors `_ARM_SLOTS` reasoning in `embodiment.py`).

If the driver factory itself raises (CAN bus down, i2rt missing and
surfacing its guided-install error), there are no per-slot readings: the
motors section reports one synthetic section-level `CheckResult` named
`"driver"`, FAULT with the exception message as detail (exit 1). The
`finally` releases the driver only if one was constructed.

No motion is commanded, but connecting is not free of consequence:
`BimanualDriver.close()` releases the handles and motor torque drops (the
embodiment parks before releasing for exactly this reason; this tool reads
one sample and cannot know a safe rest pose, so it does not park). The module
docstring and README must state the precondition — arms at rest or supported,
e-stop in hand (note: NOT holdcheck's setup, which wants arms mid-workspace;
only the e-stop discipline is shared) — and `main` prints a one-line warning
before connecting. All human-facing side prints (this warning, the
cameras-auto-skipped note) go to stderr, so `--json` stdout stays pure,
parseable JSON. Connecting keeps the config's default
`zero_gravity_mode=True` (compliant throughout).

## Module design (`src/inspect_robots_yam/health.py`)

Follow the `hold_check.py` pattern: everything injectable, real hardware
behind `# pragma: no cover` defaults.

```python
@dataclass(frozen=True)
class CheckResult:
    name: str            # "top_cam", "left_j3", "right_gripper", ...
    ok: bool
    detail: str          # "" when ok; reason otherwise

@dataclass(frozen=True)
class HealthReport:
    cameras: tuple[CheckResult, ...]   # empty when skipped
    cameras_skipped: bool
    joints: tuple[CheckResult, ...]    # empty when skipped
    joints_skipped: bool               # named for the section, not the flag
    montage_path: str | None

    @property
    def ok(self) -> bool: ...          # skipped sections do not fail

def run_health(
    cfg: YamConfig,
    *,
    out_path: str | None,
    settle_s: float,
    joint_epsilon: float,
    skip_cameras: bool,
    skip_motors: bool,
    reader_factory: ... = _default_reader_factory,    # (name, device) -> reader; NOT pragma'd
    driver_factory: DriverFactory = _default_driver_factory,  # pragma'd (hardware)
    write_montage: ... = _default_write_montage,      # NOT pragma'd; takes cv2 via _import_cv2
    sleep_fn: Callable[[float], None] = time.sleep,
) -> HealthReport: ...

def main(argv: list[str] | None = None, *, run: ... = run_health) -> int: ...
```

Module ends with the siblings' `if __name__ == "__main__": raise
SystemExit(main())` block (already coverage-excluded by pyproject), so
`python -m inspect_robots_yam.health` works too.

- Pragma discipline, pinned: `_default_reader_factory` is inert (the reader
  constructor touches no hardware and imports no cv2; devices open on
  `__call__`), so it is tested, not pragma'd — same shape as the untagged
  `_opencv_camera_reader`. `_default_write_montage` is the real montage
  implementation with a `cv2_module` parameter defaulting through the
  already-pragma'd `_import_cv2`; there is no separate wholly-pragma'd
  default. Only `_default_driver_factory` (hardware) stays excluded.
- `_format_human(report)` prints a compact table (check, OK/FAULT/SKIPPED,
  detail) plus the montage path; `--json` emits `dataclasses.asdict` plus an
  explicit top-level `"ok"` key (`ok` is a property, so `asdict` alone drops
  the overall verdict; preflight's JSON includes `"ok"` and machine consumers
  should not have to recompute it).
- `health` is not added to `__init__.__all__`, matching the `hold_check`
  precedent (`preflight` exports are the exception, not the rule);
  `tests/test_api_snapshot.py` therefore needs no change.

## Tests (`tests/test_health.py`)

Fakes for reader factory (per-camera canned frames; raising variants; a
frozen-stream fake returning the identical array twice), driver (canned
vectors: all-good, NaN slot, out-of-range slot, epsilon-inside and
epsilon-outside the bound; a factory that raises, expecting the synthetic
`"driver"` FAULT and no close call), montage writer (records path + arrays), and
`sleep_fn` (records the settle delay). Cover: all-pass exit 0; each FAULT
class (dead device, uniform, frozen, stale, NaN joint, out-of-range joint,
driver exception) exits 1 with the right detail; skip flags report SKIPPED
and cannot pass a broken section; both skips together, `--skip-cameras` with
a configured device (flag or `-E`), a zero-checks invocation (both skips, or
`--skip-motors` with no cameras configured), sub-floor `--settle-s`, partial
camera config, and flag/`-E` conflicts all exit 2; `--json` stdout parses as
JSON even when warnings fire (warnings on stderr); `_parse_scalar`
bool/int/float/string cases;
montage logic (labels, mismatched shapes, BGR swap, placeholder tiles)
against a fake cv2; montage
written only when cameras run, with placeholder tiles for FAULTed cameras;
`--json` round-trips; every reader's `close()` called even when its read
raises. 100% line coverage, mypy strict, D1 docstrings, like every other
module.

## Docs

- README: add the new CLI beside preflight/holdcheck in the tools section,
  including the arms-at-rest/torque-off warning (respect the repo
  writing-style rules: no em dashes, minimal bold).
- `src/inspect_robots_yam/CLAUDE.md` module table: one row for `health.py`.
- No `uv lock` run needed (scripts-only pyproject change; cv2 is already a
  base dependency) and no new CI job (pytest/mypy/ruff pick the module up
  automatically; `ci-ok` unchanged).

## Out of scope (v1)

`--watch` live refresh, Rerun streaming, camera latency measurement
(`scripts/measure_camera_latency.py` exists), hold-drift (holdcheck's job),
parking/homing motion, and running concurrently with evals.
