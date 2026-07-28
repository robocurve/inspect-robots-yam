"""Camera and motor health tools for an idle YAM rig.

Run this check only with both arms at rest or supported and an e-stop in hand.
Connecting and then closing the driver drops motor torque. This is not the
mid-workspace setup used by holdcheck.

The one-shot CLI exits 0 when every executed check passes and 1 when any
hardware check reports a fault. A bare invocation checks the wizard-configured
rig; ``--no-config`` restores flag-only behavior and bypasses even malformed
config files. Depth-configured slots are reported as unchecked because this
tool checks only V4L2 colour devices; an all-depth rig still checks motors but
skips cameras, and cannot use ``--watch`` because it has no servable devices.

The CLI exits 2 for bad flags, a partial camera set, an unknown ``-E`` key, a
camera device set by both a flag and ``-E``, ``--skip-cameras`` with an explicit
camera-device or depth-serial key (wizard-configured camera keys are stripped
instead), an invocation that would execute zero checks, ``--settle-s``
non-finite or below the 0.2-second floor, ``--joint-epsilon`` non-finite or
negative, ``--watch`` combined with a skip or ``--json``, ``--watch`` without
at least one configured V4L2 device (including an all-depth rig), ``--port`` or
``--bind`` without ``--watch``, or a provided port outside 1 through 65535.
Those usage errors are routed through ``parser.error``. A watch bind failure
also returns 2 directly from ``watch.serve``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from inspect_robots_yam import embodiment, packing
from inspect_robots_yam._user_config import YamDefaults, load_yam_defaults
from inspect_robots_yam.config import DEFAULT_CAMERAS, YamConfig

UNIFORM_STD_MAX = 1.0
MIN_SETTLE_S = 0.2

_CAMERA_FLAG_KEYS = frozenset({"top_cam_device", "left_cam_device", "right_cam_device"})
_RAW_STRING_KEYS = frozenset(
    {
        "top_cam_device",
        "left_cam_device",
        "right_cam_device",
        "top_depth_serial",
        "left_depth_serial",
        "right_depth_serial",
    }
)
_CAMERA_SLOT_KEYS = frozenset(
    {
        "top_cam_device",
        "left_cam_device",
        "right_cam_device",
        "top_depth_serial",
        "left_depth_serial",
        "right_depth_serial",
    }
)
_CHANNEL_KEYS = frozenset({"left_channel", "right_channel"})
_CAMERA_SLOTS = ("top", "left", "right")
_DEPTH_UNCHECKED_REASON = "depth-configured; not checked by this tool"

Image = npt.NDArray[np.uint8]


class HealthCameraReader(Protocol):
    """A single-camera reader whose hardware lifetime is explicitly closed."""

    def __call__(self, cfg: YamConfig) -> embodiment.ImageMap:
        """Return the reader's latest frame under its configured camera name."""
        ...

    def close(self) -> None:
        """Release the camera device and any drain thread."""
        ...


ReaderFactory = Callable[[str, str], HealthCameraReader]
MontageWriter = Callable[[str, Mapping[str, Image], frozenset[str]], None]
RunHealth = Callable[..., "HealthReport"]


@dataclass(frozen=True)
class CheckResult:
    """The verdict and detail for one camera, joint, gripper, or driver check."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class UncheckedCamera:
    """A configured camera slot outside this tool's V4L2 checking scope."""

    name: str
    reason: str


@dataclass(frozen=True)
class HealthReport:
    """The immutable results of the requested camera and motor sections."""

    cameras: tuple[CheckResult, ...]
    cameras_skipped: bool
    joints: tuple[CheckResult, ...]
    joints_skipped: bool
    montage_path: str | None
    unchecked_cameras: tuple[UncheckedCamera, ...] = ()

    @property
    def ok(self) -> bool:
        """Pass when every executed check passes; skipped sections are neutral."""
        return all(result.ok for result in (*self.cameras, *self.joints))


def _default_reader_factory(name: str, device: str) -> HealthCameraReader:
    """Build one inert OpenCV reader for one named device."""
    return embodiment._OpenCVCameraReader({name: device})


def _default_write_montage(
    out_path: str,
    frames: Mapping[str, Image],
    faulted: frozenset[str],
    cv2_module: Any | None = None,
) -> None:
    """Write labeled, shape-normalized camera tiles in canonical RGB camera order."""
    cv2 = cv2_module if cv2_module is not None else embodiment._import_cv2()
    reference = next(frames[name] for name in DEFAULT_CAMERAS if name in frames)
    height, width = reference.shape[:2]
    tiles: list[Image] = []
    checked_names = tuple(name for name in DEFAULT_CAMERAS if name in frames or name in faulted)
    for name in checked_names:
        tile: Image
        if name in faulted:
            tile = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            tile = np.asarray(frames[name], dtype=np.uint8)
            if tile.shape != (height, width, 3):
                tile = np.asarray(cv2.resize(tile, (width, height)), dtype=np.uint8)
            tile = tile.copy()
        cv2.putText(
            tile,
            name,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        tiles.append(tile)
    montage = np.concatenate(tiles, axis=1)
    if not cv2.imwrite(out_path, montage[..., ::-1]):
        raise RuntimeError(f"failed to write montage to {out_path}")


def _camera_devices(cfg: YamConfig) -> tuple[tuple[str, str], ...]:
    devices = (cfg.top_cam_device, cfg.left_cam_device, cfg.right_cam_device)
    return tuple(
        (name, device)
        for name, device in zip(DEFAULT_CAMERAS, devices, strict=True)
        if device is not None
    )


def _unchecked_depth_cameras(cfg: YamConfig) -> tuple[UncheckedCamera, ...]:
    """Return configured RealSense slots that this V4L2-only tool cannot check."""
    serials = (cfg.top_depth_serial, cfg.left_depth_serial, cfg.right_depth_serial)
    return tuple(
        UncheckedCamera(name=name, reason=_DEPTH_UNCHECKED_REASON)
        for name, serial in zip(DEFAULT_CAMERAS, serials, strict=True)
        if serial is not None
    )


def _run_cameras(
    cfg: YamConfig,
    *,
    settle_s: float,
    reader_factory: ReaderFactory,
    sleep_fn: Callable[[float], None],
) -> tuple[tuple[CheckResult, ...], dict[str, Image]]:
    results: list[CheckResult] = []
    captured: dict[str, Image] = {}
    for name, device in _camera_devices(cfg):
        detail = ""
        try:
            reader = reader_factory(name, device)
            try:
                first = np.asarray(reader(cfg)[name], dtype=np.uint8)
                captured[name] = first
                sleep_fn(settle_s)
                second = np.asarray(reader(cfg)[name], dtype=np.uint8)
                captured[name] = second
                if float(second.std()) < UNIFORM_STD_MAX:
                    detail = "uniform frame"
                elif np.array_equal(second, first):
                    detail = "frozen stream"
            finally:
                reader.close()
        except Exception as exc:
            detail = str(exc)
        results.append(CheckResult(name=name, ok=not detail, detail=detail))
    return tuple(results), captured


def _run_motors(
    cfg: YamConfig,
    *,
    joint_epsilon: float,
    driver_factory: embodiment.DriverFactory,
) -> tuple[CheckResult, ...]:
    try:
        driver = driver_factory(cfg)
        try:
            positions = packing.validate_dim(driver.get_joint_pos())
        finally:
            driver.close()
    except Exception as exc:
        return (CheckResult(name="driver", ok=False, detail=str(exc)),)

    grippers = {packing.ARM_DOF, packing.ARM_WIDTH + packing.ARM_DOF}
    results: list[CheckResult] = []
    for index, (name, value) in enumerate(zip(packing.DIM_LABELS, positions, strict=True)):
        if not np.isfinite(value):
            detail = "non-finite"
        elif index not in grippers and not (
            cfg.joint_low[index] - joint_epsilon <= value <= cfg.joint_high[index] + joint_epsilon
        ):
            detail = (
                f"outside [{cfg.joint_low[index] - joint_epsilon}, "
                f"{cfg.joint_high[index] + joint_epsilon}]"
            )
        else:
            detail = ""
        results.append(CheckResult(name=name, ok=not detail, detail=detail))
    return tuple(results)


def run_health(
    cfg: YamConfig,
    *,
    out_path: str | None,
    settle_s: float,
    joint_epsilon: float,
    skip_cameras: bool,
    skip_motors: bool,
    reader_factory: ReaderFactory = _default_reader_factory,
    driver_factory: embodiment.DriverFactory = embodiment._default_driver_factory,
    write_montage: MontageWriter = _default_write_montage,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> HealthReport:
    """Run the requested checks once and release every constructed hardware handle."""
    cameras_configured = bool(_camera_devices(cfg))
    cameras_skipped = skip_cameras or not cameras_configured
    unchecked_cameras = _unchecked_depth_cameras(cfg)
    camera_results: tuple[CheckResult, ...] = ()
    captured: dict[str, Image] = {}
    montage_path: str | None = None
    if not cameras_skipped:
        camera_results, captured = _run_cameras(
            cfg,
            settle_s=settle_s,
            reader_factory=reader_factory,
            sleep_fn=sleep_fn,
        )
        if out_path is not None and captured:
            faulted = frozenset(result.name for result in camera_results if not result.ok)
            # A typo'd --out is operator input; it must not traceback past the
            # motors section, so a failed write becomes one more camera FAULT.
            try:
                write_montage(out_path, captured, faulted)
                montage_path = out_path
            except Exception as exc:
                camera_results = (
                    *camera_results,
                    CheckResult(name="montage", ok=False, detail=str(exc)),
                )

    joint_results = (
        ()
        if skip_motors
        else _run_motors(
            cfg,
            joint_epsilon=joint_epsilon,
            driver_factory=driver_factory,
        )
    )
    return HealthReport(
        cameras=camera_results,
        cameras_skipped=cameras_skipped,
        joints=joint_results,
        joints_skipped=skip_motors,
        montage_path=montage_path,
        unchecked_cameras=unchecked_cameras,
    )


def _parse_scalar(text: str) -> bool | int | float | str:
    """Parse basic CLI scalars, intentionally more simply than the framework parser."""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _format_human(report: HealthReport) -> str:
    lines = ["name                  status   detail"]
    if report.cameras_skipped:
        detail = (
            " configured slots are depth-configured; not checked by this tool"
            if report.unchecked_cameras
            else ""
        )
        lines.append(f"cameras               SKIPPED {detail}".rstrip())
    else:
        lines.extend(
            f"{result.name:<21} {'OK' if result.ok else 'FAULT':<8} {result.detail}"
            for result in report.cameras
        )
    if report.joints_skipped:
        lines.append("motors                SKIPPED")
    else:
        lines.extend(
            f"{result.name:<21} {'OK' if result.ok else 'FAULT':<8} {result.detail}"
            for result in report.joints
        )
    lines.append(f"montage: {report.montage_path or '(not written)'}")
    return "\n".join(lines)


def _parse_extras(parser: argparse.ArgumentParser, values: list[str]) -> dict[str, object]:
    extras: dict[str, object] = {}
    for assignment in values:
        if "=" not in assignment:
            parser.error(f"-E expects key=value, got {assignment!r}")
        key, value = assignment.split("=", 1)
        if not key:
            parser.error("-E config key cannot be empty")
        extras[key] = value if key in _RAW_STRING_KEYS else _parse_scalar(value)
    return extras


def main(
    argv: list[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    run: RunHealth = run_health,
) -> int:
    """Run the CLI, returning 0 for healthy, 1 for faults, or exiting 2 on usage errors."""
    parser = argparse.ArgumentParser(
        prog="inspect-robots-yam-health",
        description="Check the idle rig once or serve configured cameras for aiming.",
    )
    parser.add_argument("--top-cam", dest="top_cam_device")
    parser.add_argument("--left-cam", dest="left_cam_device")
    parser.add_argument("--right-cam", dest="right_cam_device")
    parser.add_argument("--out", default="health.jpg")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--skip-cameras", action="store_true")
    parser.add_argument("--skip-motors", action="store_true")
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="ignore wizard-configured devices and use only flags and builtins",
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--bind", default=None)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--joint-epsilon", type=float, default=0.02)
    parser.add_argument("-E", dest="extras", action="append", default=[], metavar="key=value")
    args = parser.parse_args(argv)

    if not math.isfinite(args.settle_s) or args.settle_s < MIN_SETTLE_S:
        parser.error(f"--settle-s must be finite and at least {MIN_SETTLE_S}")
    if not math.isfinite(args.joint_epsilon) or args.joint_epsilon < 0:
        parser.error("--joint-epsilon must be finite and non-negative")

    extras = _parse_extras(parser, args.extras)
    flag_values = {
        key: getattr(args, key) for key in _CAMERA_FLAG_KEYS if getattr(args, key) is not None
    }
    conflicts = _CAMERA_FLAG_KEYS & extras.keys() & flag_values.keys()
    if conflicts:
        parser.error(f"camera device set by both flag and -E: {sorted(conflicts)}")
    if args.watch and (args.skip_cameras or args.skip_motors or args.json):
        parser.error("--watch cannot be combined with --skip-cameras, --skip-motors, or --json")
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not args.watch and (args.port is not None or args.bind is not None):
        parser.error("--port and --bind require --watch")
    if args.skip_cameras and (_CAMERA_SLOT_KEYS & (extras.keys() | flag_values.keys())):
        parser.error(
            "--skip-cameras cannot be combined with explicit camera devices or depth serials"
        )

    yam_defaults = (
        YamDefaults(args={}, source=None, owner=None)
        if args.no_config
        else load_yam_defaults(os.environ if env is None else env)
    )
    config_args = dict(yam_defaults.args)
    if args.skip_cameras:
        for key in _CAMERA_SLOT_KEYS:
            config_args.pop(key, None)
    if args.skip_motors:
        for key in _CHANNEL_KEYS:
            config_args.pop(key, None)

    explicit_keys = extras.keys() | flag_values.keys()
    explicit_camera_keys = _CAMERA_SLOT_KEYS & explicit_keys
    for slot in _CAMERA_SLOTS:
        slot_keys = {f"{slot}_cam_device", f"{slot}_depth_serial"}
        if slot_keys & explicit_camera_keys:
            for key in slot_keys:
                config_args.pop(key, None)

    contributed_keys = config_args.keys() - explicit_keys
    config_values = {**config_args, **extras, **flag_values}
    if contributed_keys:
        print(
            f"devices: from {yam_defaults.source} (embodiment {yam_defaults.owner})",
            file=sys.stderr,
        )
    try:
        cfg = YamConfig.from_kwargs(**config_values)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    cameras_configured = bool(_camera_devices(cfg))
    if args.watch and not cameras_configured:
        if _unchecked_depth_cameras(cfg):
            parser.error(
                "--watch requires configured V4L2 camera devices; configured slots "
                "are depth-configured and not served by this tool"
            )
        parser.error("--watch requires configured camera devices")
    if args.watch:
        for key, value in (("cam_width", 640), ("cam_height", 480)):
            if key not in config_values:
                cfg = dataclasses.replace(cfg, **{key: cast(Any, value)})
        from inspect_robots_yam import watch

        port = args.port if args.port is not None else 8807
        bind = args.bind if args.bind is not None else "0.0.0.0"
        bind_was_explicit = args.bind is not None
        return watch.serve(
            cfg,
            port=port,
            bind=bind,
            bind_was_explicit=bind_was_explicit,
        )

    cameras_will_run = cameras_configured and not args.skip_cameras
    motors_will_run = not args.skip_motors
    if not cameras_will_run and not motors_will_run:
        parser.error("invocation would execute zero checks")

    unchecked_cameras = _unchecked_depth_cameras(cfg)
    if not args.skip_cameras:
        for unchecked in unchecked_cameras:
            print(
                f"{unchecked.name}: skipped ({unchecked.reason})",
                file=sys.stderr,
            )
        if not cameras_configured:
            if unchecked_cameras:
                print(
                    "note: camera checks are skipped because configured slots "
                    "are depth-configured; not checked by this tool",
                    file=sys.stderr,
                )
            else:
                print(
                    "note: no camera devices configured; camera checks are skipped",
                    file=sys.stderr,
                )
    if motors_will_run:
        print(
            "warning: arms must be at rest or supported; closing the driver drops motor torque",
            file=sys.stderr,
        )

    report = run(
        cfg,
        out_path=args.out,
        settle_s=args.settle_s,
        joint_epsilon=args.joint_epsilon,
        skip_cameras=args.skip_cameras,
        skip_motors=args.skip_motors,
    )
    if args.json:
        payload = dataclasses.asdict(report)
        payload["ok"] = report.ok
        print(json.dumps(payload, indent=2))
    else:
        print(_format_human(report))
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
