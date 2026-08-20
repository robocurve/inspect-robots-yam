"""Capture, inspect, and safely visit named YAM start poses.

Every subcommand reads the same wizard configuration, including store-only
commands. Consequently a malformed ``config.ini`` also breaks ``pose list``;
pass ``--no-config`` to bypass that file.

The command exits 0 on success, 1 on store or hardware failure, and 2 for
usage errors. Capture and goto require an interactive terminal before hardware
is opened unless an ``OperatorIO`` seam is injected by a caller.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import socket
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone

import numpy as np
from inspect_robots._dotenv import init_dotenv

from inspect_robots_yam import embodiment, packing, poses
from inspect_robots_yam._user_config import YamDefaults, load_yam_defaults
from inspect_robots_yam.config import YamConfig
from inspect_robots_yam.operator import OperatorIO, stdin_interactive

_POSE_CONFIG_KEYS = frozenset({"pose_dir", "start_pose"})
_RAW_STRING_KEYS = _POSE_CONFIG_KEYS | frozenset(
    {
        "top_cam_device",
        "left_cam_device",
        "right_cam_device",
        "top_depth_serial",
        "left_depth_serial",
        "right_depth_serial",
    }
)
_GRIPPER_INDICES = (packing.ARM_DOF, packing.ARM_WIDTH + packing.ARM_DOF)
_ARM_INDICES = tuple(index for index in range(packing.TOTAL_DIM) if index not in _GRIPPER_INDICES)

NowFn = Callable[[], datetime]


def _parse_scalar(text: str) -> bool | int | float | str | None:
    """Parse the scalar forms accepted by CLI ``-E`` assignments."""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "none":
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _parse_extras(parser: argparse.ArgumentParser, values: list[str]) -> dict[str, object]:
    """Parse repeated ``-E key=value`` arguments with lossless pose strings."""
    extras: dict[str, object] = {}
    for assignment in values:
        if "=" not in assignment:
            parser.error(f"-E expects key=value, got {assignment!r}")
        key, value = assignment.split("=", 1)
        if not key:
            parser.error("-E config key cannot be empty")
        extras[key] = value if key in _RAW_STRING_KEYS else _parse_scalar(value)
    return extras


def _build_parser() -> argparse.ArgumentParser:
    """Build the complete pose CLI parser."""
    parser = argparse.ArgumentParser(
        prog="inspect-robots-yam-pose",
        description="Capture and manage named bimanual YAM start poses.",
    )
    parser.add_argument("--pose-dir", default=None)
    parser.add_argument("-E", dest="extras", action="append", default=[], metavar="key=value")
    parser.add_argument("--no-config", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    capture = commands.add_parser("capture", help="capture a hand-posed snapshot")
    capture.add_argument("name")
    capture.add_argument("--notes", default="")
    capture.add_argument("--force", action="store_true")
    capture.add_argument("--clamp", action="store_true")
    capture.add_argument("--park", action="store_true")

    goto = commands.add_parser("goto", help="ramp to a stored pose")
    goto.add_argument("name")
    goto.add_argument("--park", action="store_true")

    commands.add_parser("list", help="list stored poses")
    show = commands.add_parser("show", help="show one stored pose")
    show.add_argument("name")
    delete = commands.add_parser("delete", help="delete one stored pose")
    delete.add_argument("name")
    rename = commands.add_parser("rename", help="rename one stored pose")
    rename.add_argument("old")
    rename.add_argument("new")
    return parser


def _config(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    env: Mapping[str, str],
) -> YamConfig:
    """Merge wizard defaults, extras, and the explicit pose-directory flag."""
    extras = _parse_extras(parser, args.extras)
    if args.pose_dir is not None and "pose_dir" in extras:
        parser.error("pose_dir set by both --pose-dir and -E")
    defaults = (
        YamDefaults(args={}, source=None, owner=None)
        if args.no_config
        else load_yam_defaults(env, extra_keys=_POSE_CONFIG_KEYS)
    )
    values = {**defaults.args, **extras}
    if args.pose_dir is not None:
        values["pose_dir"] = args.pose_dir
    try:
        return YamConfig.from_kwargs(**values)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


def _validate_cli_names(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Route invalid positional pose names through argparse's exit-2 path."""
    names: tuple[str, ...]
    if args.command == "rename":
        names = (args.old, args.new)
    elif args.command in {"capture", "goto", "show", "delete"}:
        names = (args.name,)
    else:
        names = ()
    for name in names:
        try:
            poses.validate_pose_name(name)
        except poses.PoseStoreError as exc:
            parser.error(str(exc))


def _limit_errors(values: packing.Vec, cfg: YamConfig) -> tuple[int, ...]:
    """Return packed indices outside the configuration's joint bounds."""
    return tuple(
        int(index) for index in np.flatnonzero((values < cfg.low) | (values > cfg.high)).tolist()
    )


def _format_limit_error(
    name: str, values: packing.Vec, cfg: YamConfig, indices: tuple[int, ...]
) -> str:
    """Describe every out-of-bounds packed joint with its value and bounds."""
    details = ", ".join(
        f"{packing.DIM_LABELS[index]} (packed index {index})={values[index]} outside "
        f"[{cfg.low[index]}, {cfg.high[index]}]"
        for index in indices
    )
    return f"pose {name!r} is outside configured joint bounds: {details}"


def _ramp_driver(
    driver: embodiment.BimanualDriver,
    cfg: YamConfig,
    target: packing.Vec,
    sleep_fn: Callable[[float], None],
) -> packing.Vec:
    """Ramp a driver to one wire-shaped target with embodiment-equivalent sends."""
    start = packing.norm_grippers(
        packing.validate_dim(driver.get_joint_pos()),
        gripper_open=cfg.gripper_open,
        gripper_closed=cfg.gripper_closed,
    )
    hz = cfg.control_hz if cfg.control_hz > 0 else 10.0
    count = max(1, round(cfg.rest_secs * hz))
    sent = start
    for waypoint in embodiment.ramp_waypoints(start, target, count):
        sent = np.clip(waypoint, cfg.low, cfg.high)
        physical = packing.denorm_grippers(
            sent,
            gripper_open=cfg.gripper_open,
            gripper_closed=cfg.gripper_closed,
        )
        driver.command_joint_pos(physical)
        sleep_fn(1.0 / hz)
    return sent


def _safe_exit(
    driver: embodiment.BimanualDriver,
    cfg: YamConfig,
    io: OperatorIO,
    sleep_fn: Callable[[float], None],
    *,
    park: bool,
) -> None:
    """Optionally park behind a stand-clear gate, then gate torque release."""
    try:
        if park:
            io.wait_ready(
                "Arms will move to the rest pose - stand clear, then press Enter...",
                drain=False,
            )
            assert cfg.rest_pose is not None
            _ramp_driver(driver, cfg, np.asarray(cfg.rest_pose, dtype=np.float64), sleep_fn)
    finally:
        try:
            io.wait_ready(
                "support both arms, then press Enter to release torque",
                drain=False,
            )
        finally:
            driver.close()


def _capture(
    args: argparse.Namespace,
    cfg: YamConfig,
    driver_factory: embodiment.DriverFactory,
    io: OperatorIO,
    sleep_fn: Callable[[float], None],
    now_fn: NowFn,
    hostname_fn: Callable[[], str],
) -> int:
    """Run the zero-gravity capture workflow and its torque-safe teardown."""
    cfg = dataclasses.replace(cfg, zero_gravity_mode=True)
    if args.name.lower() == "none":
        io.output_fn(
            "warning: the eval CLI parses -E start_pose=none as unset, so this "
            "pose will be unreachable there; prefer another name"
        )
    path = poses.pose_path(cfg.pose_dir, args.name)
    if path.exists() and not args.force:
        io.output_fn(
            f"error: pose {args.name!r} already exists at {path}; pass --force to replace it"
        )
        return 1

    driver: embodiment.BimanualDriver | None = None
    park_on_exit = bool(args.park)
    result = 1
    try:
        driver = driver_factory(cfg)
        io.output_fn(
            "Arms are in gravity-comp idle. Pose them by hand, including gripper apertures."
        )
        try:
            io.wait_ready("Press Enter to snapshot...", drain=False)
        except KeyboardInterrupt:
            io.output_fn("capture aborted; no pose was written")
            park_on_exit = False
            return 1

        values = packing.norm_grippers(
            packing.validate_dim(driver.get_joint_pos()),
            gripper_open=cfg.gripper_open,
            gripper_closed=cfg.gripper_closed,
        )
        values[list(_GRIPPER_INDICES)] = np.clip(values[list(_GRIPPER_INDICES)], 0.0, 1.0)
        bad_arm = tuple(
            index
            for index in _ARM_INDICES
            if values[index] < cfg.low[index] or values[index] > cfg.high[index]
        )
        if bad_arm and not args.clamp:
            io.output_fn(f"error: {_format_limit_error(args.name, values, cfg, bad_arm)}")
            return 1
        if bad_arm:
            for index in bad_arm:
                original = float(values[index])
                changed = float(np.clip(original, cfg.low[index], cfg.high[index]))
                io.output_fn(
                    f"clamped {packing.DIM_LABELS[index]} (packed index {index}): "
                    f"{original} -> {changed}"
                )
                values[index] = changed

        pose = poses.StartPose(
            name=args.name,
            joints=tuple(float(value) for value in values),
            created_at=now_fn().isoformat(),
            notes=args.notes,
            rig=hostname_fn(),
        )
        saved = poses.save_pose(cfg.pose_dir, pose, overwrite=bool(args.force))
        io.output_fn(f"saved pose to {saved}")
        io.output_fn(f"use it with: -E start_pose={args.name}")
        result = 0
    except Exception as exc:
        io.output_fn(f"error: {exc}")
        result = 1
    finally:
        if driver is not None:
            try:
                _safe_exit(driver, cfg, io, sleep_fn, park=park_on_exit)
            except Exception as exc:
                io.output_fn(f"error during torque-off teardown: {exc}")
                result = 1
    return result


def _goto(
    args: argparse.Namespace,
    cfg: YamConfig,
    driver_factory: embodiment.DriverFactory,
    io: OperatorIO,
    sleep_fn: Callable[[float], None],
) -> int:
    """Ramp to a validated stored pose, hold for inspection, and exit safely."""
    try:
        pose = poses.load_pose(cfg.pose_dir, args.name)
        target = np.asarray(pose.joints, dtype=np.float64)
        bad = _limit_errors(target, cfg)
        if bad:
            io.output_fn(f"error: {_format_limit_error(pose.name, target, cfg, bad)}")
            return 1
    except Exception as exc:
        io.output_fn(f"error: {exc}")
        return 1

    driver: embodiment.BimanualDriver | None = None
    result = 1
    try:
        driver = driver_factory(cfg)
        io.wait_ready(
            f"Arms will move to start pose {pose.name!r} - stand clear, then press Enter...",
            drain=False,
        )
        _ramp_driver(driver, cfg, target, sleep_fn)
        measured = packing.norm_grippers(
            packing.validate_dim(driver.get_joint_pos()),
            gripper_open=cfg.gripper_open,
            gripper_closed=cfg.gripper_closed,
        )
        io.output_fn(f"final measured pose: {measured.tolist()}")
        io.wait_ready("Press Enter to finish inspecting this pose...", drain=False)
        result = 0
    except KeyboardInterrupt:
        io.output_fn("goto interrupted")
        result = 1
    except Exception as exc:
        io.output_fn(f"error: {exc}")
        result = 1
    finally:
        if driver is not None:
            try:
                _safe_exit(driver, cfg, io, sleep_fn, park=bool(args.park))
            except Exception as exc:
                io.output_fn(f"error during torque-off teardown: {exc}")
                result = 1
    return result


def _store_command(args: argparse.Namespace, cfg: YamConfig, io: OperatorIO) -> int:
    """Execute one hardware-free pose-store command."""
    try:
        if args.command == "list":
            io.output_fn("name\tcreated_at\trig\tnotes")
            for pose in poses.list_poses(cfg.pose_dir):
                io.output_fn(f"{pose.name}\t{pose.created_at}\t{pose.rig or ''}\t{pose.notes}")
        elif args.command == "show":
            pose = poses.load_pose(cfg.pose_dir, args.name)
            io.output_fn(f"name: {pose.name}")
            io.output_fn(f"created_at: {pose.created_at}")
            io.output_fn(f"rig: {pose.rig or ''}")
            io.output_fn(f"notes: {pose.notes}")
            io.output_fn(f"path: {poses.pose_path(cfg.pose_dir, pose.name)}")
            left, right = packing.split(pose.joints)
            io.output_fn(f"left: {left.tolist()}")
            io.output_fn(f"right: {right.tolist()}")
        elif args.command == "delete":
            poses.delete_pose(cfg.pose_dir, args.name)
            io.output_fn(f"deleted pose {args.name!r}")
        elif args.command == "rename":
            path = poses.rename_pose(cfg.pose_dir, args.old, args.new)
            io.output_fn(f"renamed pose {args.old!r} to {args.new!r}: {path}")
        else:
            raise AssertionError(f"unexpected store command {args.command!r}")
    except Exception as exc:
        io.output_fn(f"error: {exc}")
        return 1
    return 0


def _default_now() -> datetime:  # pragma: no cover - real wall clock
    """Return the current UTC timestamp for real CLI capture provenance."""
    return datetime.now(timezone.utc)


def main(
    argv: list[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    driver_factory: embodiment.DriverFactory | None = None,
    io: OperatorIO | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    now_fn: NowFn | None = None,
    hostname_fn: Callable[[], str] | None = None,
) -> int:
    """Run the pose CLI with injectable hardware, console, clock, and host seams."""
    if env is None:
        init_dotenv(os.environ)
    config_env = os.environ if env is None else env
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_cli_names(parser, args)
    cfg = _config(parser, args, config_env)
    if args.command in {"capture", "goto"} and args.park and cfg.rest_pose is None:
        parser.error("--park requires rest_pose to be configured")

    default_io = io is None
    if io is None:  # pragma: no cover - real console wiring
        io = OperatorIO()
    if driver_factory is None:  # pragma: no cover - real hardware wiring
        driver_factory = embodiment._default_driver_factory
    if sleep_fn is None:  # pragma: no cover - real clock wiring
        sleep_fn = time.sleep
    if now_fn is None:  # pragma: no cover - real wall-clock wiring
        now_fn = _default_now
    if hostname_fn is None:  # pragma: no cover - real host wiring
        hostname_fn = socket.gethostname

    if args.command in {"capture", "goto"} and default_io and not stdin_interactive():
        parser.error(f"{args.command} requires an interactive terminal")
    if args.command == "capture":
        return _capture(args, cfg, driver_factory, io, sleep_fn, now_fn, hostname_fn)
    if args.command == "goto":
        return _goto(args, cfg, driver_factory, io, sleep_fn)
    return _store_command(args, cfg, io)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
