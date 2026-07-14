"""``inspect-robots-yam-holdcheck`` — the on-rig hold-behavior verification.

Slow policies (VLA servers, LLM agents) leave multi-second gaps between
action chunks; during a gap no command reaches the motors, and the arm must
hold its last commanded pose. This script measures exactly that: it commands
the arm's *current* pose once (so no motion is expected), then samples joint
positions for a while and reports the drift.

Run it per arm and per mode, arms mid-workspace, e-stop in hand::

    inspect-robots-yam-holdcheck can0 --zero-gravity false
    inspect-robots-yam-holdcheck can1 --zero-gravity false
    inspect-robots-yam-holdcheck can0 --zero-gravity true
    inspect-robots-yam-holdcheck can1 --zero-gravity true

PASS in the mode you run agents in closes verification item 6.4 of the
inspect-robots plan-0008 quickstart. If gravity-comp mode (``true``) drifts
but stiff mode (``false``) holds, run agents with
``-E zero_gravity_mode=false``. If both drift, file an issue with the
numbers: the embodiment needs a hold heartbeat.

The robot handle, sleep, and output are injected so the whole module tests
without hardware; the real i2rt connection is a pragma'd default.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from inspect_robots_yam.embodiment import _load_i2rt

DEFAULT_SETTLE_RAD = 0.05
DEFAULT_TREND_RAD = 0.01
DEFAULT_DURATION_S = 60.0
DEFAULT_INTERVAL_S = 5.0


class SingleArm(Protocol):
    """The one-arm slice of the i2rt driver this check needs."""

    def get_joint_pos(self) -> npt.NDArray[np.floating[Any]]:
        """Read one arm pose in driver-native units."""
        ...

    def command_joint_pos(self, target: npt.NDArray[np.floating[Any]]) -> None:
        """Command a pose in the same units returned by ``get_joint_pos``."""
        ...


RobotFactory = Callable[[str, bool], SingleArm]
EmitFn = Callable[[str], None]


def _print_flushed(line: str) -> None:
    """Default emit: flush per line so output streams over ssh pipes."""
    print(line, flush=True)


def _default_robot_factory(  # pragma: no cover - real hardware
    channel: str, zero_gravity_mode: bool
) -> SingleArm:
    get_yam_robot, GripperType = _load_i2rt()

    robot: SingleArm = get_yam_robot(
        channel=channel,
        gripper_type=GripperType["LINEAR_4310"],
        zero_gravity_mode=zero_gravity_mode,
    )
    return robot


@dataclass(frozen=True)
class HoldResult:
    """The verdict plus the per-interval drift history for the report.

    Two distinct signals: ``settle`` is the first-sample drift (a one-time
    steady-state control offset right after the command — benign if small),
    while ``trend`` is how much further the arm moved after settling
    (accumulating sag/walk-off — the dangerous one). Rig data shows the
    settle is mode-independent (~0.012-0.015 rad on YAM joint 3), so the two
    need separate thresholds.
    """

    max_drift: float
    settle: float
    worst_joint: int
    settle_rad: float
    trend_rad: float
    samples: tuple[tuple[float, float], ...]  # (elapsed_s, max_abs_drift)

    @property
    def trend(self) -> float:
        """Measure drift growth beyond the first sample, floored at zero radians."""
        return max(0.0, self.max_drift - self.settle)

    @property
    def passed(self) -> bool:
        """Require settle and trend to stay within their independent radian limits."""
        return self.settle <= self.settle_rad and self.trend <= self.trend_rad


def run_hold_check(
    robot: SingleArm,
    *,
    duration_s: float = DEFAULT_DURATION_S,
    interval_s: float = DEFAULT_INTERVAL_S,
    settle_rad: float = DEFAULT_SETTLE_RAD,
    trend_rad: float = DEFAULT_TREND_RAD,
    sleep_fn: Callable[[float], None] = time.sleep,
    emit: EmitFn = _print_flushed,
) -> HoldResult:
    """Command the current pose once, then watch drift for ``duration_s``."""
    pose = np.asarray(robot.get_joint_pos(), dtype=np.float64)
    emit(f"start pose: {np.round(pose, 3).tolist()}")
    robot.command_joint_pos(pose)  # one command, like a chunk ending

    samples: list[tuple[float, float]] = []
    max_drift = 0.0
    worst_joint = 0
    elapsed = 0.0
    while elapsed < duration_s:
        sleep_fn(interval_s)
        elapsed += interval_s
        drift = np.asarray(robot.get_joint_pos(), dtype=np.float64) - pose
        step_max = float(np.abs(drift).max())
        samples.append((elapsed, step_max))
        if step_max > max_drift:
            max_drift = step_max
            worst_joint = int(np.argmax(np.abs(drift)))
        emit(
            f"{elapsed:5.0f}s  max |drift| = {step_max:.4f} rad"
            f"  (joint {int(np.argmax(np.abs(drift)))}: {drift[np.argmax(np.abs(drift))]:+.4f})"
        )
    return HoldResult(
        max_drift=max_drift,
        settle=samples[0][1] if samples else 0.0,
        worst_joint=worst_joint,
        settle_rad=settle_rad,
        trend_rad=trend_rad,
        samples=tuple(samples),
    )


def _parse_bool(text: str) -> bool:
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")


def main(
    argv: list[str] | None = None,
    *,
    robot_factory: RobotFactory | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    emit: EmitFn = _print_flushed,
) -> int:
    """CLI entry point. Exit 0 on PASS, 1 on FAIL."""
    parser = argparse.ArgumentParser(
        prog="inspect-robots-yam-holdcheck",
        description="Verify the arm holds position between action chunks (item 6.4).",
    )
    parser.add_argument("channel", help="CAN channel of the arm to test (can0 / can1)")
    parser.add_argument(
        "--zero-gravity",
        type=_parse_bool,
        required=True,
        metavar="true|false",
        help="driver mode to test; run both, agent runs use whichever passes",
    )
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--interval-s", type=float, default=DEFAULT_INTERVAL_S)
    parser.add_argument(
        "--settle-rad",
        type=float,
        default=DEFAULT_SETTLE_RAD,
        help="max acceptable one-time settle (first-sample drift)",
    )
    parser.add_argument(
        "--trend-rad",
        type=float,
        default=DEFAULT_TREND_RAD,
        help="max acceptable drift GROWTH after the first sample (sag/walk-off)",
    )
    args = parser.parse_args(argv)

    factory = robot_factory if robot_factory is not None else _default_robot_factory
    emit(f"{args.channel} zero_gravity={args.zero_gravity}: watching for {args.duration_s:.0f}s")
    robot = factory(args.channel, args.zero_gravity)
    try:
        result = run_hold_check(
            robot,
            duration_s=args.duration_s,
            interval_s=args.interval_s,
            settle_rad=args.settle_rad,
            trend_rad=args.trend_rad,
            sleep_fn=sleep_fn,
            emit=emit,
        )
    finally:
        # Release the motor chain: i2rt's receive thread is non-daemon, so a
        # never-closed handle keeps the process alive after the verdict AND
        # holds the CAN channel, wedging the next connection until a power
        # cycle. Verified the hard way on a real rig.
        closer = getattr(robot, "close", None)
        if callable(closer):
            closer()
    verdict = "PASS" if result.passed else "FAIL"
    emit(
        f"{verdict}: settle {result.settle:.4f} rad (limit {result.settle_rad}), "
        f"trend {result.trend:.4f} rad (limit {result.trend_rad}), "
        f"worst joint {result.worst_joint}"
    )
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
